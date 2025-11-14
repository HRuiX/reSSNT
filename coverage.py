import pandas as pd
import numpy as np
import torch
import os
from pyflann import FLANN
import torch.nn.functional as F
import copy
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde


def compare_dict_tensors(dict1, dict2):
    if dict1.keys() != dict2.keys():
        return False

    for key in dict1:
        val1, val2 = dict1[key], dict2[key]
        if isinstance(val1, list) and isinstance(val2, list):
            if len(val1) != len(val2):
                return False
            for v1, v2 in zip(val1, val2):
                if isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
                    if not torch.equal(v1, v2):
                        return False
                elif v1 != v2:
                    return False
        elif isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
            if not torch.equal(val1, val2):
                return False
        elif val1 != val2:
            return False

    return True


def scale(out, dim=-1, rmax=1, rmin=0):
    out_max = out.max(dim)[0].unsqueeze(dim)
    out_min = out.min(dim)[0].unsqueeze(dim)
    output_std = (out - out_min) / (out_max - out_min)
    output_scaled = output_std * (rmax - rmin) + rmin
    return output_scaled


class Estimator(object):
    def __init__(self, feature_num, num_class=1):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_class = num_class
        self.CoVariance = torch.zeros(num_class, feature_num, feature_num).to(self.device)
        self.Ave = torch.zeros(num_class, feature_num).to(self.device)
        self.Amount = torch.zeros(num_class).to(self.device)
        self.CoVarianceInv = torch.zeros(num_class, feature_num, feature_num).to(self.device)

    def calculate(self, features, labels=None):
        N = features.size(0)
        C = self.num_class
        A = features.size(1)

        if labels is None:
            labels = torch.zeros(N).type(torch.LongTensor).to(self.device)

        NxCxFeatures = features.view(
            N, 1, A
        ).expand(
            N, C, A
        )
        onehot = torch.zeros(N, C).to(self.device)
        onehot.scatter_(1, labels.view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        var_temp = features_by_sort - \
                   ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

        var_temp = torch.bmm(
            var_temp.permute(1, 2, 0),
            var_temp.permute(1, 0, 2)
        ).div(Amount_CxA.view(C, A, 1).expand(C, A, A))

        sum_weight_CV = onehot.sum(0).view(C, 1, 1).expand(C, A, A)

        sum_weight_AV = onehot.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(
            sum_weight_CV + self.Amount.view(C, 1, 1).expand(C, A, A)
        )
        weight_CV[weight_CV != weight_CV] = 0

        weight_AV = sum_weight_AV.div(
            sum_weight_AV + self.Amount.view(C, 1).expand(C, A)
        )
        weight_AV[weight_AV != weight_AV] = 0

        additional_CV = weight_CV.mul(1 - weight_CV).mul(
            torch.bmm(
                (self.Ave - ave_CxA).view(C, A, 1),
                (self.Ave - ave_CxA).view(C, 1, A)
            )
        )

        new_CoVariance = (self.CoVariance.mul(1 - weight_CV) + var_temp
                          .mul(weight_CV)).detach() + additional_CV.detach()

        new_Ave = (self.Ave.mul(1 - weight_AV) + ave_CxA.mul(weight_AV)).detach()

        new_Amount = self.Amount + onehot.sum(0)

        return {
            'Ave': new_Ave,
            'CoVariance': new_CoVariance,
            'Amount': new_Amount
        }

    def update(self, dic):
        self.Ave = dic['Ave']
        self.CoVariance = dic['CoVariance']
        self.Amount = dic['Amount']

    def invert(self):
        self.CoVarianceInv = torch.linalg.inv(self.CoVariance)

    def transform(self, features, labels):
        CV = self.CoVariance[labels]
        (N, A) = features.size()
        transformed = torch.bmm(F.normalize(CV), features.view(N, A, 1))
        return transformed.squeeze(-1)


class Coverage:
    def __init__(self, model_name, layer_size_dict, threshold, device, save_path, TOOL_LOG_FILE_PATH, **kwargs):
        self.device = device
        # self.model = model
        # self.model.to(self.device)
        self.layer_size_dict = layer_size_dict  # 包含各层的名称和输出尺寸
        self.every_pic_cover_record = {}
        self.every_pic_gain_record = {}
        self.model_name = model_name
        self.current_record = []
        self.TOOL_LOG_FILE_PATH = TOOL_LOG_FILE_PATH
        self.init_variable(threshold, save_path, **kwargs)

    def init_variable(self, **kwargs):
        raise NotImplementedError

    def calculate(self, data):
        raise NotImplementedError

    def coverage(self, cove_dict):
        raise NotImplementedError

    def save(self, Type):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def calculate_coverage_for_one(self, cov_dict, file_name):
        raise NotImplementedError

    def load_build(self):
        NotImplemented

    def build(self, data_loader):
        NotImplemented

    def step(self, data, file_name):
        tmp_cove_dict, cove_dict = self.calculate(data)
        self.every_pic_cover_record[file_name] = self.coverage(tmp_cove_dict)
        gain = self.gain(cove_dict)
        self.every_pic_gain_record[file_name] = gain
        self.update(cove_dict, gain)

    def update(self, all_cove_dict, delta=None):
        self.coverage_dict = all_cove_dict

        if delta:
            self.current += delta
        else:
            self.current = self.coverage(all_cove_dict)

        self.current_record.append(self.current)

    def gain(self, cove_dict_new):
        new_rate = self.coverage(cove_dict_new)
        return new_rate - self.current


class NLC(Coverage):
    def init_variable(self, hyper, save_path, **kwargs):
        # assert hyper is None, 'NLC has no hyper-parameter'
        self.estimator_dict = {}
        self.current = 1
        self.k = None
        for (layer_name, layer_size) in self.layer_size_dict.items():
            self.estimator_dict[layer_name] = Estimator(feature_num=layer_size["Output"][0])

        self.init_estimator_dict = copy.deepcopy(self.estimator_dict)
        self.name = "NLC"
        self.save_path = save_path + "/NLC"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)

    def calculate(self, data):
        stat_dict = {}
        tmp_stat_dict = {}
        tmp_init_estimator_dict = copy.deepcopy(self.init_estimator_dict)
        for (layer_name, layer_output) in data.items():
            info_dict = self.estimator_dict[layer_name].calculate(layer_output.to(self.device))
            stat_dict[layer_name] = (info_dict['Ave'], info_dict['CoVariance'], info_dict['Amount'])

            tmp_info_data = tmp_init_estimator_dict[layer_name].calculate(layer_output.to(self.device))
            tmp_stat_dict[layer_name] = (tmp_info_data['Ave'], tmp_info_data['CoVariance'], tmp_info_data['Amount'])
        return tmp_stat_dict, stat_dict

    def gain(self, stat_new):
        total = 0
        layer_to_update = []
        for i, layer_name in enumerate(stat_new.keys()):
            (new_Ave, new_CoVar, new_Amt) = stat_new[layer_name]
            value = self.norm(new_CoVar) - self.norm(self.estimator_dict[layer_name].CoVariance)

            if value > 0:
                layer_to_update.append(layer_name)
                total += value

        if total > 0:
            return (total, layer_to_update)
        else:
            return None

    def update(self, stat_dict, gain=None):
        if gain is None:
            for i, layer_name in enumerate(stat_dict.keys()):
                (new_Ave, new_CoVariance, new_Amount) = stat_dict[layer_name]
                self.estimator_dict[layer_name].Ave = new_Ave
                self.estimator_dict[layer_name].CoVariance = new_CoVariance
                self.estimator_dict[layer_name].Amount = new_Amount
            self.current = self.coverage(self.estimator_dict)
        else:
            (delta, layer_to_update) = gain
            for layer_name in layer_to_update:
                (new_Ave, new_CoVariance, new_Amount) = stat_dict[layer_name]
                self.estimator_dict[layer_name].Ave = new_Ave
                self.estimator_dict[layer_name].CoVariance = new_CoVariance
                self.estimator_dict[layer_name].Amount = new_Amount
            self.current += delta
        self.current_record.append(self.current)

    def coverage(self, stat_dict):
        val = 0
        try:
            for i, layer_name in enumerate(stat_dict.keys()):
                CoVariance = stat_dict[layer_name].CoVariance
                val += self.norm(CoVariance)
        except:
            for i, layer_name in enumerate(stat_dict.keys()):
                (new_Ave, CoVariance, new_Amount) = stat_dict[layer_name]
                val += self.norm(CoVariance)

        return val

    def norm(self, vec, mode='L1', reduction='mean'):
        m = np.prod(vec.size())
        assert mode in ['L1', 'L2']
        assert reduction in ['mean', 'sum']
        if mode == 'L1':
            total = vec.abs().sum()
        elif mode == 'L2':
            total = vec.pow(2).sum().sqrt()
        if reduction == 'mean':
            return total / m
        elif reduction == 'sum':
            return total

    def save(self, Type):
        stat_dict = {}
        for layer_name in self.estimator_dict.keys():
            stat_dict[layer_name] = {
                'Ave': self.estimator_dict[layer_name].Ave,
                'CoVariance': self.estimator_dict[layer_name].CoVariance,
                'Amount': self.estimator_dict[layer_name].Amount
            }
        coverage_dict = {"stat": stat_dict}

        path_every_pic_cover_record = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_every_pic_cover_record.pth"
        path_every_pic_gain_record = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_every_pic_gain_record.pth"
        path_coverage_dict = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_last_coverage_dict.pth"
        path_everytime_current = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_everytime_current.pth"

        torch.save(self.every_pic_cover_record, path_every_pic_cover_record)
        torch.save(self.every_pic_gain_record, path_every_pic_gain_record)
        torch.save(coverage_dict, path_coverage_dict)
        torch.save(self.current_record, path_everytime_current)

    def recvoery_progress(self, Type):
        path_coverage_dict = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_last_coverage_dict.pth"
        path_everytime_current = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_everytime_current.pth"

        self.current_record = torch.load(path_everytime_current)
        self.current = self.current_record[-1]

        coverage_dict = torch.load(path_coverage_dict)
        stat_dict = coverage_dict['stat']
        for layer_name in stat_dict.keys():
            self.estimator_dict[layer_name].Ave = stat_dict[layer_name]['Ave']
            self.estimator_dict[layer_name].CoVariance = stat_dict[layer_name]['CoVariance']
            self.estimator_dict[layer_name].Amount = stat_dict[layer_name]['Amount']


class NC(Coverage):
    def init_variable(self, hyper, save_path, **kwargs):
        self.threshold = hyper
        self.name = "NC"
        self.coverage_dict = {}
        for (layer_name, layer_size) in self.layer_size_dict.items():
            self.coverage_dict[layer_name] = torch.zeros(layer_size["Output"][0]).type(torch.BoolTensor).to(self.device)
        self.current = 0
        self.save_path = save_path + f"/NC"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)

    def calculate(self, data):
        cove_dict = {}
        tmp_cove_dict = {}
        for (layer_name, layer_output) in data.items():
            scaled_output = scale(layer_output)
            mask_index = scaled_output > self.threshold
            is_covered = mask_index.sum(0) > 0
            tmp_cove_dict[layer_name] = is_covered
            cove_dict[layer_name] = is_covered | self.coverage_dict[layer_name]
        return tmp_cove_dict, cove_dict

    def coverage(self, cove_dict):
        (cove, total) = (0, 0)
        for k in cove_dict.keys():
            is_covered = cove_dict[k]
            cove += is_covered.sum()
            try:
                total += len(is_covered)
            except TypeError:
                total += 1

        avg_false_ratio = (cove / total).item()
        return avg_false_ratio

    def save(self, Type):
        path_every_pic_cover_record = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.threshold}_every_pic_cover_record.pth"
        path_every_pic_gain_record = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.threshold}_every_pic_gain_record.pth"
        path_everytime_current = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.threshold}_everytime_current.pth"

        torch.save(self.every_pic_cover_record, path_every_pic_cover_record)
        torch.save(self.every_pic_gain_record, path_every_pic_gain_record)
        torch.save(self.current_record, path_everytime_current)

    def recvoery_progress(self, Type):
        path_everytime_current = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.threshold}_everytime_current.pth"
        self.current_record = torch.load(path_everytime_current)
        self.current = self.current_record[-1]


class KMNC(Coverage):
    def init_variable(self, hyper, save_path, **kwargs):
        self.k = int(hyper)
        self.name = 'KMNC'
        self.range_dict = {}
        coverage_multisec_dict = {}
        for (layer_name, layer_size) in self.layer_size_dict.items():
            num_neuron = layer_size["Output"][0]
            coverage_multisec_dict[layer_name] = torch.zeros((num_neuron, self.k + 1)).type(torch.BoolTensor).to(
                self.device)
            self.range_dict[layer_name] = [torch.ones(num_neuron).to(self.device) * 10000,
                                           torch.ones(num_neuron).to(self.device) * -10000]
        self.coverage_dict = {
            'multisec': coverage_multisec_dict
        }

        self.init_range_dict = copy.deepcopy(self.range_dict)
        self.current = 0
        self.save_path = save_path + "/KMNC"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)

        self.load_build()

    def load_build(self):
        path = self.TOOL_LOG_FILE_PATH + f"/coverages/{self.name}-{self.k}-range_dict.pth"
        if os.path.exists(path):
            self.range_dict = torch.load(path)

    def build(self, data):
        self.set_range(data)

    def set_range(self, data):
        for (layer_name, layer_output) in data.items():
            cur_max, _ = layer_output.max(0)
            cur_min, _ = layer_output.min(0)

            is_less = cur_min < self.range_dict[layer_name][0]
            is_greater = cur_max > self.range_dict[layer_name][1]

            self.range_dict[layer_name][0] = is_less * cur_min + ~is_less * self.range_dict[layer_name][0]
            self.range_dict[layer_name][1] = is_greater * cur_max + ~is_greater * self.range_dict[layer_name][1]

    def save_build(self):
        path = self.TOOL_LOG_FILE_PATH + "/coverages"

        if not os.path.exists(path):
            os.makedirs(path)

        path = path + f"/{self.name}-{self.k}-range_dict.pth"
        torch.save(self.range_dict, path)

    def calculate(self, data):
        multisec_cove_dict = {}  
        tmp_cove_dict = {}
        for (layer_name, layer_output) in data.items():
            [l_bound, u_bound] = self.range_dict[layer_name]
            num_neuron = layer_output.size(1) 

            try:
                multisec_index = (u_bound > l_bound) & (layer_output >= l_bound) & (layer_output <= u_bound)
            except:
                print(layer_name, layer_output.size())
                print(u_bound.size(), l_bound.size())

            multisec_covered = torch.zeros(num_neuron, self.k + 1).type(torch.BoolTensor).to(self.device)

            div_index = u_bound > l_bound
            div = (~div_index) * 1e-6 + div_index * (u_bound - l_bound)

            multisec_output = torch.ceil((layer_output - l_bound) / div * self.k).type(torch.LongTensor).to(
                self.device) * multisec_index

            index = tuple([torch.LongTensor(list(range(num_neuron))), multisec_output])
            multisec_covered[index] = True

            multisec_cove_dict[layer_name] = multisec_covered | self.coverage_dict['multisec'][layer_name]
            tmp_cove_dict[layer_name] = multisec_covered

        return {'multisec': tmp_cove_dict}, {'multisec': multisec_cove_dict}

    def coverage(self, cove_dict):
        multisec_cove_dict = cove_dict['multisec']
        (multisec_cove, multisec_total) = (0, 0)
        for layer_name in multisec_cove_dict.keys():
            multisec_covered = multisec_cove_dict[layer_name]
            num_neuron = multisec_covered.size(0)
            multisec_cove += torch.sum(multisec_covered[:, 1:])
            multisec_total += (num_neuron * self.k)
        multisec_rate = multisec_cove / multisec_total

        return multisec_rate.item()

    def save(self, Type):
        path_every_pic_cover_record = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_every_pic_cover_record.pth"
        path_every_pic_gain_record = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_every_pic_gain_record.pth"
        path_range = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_range_dict.pth"
        path_everytime_current = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_everytime_current.pth"

        torch.save(self.every_pic_cover_record, path_every_pic_cover_record)
        torch.save(self.every_pic_gain_record, path_every_pic_gain_record)
        torch.save(self.range_dict, path_range)
        torch.save(self.current_record, path_everytime_current)

    def recvoery_progress(self, Type):
        path_range = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_range_dict.pth"
        path_everytime_current = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_everytime_current.pth"

        self.range_dict = torch.load(path_range)
        self.current_record = torch.load(path_everytime_current)
        self.current = self.current_record[-1]


class SNAC(KMNC):

    def init_variable(self, hyper, save_path, **kwargs):
        # assert hyper is None
        self.k = None
        self.name = 'SNAC'
        self.range_dict = {}
        coverage_upper_dict = {}
        for (layer_name, layer_size) in self.layer_size_dict.items():
            num_neuron = layer_size["Output"][0]
            coverage_upper_dict[layer_name] = torch.zeros(num_neuron).type(torch.BoolTensor).to(self.device)
            self.range_dict[layer_name] = [torch.ones(num_neuron).to(self.device) * 10000,
                                           torch.ones(num_neuron).to(self.device) * -10000]
        self.coverage_dict = {
            'upper': coverage_upper_dict
        }
        self.current = 0
        self.init_range_dict = copy.deepcopy(self.range_dict)
        self.save_path = save_path + "/SNAC"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
        self.load_build()

    def calculate(self, data):
        upper_cove_dict = {}
        tmp_cove_dict = {}
        for (layer_name, layer_output) in data.items():
            [l_bound, u_bound] = self.range_dict[layer_name]
            upper_covered = (layer_output > u_bound).sum(0) > 0
            upper_cove_dict[layer_name] = upper_covered | self.coverage_dict['upper'][layer_name]
            tmp_cove_dict[layer_name] = upper_covered
        return {'upper': tmp_cove_dict}, {'upper': upper_cove_dict}

    def calculate_coverage_for_one(self, cov_dict, file_name):
        self.every_pic_cover_record[file_name] = {}
        upper_cove_dict = cov_dict['upper']
        for layer_name in upper_cove_dict.keys():
            upper_covered = upper_cove_dict[layer_name]
            upper_cove = upper_covered.sum()
            upper_total = len(upper_covered)
            upper_rate = upper_cove / upper_total
            self.every_pic_cover_record[file_name].update({layer_name: upper_rate})

    def coverage(self, cove_dict):
        upper_cove_dict = cove_dict['upper']
        (upper_cove, upper_total) = (0, 0)
        for layer_name in upper_cove_dict.keys():
            upper_covered = upper_cove_dict[layer_name]
            upper_cove += upper_covered.sum()
            upper_total += len(upper_covered)
        upper_rate = upper_cove / upper_total
        return upper_rate.item()


class NBC(KMNC):
    def init_variable(self, hyper, save_path, **kwargs):
        self.k = None
        self.name = 'NBC'
        self.range_dict = {}
        coverage_lower_dict = {}
        coverage_upper_dict = {}
        for (layer_name, layer_size) in self.layer_size_dict.items():
            num_neuron = layer_size["Output"][0]
            coverage_lower_dict[layer_name] = torch.zeros(num_neuron).type(torch.BoolTensor).to(self.device)
            coverage_upper_dict[layer_name] = torch.zeros(num_neuron).type(torch.BoolTensor).to(self.device)
            self.range_dict[layer_name] = [torch.ones(num_neuron).to(self.device) * 10000,
                                           torch.ones(num_neuron).to(self.device) * -10000]
        self.coverage_dict = {
            'lower': coverage_lower_dict,
            'upper': coverage_upper_dict
        }
        self.current = 0
        self.init_range_dict = copy.deepcopy(self.range_dict)
        self.save_path = save_path + "/NBC"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)

        self.load_build()

    def calculate(self, data):
        lower_cove_dict = {}
        upper_cove_dict = {}
        tmp_lower_cove_dict = {}
        tmp_upper_cove_dict = {}

        for (layer_name, layer_output) in data.items():
            [l_bound, u_bound] = self.range_dict[layer_name]

            lower_covered = (layer_output < l_bound).sum(0) > 0
            upper_covered = (layer_output > u_bound).sum(0) > 0

            lower_cove_dict[layer_name] = lower_covered | self.coverage_dict['lower'][layer_name]
            upper_cove_dict[layer_name] = upper_covered | self.coverage_dict['upper'][layer_name]

            tmp_lower_cove_dict[layer_name] = lower_covered
            tmp_upper_cove_dict[layer_name] = upper_covered

        return {'lower': tmp_lower_cove_dict, 'upper': tmp_upper_cove_dict}, {'lower': lower_cove_dict,
                                                                              'upper': upper_cove_dict}

    def coverage(self, cove_dict):
        lower_cove_dict = cove_dict['lower']
        upper_cove_dict = cove_dict['upper']

        (lower_cove, lower_total) = (0, 0)
        (upper_cove, upper_total) = (0, 0)
        for layer_name in lower_cove_dict.keys():
            lower_covered = lower_cove_dict[layer_name]
            upper_covered = upper_cove_dict[layer_name]

            lower_cove += lower_covered.sum()
            upper_cove += upper_covered.sum()

            lower_total += len(lower_covered)
            upper_total += len(upper_covered)
        lower_rate = lower_cove / lower_total
        upper_rate = upper_cove / upper_total

        return (lower_rate + upper_rate).item() / 2


class TKNC(Coverage):

    def init_variable(self, hyper, save_path, **kwargs):
        assert hyper is not None
        self.k = int(hyper)
        self.name = 'TKNC'
        self.coverage_dict = {}
        for (layer_name, layer_size) in self.layer_size_dict.items():
            num_neuron = layer_size["Output"][0]
            self.coverage_dict[layer_name] = torch.zeros(num_neuron).type(torch.BoolTensor).to(self.device)
        self.current = 0

        self.save_path = save_path + "/TKNC"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)

    def calculate(self, data):
        cove_dict = {}
        tmp_cove_dict = {}
        for (layer_name, layer_output) in data.items():
            batch_size = layer_output.size(0)
            num_neuron = layer_output.size(1)
            # layer_output: (batch_size, num_neuron)
            _, idx = layer_output.topk(min(self.k, num_neuron), dim=1, largest=True, sorted=False)
            # idx: (batch_size, k)
            covered = torch.zeros(layer_output.size()).to(self.device)
            index = tuple([torch.LongTensor(list(range(batch_size))), idx.transpose(0, 1)])
            covered[index] = 1
            is_covered = covered.sum(0) > 0
            cove_dict[layer_name] = is_covered | self.coverage_dict[layer_name]
            tmp_cove_dict[layer_name] = is_covered

        return tmp_cove_dict, cove_dict

    def coverage(self, cove_dict):
        (cove, total) = (0, 0)
        for layer_name in cove_dict.keys():
            is_covered = cove_dict[layer_name]
            cove += is_covered.sum()
            total += len(is_covered)
        rate = (cove / total).item()

        return rate

    def save(self, Type):
        path_every_pic_cover_record = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_every_pic_cover_record.pth"
        path_every_pic_gain_record = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_every_pic_gain_record.pth"
        path_everytime_current = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_everytime_current.pth"

        torch.save(self.every_pic_cover_record, path_every_pic_cover_record)
        torch.save(self.every_pic_gain_record, path_every_pic_gain_record)
        torch.save(self.current_record, path_everytime_current)

    def recvoery_progress(self, Type):
        path_everytime_current = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_everytime_current.pth"
        self.current_record = torch.load(path_everytime_current)
        self.current = self.current_record[-1]


class TKNP(Coverage):

    def init_variable(self, hyper, save_path, **kwargs):
        assert hyper is not None
        self.k = int(hyper)
        self.name = "TKNP"
        self.layer_pattern = {}
        self.network_pattern = set()
        self.current = 0
        for (layer_name, layer_size) in self.layer_size_dict.items():
            self.layer_pattern[layer_name] = set()

        self.coverage_dict = {
            'layer_pattern': self.layer_pattern,
            'network_pattern': self.network_pattern
        }

        self.init_coverage_dict = copy.deepcopy(self.coverage_dict)
        self.init_coverage_dict_modify = False

        self.save_path = save_path + "/TKNP"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)

    def calculate(self, data):
        layer_pat = {}
        topk_idx_list = []
        tmp_coverage_dict = copy.deepcopy(self.init_coverage_dict)
        tmp_layer_pat = {}
        for (layer_name, layer_output) in data.items():
            num_neuron = layer_output.size(1)
            _, idx = layer_output.topk(min(self.k, num_neuron), dim=1, largest=True, sorted=True)
            pat = set([str(s) for s in list(idx[:,])])
            topk_idx_list.append(idx)
            layer_pat[layer_name] = set.union(pat, self.layer_pattern[layer_name])
            tmp_layer_pat[layer_name] = set.union(pat, tmp_coverage_dict["layer_pattern"][layer_name])

        network_topk_idx = torch.cat(topk_idx_list, 1)
        network_pat = set([str(s) for s in list(network_topk_idx[:,])])
        network_pat = set.union(network_pat, self.network_pattern)
        tmp_network_pat = set.union(network_pat, tmp_coverage_dict["network_pattern"])
        results = {
            'layer_pattern': layer_pat,
            'network_pattern': network_pat
        }
        tmp_results = {
            'layer_pattern': tmp_layer_pat,
            'network_pattern': tmp_network_pat
        }

        if not self.init_coverage_dict_modify:
            self.init_coverage_dict["layer_pattern"] = tmp_results['layer_pattern']
            self.init_coverage_dict['network_pattern'] = tmp_results['network_pattern']
            self.init_coverage_dict_modify = True

        return tmp_results, results


    def coverage(self, cove_dict, mode='layer'):
        assert mode in ['network', 'layer']
        if mode == 'network':
            return len(cove_dict['network_pattern'])
        cnt = 0
        if mode == 'layer':
            for layer_name in cove_dict['layer_pattern'].keys():
                cnt += len(cove_dict['layer_pattern'][layer_name])
        return cnt

    def update(self, all_cove_dict, delta=None):
        self.layer_pattern = all_cove_dict['layer_pattern']
        self.network_pattern = all_cove_dict['network_pattern']

        if delta:
            # 如果delta被提供，则将其加到当前值上
            self.current += delta
        else:
            # 如果没有delta，重新计算当前值基于新的覆盖字典
            self.current = self.coverage(all_cove_dict)

        self.current_record.append(self.current)

    def save(self, Type):
        path_every_pic_cover_record = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_every_pic_cover_record.pth"
        path_every_pic_gain_record = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_every_pic_gain_record.pth"
        path_everytime_current = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_everytime_current.pth"
        path_hyper_setting = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_path_hyper_setting.pth"

        hyper_setting = {
            "layer_pattern": self.layer_pattern,
            "network_pattern": self.network_pattern,
        }

        torch.save(self.every_pic_cover_record, path_every_pic_cover_record)
        torch.save(self.every_pic_gain_record, path_every_pic_gain_record)
        torch.save(self.current_record, path_everytime_current)
        torch.save(hyper_setting, path_hyper_setting)

    def recvoery_progress(self, Type):
        path_everytime_current = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_everytime_current.pth"
        path_hyper_setting = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_path_hyper_setting.pth"

        self.current_record = torch.load(path_everytime_current)
        self.current = self.current_record[-1]
        hyper_setting = torch.load(path_hyper_setting)
        self.layer_pattern = hyper_setting["layer_pattern"]
        self.network_pattern = hyper_setting["network_pattern"]


class CC(Coverage):
    def init_variable(self, hyper, save_path, **kwargs):
        assert hyper is not None
        self.threshold = int(hyper)
        self.distant_dict = {}
        self.flann_dict = {}
        self.current = 0
        self.name = "CC"

        for (layer_name, layer_size) in self.layer_size_dict.items():
            self.flann_dict[layer_name] = FLANN()
            self.distant_dict[layer_name] = []

        self.init_var = {
            "flann_dict": copy.deepcopy(self.flann_dict),
            "distant_dict": copy.deepcopy(self.distant_dict),
        }

        self.save_path = save_path + "/CC"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)

    def help_cal(self, layer_output, layer_name, dist_dict, distant_dict, flann_dict):
        for single_output in layer_output:
            single_output = single_output.cpu().numpy()
            if len(distant_dict[layer_name]) > 0:
                query = np.expand_dims(single_output, 0)
                _, approx_distances = flann_dict[layer_name].nn_index(query, num_neighbors=1)
                exact_distances = [
                    np.sum(np.square(single_output - distant_vec))
                    for distant_vec in distant_dict[layer_name]
                ]
                buffer_distances = [
                    np.sum(np.square(single_output - buffer_vec))
                    for buffer_vec in dist_dict[layer_name]
                ]
                nearest_distance = min(exact_distances + approx_distances.tolist() + buffer_distances)
                if nearest_distance > self.threshold:
                    dist_dict[layer_name].append(single_output)
            else:
                flann_dict[layer_name].build_index(single_output)
                distant_dict[layer_name].append(single_output)

        return dist_dict, flann_dict, distant_dict

    def calculate(self, data):
        dist_dict = {}
        tmp_dist_dict = {}
        for (layer_name, layer_output) in data.items():
            dist_dict[layer_name] = []
            tmp_dist_dict[layer_name] = []

            dist_dict, self.flann_dict, self.distant_dict = self.help_cal(layer_output, layer_name, dist_dict,
                                                                          self.distant_dict, self.flann_dict)

            tmp_distant_dict = copy.deepcopy(self.init_var["distant_dict"])
            tmp_flann_dict = copy.deepcopy(self.init_var["flann_dict"])
            tmp_dist_dict, _, _ = self.help_cal(layer_output, layer_name, dist_dict, tmp_distant_dict, tmp_flann_dict)

        return tmp_dist_dict, dist_dict

    def gain(self, dist_dict):
        increased = self.coverage(dist_dict)
        return increased

    def update(self, dist_dict, delta=None):
        for layer_name in self.distant_dict.keys():
            self.distant_dict[layer_name] += dist_dict[layer_name]
            self.flann_dict[layer_name].build_index(np.array(self.distant_dict[layer_name]))
        if delta:
            self.current += delta
        else:
            self.current += self.coverage(dist_dict)
        self.current_record.append(self.current)

    def coverage(self, dist_dict):
        total = 0
        for layer_name in dist_dict.keys():
            total += len(dist_dict[layer_name])

        return total

    def save(self, Type):
        path_every_pic_cover_record = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.threshold}_every_pic_cover_record.pth"
        path_every_pic_gain_record = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.threshold}_every_pic_gain_record.pth"
        path_everytime_current = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.threshold}_everytime_current.pth"
        path_hyper_setting = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.threshold}_path_hyper_setting.pth"

        hyper_setting = {
            "flann_dict": self.flann_dict,
            "distant_dict": self.distant_dict,
        }

        torch.save(self.every_pic_cover_record, path_every_pic_cover_record)
        torch.save(self.every_pic_gain_record, path_every_pic_gain_record)
        torch.save(self.current_record, path_everytime_current)
        torch.save(hyper_setting, path_hyper_setting)

    def recvoery_progress(self, Type):
        path_everytime_current = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.threshold}_everytime_current.pth"
        path_hyper_setting = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.threshold}_path_hyper_setting.pth"

        self.current_record = torch.load(path_everytime_current)
        self.current = self.current_record[-1]
        hyper_setting = torch.load(path_hyper_setting)
        self.flann_dict = hyper_setting["flann_dict"]
        self.distant_dict = hyper_setting["distant_dict"]


class ADC(Coverage):

    def init_variable(self, hyper, save_path, **kwargs):
        if hyper is None:
            self.k = 1000
            self.extension_factor = 0.2 
        elif isinstance(hyper, str) and '-' in hyper:
            parts = hyper.split('-')
            self.k = int(parts[0])
            self.extension_factor = float(parts[1]) if len(parts) > 1 else 0.1
        else:
            self.k = int(hyper)
            self.extension_factor = 0.2 

        self.name = 'ADC'

        self.range_dict = {} 
        self.extended_bounds_dict = {}
        self.bucket_covered_dict = {}
        self.la_covered_dict = {} 
        self.ra_covered_dict = {}
        self.la_counter_dict = {} 
        self.ra_counter_dict = {} 
        self.anomaly_samples = [] 

        for (layer_name, layer_size) in self.layer_size_dict.items():
            num_neuron = layer_size["Output"][0]

            self.range_dict[layer_name] = [
                torch.ones(num_neuron).to(self.device) * 10000,  # min_val
                torch.ones(num_neuron).to(self.device) * -10000  # max_val
            ]

            self.extended_bounds_dict[layer_name] = [
                torch.zeros(num_neuron).to(self.device),  # left_bound
                torch.zeros(num_neuron).to(self.device)  # right_bound
            ]

            self.bucket_covered_dict[layer_name] = torch.zeros(
                (num_neuron, 3 * self.k)
            ).type(torch.BoolTensor).to(self.device)

            self.la_covered_dict[layer_name] = torch.zeros(num_neuron).type(torch.BoolTensor).to(self.device)
            self.ra_covered_dict[layer_name] = torch.zeros(num_neuron).type(torch.BoolTensor).to(self.device)

            self.la_counter_dict[layer_name] = torch.zeros(num_neuron).type(torch.LongTensor).to(self.device)
            self.ra_counter_dict[layer_name] = torch.zeros(num_neuron).type(torch.LongTensor).to(self.device)

        self.coverage_dict = {
            'bucket_covered': self.bucket_covered_dict,
            'la_covered': self.la_covered_dict,
            'ra_covered': self.ra_covered_dict
        }

        self.init_range_dict = copy.deepcopy(self.range_dict)
        self.current = 0.0

        self.save_path = save_path + "/ADC"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)

        self.anomaly_path = self.save_path + "/anomaly_samples"
        if not os.path.exists(self.anomaly_path):
            os.makedirs(self.anomaly_path, exist_ok=True)

        self.load_build()

    def load_build(self):
        range_path = self.TOOL_LOG_FILE_PATH + f"/coverages/{self.name}-{self.k}-range_dict.pth"
        bounds_path = self.TOOL_LOG_FILE_PATH + f"/coverages/{self.name}-{self.k}-extended_bounds_dict.pth"

        if os.path.exists(range_path):
            self.range_dict = torch.load(range_path)
        if os.path.exists(bounds_path):
            self.extended_bounds_dict = torch.load(bounds_path)

    def build(self, data):
        self.set_range(data)
        self.compute_extended_bounds()

    def set_range(self, data):
        for (layer_name, layer_output) in data.items():
            cur_max, _ = layer_output.max(0)
            cur_min, _ = layer_output.min(0)

            is_less = cur_min < self.range_dict[layer_name][0]
            is_greater = cur_max > self.range_dict[layer_name][1]

            self.range_dict[layer_name][0] = is_less * cur_min + ~is_less * self.range_dict[layer_name][0]
            self.range_dict[layer_name][1] = is_greater * cur_max + ~is_greater * self.range_dict[layer_name][1]

    def compute_extended_bounds(self):
        for layer_name in self.range_dict.keys():
            min_val = self.range_dict[layer_name][0]
            max_val = self.range_dict[layer_name][1]

            # 计算扩展量 a
            range_val = max_val - min_val
            extension = self.extension_factor * range_val

            # 计算扩展边界
            self.extended_bounds_dict[layer_name][0] = min_val - extension  # left_bound
            self.extended_bounds_dict[layer_name][1] = max_val + extension  # right_bound

    def save_build(self):
        path = self.TOOL_LOG_FILE_PATH + "/coverages"
        if not os.path.exists(path):
            os.makedirs(path)

        range_path = path + f"/{self.name}-{self.k}-range_dict.pth"
        bounds_path = path + f"/{self.name}-{self.k}-extended_bounds_dict.pth"

        torch.save(self.range_dict, range_path)
        torch.save(self.extended_bounds_dict, bounds_path)

    def calculate(self, data, sample_info=None):
        tmp_bucket_covered = {}
        tmp_la_covered = {}
        tmp_ra_covered = {}

        new_bucket_covered = {}
        new_la_covered = {}
        new_ra_covered = {}

        anomaly_info_batch = [] if sample_info is not None else None

        for (layer_name, layer_output) in data.items():
            left_bound = self.extended_bounds_dict[layer_name][0]  # (num_neuron,)
            right_bound = self.extended_bounds_dict[layer_name][1]  # (num_neuron,)
            min_val = self.range_dict[layer_name][0]  # (num_neuron,)
            max_val = self.range_dict[layer_name][1]  # (num_neuron,)
            num_neuron = layer_output.size(1)
            batch_size = layer_output.size(0)

            current_bucket_covered = torch.zeros(num_neuron, 3 * self.k).type(torch.BoolTensor).to(self.device)

            lb = left_bound.unsqueeze(0)  # (1, num_neuron)
            rb = right_bound.unsqueeze(0)  # (1, num_neuron)
            mv = min_val.unsqueeze(0)  # (1, num_neuron)
            xv = max_val.unsqueeze(0)  # (1, num_neuron)

            la_mask = layer_output < lb
            current_la_covered = la_mask.any(dim=0)  # (num_neuron,)
            la_count = la_mask.sum(dim=0).long()  # (num_neuron,)
            self.la_counter_dict[layer_name] += la_count

            if anomaly_info_batch is not None and la_mask.any():
                la_batch_idx, la_neuron_idx = torch.where(la_mask)
                for b_idx, n_idx in zip(la_batch_idx.cpu().numpy(), la_neuron_idx.cpu().numpy()):
                    anomaly_info_batch.append({
                        'sample_info': sample_info,
                        'layer_name': layer_name,
                        'neuron_idx': n_idx,
                        'activation_value': layer_output[b_idx, n_idx].item(),
                        'anomaly_type': 'LA',
                        'left_bound': left_bound[n_idx].item(),
                        'right_bound': right_bound[n_idx].item()
                    })

            le_mask = (layer_output >= lb) & (layer_output < mv)
            if le_mask.any():
                div = (mv - lb).clamp(min=1e-6)
                bucket_indices = torch.floor((layer_output - lb) / div * self.k).long()
                bucket_indices = torch.clamp(bucket_indices, 0, self.k - 1)

                le_batch_idx, le_neuron_idx = torch.where(le_mask)
                if len(le_batch_idx) > 0:
                    neuron_indices = le_neuron_idx
                    bucket_offset = bucket_indices[le_batch_idx, le_neuron_idx]
                    current_bucket_covered[neuron_indices, bucket_offset] = True

            core_mask = (layer_output >= mv) & (layer_output <= xv)
            if core_mask.any():
                div = (xv - mv).clamp(min=1e-6)
                bucket_indices = torch.floor((layer_output - mv) / div * self.k).long()
                bucket_indices = torch.clamp(bucket_indices, 0, self.k - 1)

                core_batch_idx, core_neuron_idx = torch.where(core_mask)
                if len(core_batch_idx) > 0:
                    neuron_indices = core_neuron_idx
                    bucket_offset = self.k + bucket_indices[core_batch_idx, core_neuron_idx]
                    current_bucket_covered[neuron_indices, bucket_offset] = True

            re_mask = (layer_output > xv) & (layer_output <= rb)
            if re_mask.any():
                div = (rb - xv).clamp(min=1e-6)
                bucket_indices = torch.floor((layer_output - xv) / div * self.k).long()
                bucket_indices = torch.clamp(bucket_indices, 0, self.k - 1)

                re_batch_idx, re_neuron_idx = torch.where(re_mask)
                if len(re_batch_idx) > 0:
                    neuron_indices = re_neuron_idx
                    bucket_offset = 2 * self.k + bucket_indices[re_batch_idx, re_neuron_idx]
                    current_bucket_covered[neuron_indices, bucket_offset] = True

            ra_mask = layer_output > rb
            current_ra_covered = ra_mask.any(dim=0)  # (num_neuron,)
            ra_count = ra_mask.sum(dim=0).long()  # (num_neuron,)
            self.ra_counter_dict[layer_name] += ra_count

            if anomaly_info_batch is not None and ra_mask.any():
                ra_batch_idx, ra_neuron_idx = torch.where(ra_mask)
                for b_idx, n_idx in zip(ra_batch_idx.cpu().numpy(), ra_neuron_idx.cpu().numpy()):
                    anomaly_info_batch.append({
                        'sample_info': sample_info,
                        'layer_name': layer_name,
                        'neuron_idx': n_idx,
                        'activation_value': layer_output[b_idx, n_idx].item(),
                        'anomaly_type': 'RA',
                        'left_bound': left_bound[n_idx].item(),
                        'right_bound': right_bound[n_idx].item()
                    })

            new_bucket_covered[layer_name] = current_bucket_covered | self.bucket_covered_dict[layer_name]
            new_la_covered[layer_name] = current_la_covered | self.la_covered_dict[layer_name]
            new_ra_covered[layer_name] = current_ra_covered | self.ra_covered_dict[layer_name]

            tmp_bucket_covered[layer_name] = current_bucket_covered
            tmp_la_covered[layer_name] = current_la_covered
            tmp_ra_covered[layer_name] = current_ra_covered

        if anomaly_info_batch:
            self._save_anomaly_samples_batch(anomaly_info_batch)

        tmp_coverage_dict = {
            'bucket_covered': tmp_bucket_covered,
            'la_covered': tmp_la_covered,
            'ra_covered': tmp_ra_covered
        }

        new_coverage_dict = {
            'bucket_covered': new_bucket_covered,
            'la_covered': new_la_covered,
            'ra_covered': new_ra_covered
        }

        return tmp_coverage_dict, new_coverage_dict

    def _save_anomaly_samples_batch(self, anomaly_info_batch):
        import json
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for info in anomaly_info_batch:
            sample_info = info['sample_info']
            layer_name = info['layer_name']
            neuron_idx = info['neuron_idx']
            activation_value = info['activation_value']
            anomaly_type = info['anomaly_type']
            left_bound = info['left_bound']
            right_bound = info['right_bound']

            # 计算偏差
            if anomaly_type == 'LA':
                deviation = left_bound - activation_value
            else:  # RA
                deviation = activation_value - right_bound

            metadata = {
                'sample_id': sample_info.get('id', 'unknown') if sample_info else 'unknown',
                'image_path': sample_info.get('path', 'unknown') if sample_info else 'unknown',
                'layer': layer_name,
                'neuron_index': neuron_idx,
                'activation_value': float(activation_value),
                'anomaly_type': anomaly_type,
                'left_bound': float(left_bound),
                'right_bound': float(right_bound),
                'deviation': float(deviation),
                'timestamp': timestamp
            }

            self.anomaly_samples.append(metadata)

    def _save_anomaly_sample(self, sample_info, layer_name, neuron_idx, activation_value,
                             anomaly_type, left_bound, right_bound):
        import json
        import datetime

        # 计算偏差
        if anomaly_type == 'LA':
            deviation = left_bound - activation_value
        else:  # RA
            deviation = activation_value - right_bound

        metadata = {
            'sample_id': sample_info.get('id', 'unknown') if sample_info else 'unknown',
            'image_path': sample_info.get('path', 'unknown') if sample_info else 'unknown',
            'layer': layer_name,
            'neuron_index': neuron_idx,
            'activation_value': float(activation_value),
            'anomaly_type': anomaly_type,
            'left_bound': float(left_bound),
            'right_bound': float(right_bound),
            'deviation': float(deviation),
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        self.anomaly_samples.append(metadata)

    def coverage(self, cove_dict):
        bucket_covered_dict = cove_dict['bucket_covered']
        la_covered_dict = cove_dict['la_covered']
        ra_covered_dict = cove_dict['ra_covered']

        total_coverage = 0.0
        total_neurons = 0

        for layer_name in bucket_covered_dict.keys():
            bucket_covered = bucket_covered_dict[layer_name]  # (num_neuron, 3k)
            la_covered = la_covered_dict[layer_name]          # (num_neuron,)
            ra_covered = ra_covered_dict[layer_name]          # (num_neuron,)

            num_neuron = bucket_covered.size(0)

            covered_buckets = bucket_covered.sum(dim=1).float()  # (num_neuron,)

            covered_buckets += la_covered.float()
            covered_buckets += ra_covered.float()

            neuron_coverage = covered_buckets / (3 * self.k + 2)

            total_coverage += neuron_coverage.sum().item()
            total_neurons += num_neuron

        if total_neurons > 0:
            return total_coverage / total_neurons
        else:
            return 0.0

    def get_bucket_coverage(self):
        total_covered_buckets = 0
        total_buckets = 0

        for layer_name in self.bucket_covered_dict.keys():
            bucket_covered = self.bucket_covered_dict[layer_name]  # (num_neuron, 3k)
            total_covered_buckets += bucket_covered.sum().item()
            total_buckets += bucket_covered.numel()

        if total_buckets > 0:
            return total_covered_buckets / total_buckets
        else:
            return 0.0

    def get_zone_coverage_breakdown(self):
        left_ext_covered = 0
        core_covered = 0
        right_ext_covered = 0
        total_buckets_per_zone = 0

        for layer_name in self.bucket_covered_dict.keys():
            bucket_covered = self.bucket_covered_dict[layer_name]  # (num_neuron, 3k)
            num_neuron = bucket_covered.size(0)

            left_ext_covered += bucket_covered[:, :self.k].sum().item()
            core_covered += bucket_covered[:, self.k:2*self.k].sum().item()
            right_ext_covered += bucket_covered[:, 2*self.k:3*self.k].sum().item()

            total_buckets_per_zone += num_neuron * self.k

        if total_buckets_per_zone > 0:
            return {
                'left_extension': left_ext_covered / total_buckets_per_zone,
                'core': core_covered / total_buckets_per_zone,
                'right_extension': right_ext_covered / total_buckets_per_zone
            }
        else:
            return {
                'left_extension': 0.0,
                'core': 0.0,
                'right_extension': 0.0
            }

    def get_anomaly_stats(self):
        total_la = 0
        total_ra = 0
        anomaly_neurons = 0
        total_neurons = 0

        layer_stats = {}

        for layer_name in self.la_counter_dict.keys():
            la_count = self.la_counter_dict[layer_name].sum().item()
            ra_count = self.ra_counter_dict[layer_name].sum().item()

            la_covered = self.la_covered_dict[layer_name]
            ra_covered = self.ra_covered_dict[layer_name]

            layer_anomaly_neurons = (la_covered | ra_covered).sum().item()
            layer_total_neurons = la_covered.numel()

            total_la += la_count
            total_ra += ra_count
            anomaly_neurons += layer_anomaly_neurons
            total_neurons += layer_total_neurons

            layer_stats[layer_name] = {
                'la_samples': la_count,
                'ra_samples': ra_count,
                'anomaly_neurons': layer_anomaly_neurons,
                'total_neurons': layer_total_neurons
            }

        return {
            'total_la_samples': total_la,
            'total_ra_samples': total_ra,
            'total_anomaly_samples': total_la + total_ra,
            'anomaly_neurons': anomaly_neurons,
            'total_neurons': total_neurons,
            'anomaly_neuron_rate': anomaly_neurons / total_neurons if total_neurons > 0 else 0.0,
            'layer_stats': layer_stats
        }

    def generate_report(self):
        overall_coverage = self.current
        bucket_coverage = self.get_bucket_coverage()
        anomaly_stats = self.get_anomaly_stats()

        report = "=" * 80 + "\n"
        report += "ADC Coverage Test Report\n"
        report += "=" * 80 + "\n\n"

        report += f"Overall Coverage: {overall_coverage:.2%}\n"
        report += f"Bucket Coverage (3k={3*self.k} buckets): {bucket_coverage:.2%}\n"

        zone_breakdown = self.get_zone_coverage_breakdown()
        report += f"  - Left Extension Zone: {zone_breakdown['left_extension']:.2%}\n"
        report += f"  - Core Zone: {zone_breakdown['core']:.2%}\n"
        report += f"  - Right Extension Zone: {zone_breakdown['right_extension']:.2%}\n"

        report += f"Anomaly Neuron Rate: {anomaly_stats['anomaly_neuron_rate']:.2%}\n\n"

        report += "Anomaly Samples:\n"
        report += f"  - Left Anomaly (LA): {anomaly_stats['total_la_samples']}\n"
        report += f"  - Right Anomaly (RA): {anomaly_stats['total_ra_samples']}\n"
        report += f"  - Total: {anomaly_stats['total_anomaly_samples']}\n\n"

        report += "Layer Statistics:\n"
        report += "-" * 80 + "\n"

        for layer_name, stats in anomaly_stats['layer_stats'].items():
            layer_coverage = 0.0
            if layer_name in self.bucket_covered_dict:
                bucket_covered = self.bucket_covered_dict[layer_name]
                la_covered = self.la_covered_dict[layer_name]
                ra_covered = self.ra_covered_dict[layer_name]

                num_neuron = bucket_covered.size(0)
                covered_buckets = bucket_covered.sum(dim=1).float()
                covered_buckets += la_covered.float() + ra_covered.float()
                layer_coverage = (covered_buckets / (3 * self.k + 2)).mean().item()

            anomaly_rate = (stats['la_samples'] + stats['ra_samples']) / stats['total_neurons'] if stats[
                                                                                                       'total_neurons'] > 0 else 0.0

            report += f"  {layer_name}:\n"
            report += f"    Coverage: {layer_coverage:.2%}\n"
            report += f"    Anomaly Rate: {anomaly_rate:.4f}\n"
            report += f"    LA Samples: {stats['la_samples']}, RA Samples: {stats['ra_samples']}\n"

        report += "=" * 80 + "\n"

        return report

    def update(self, all_cove_dict, delta=None):
        self.coverage_dict = all_cove_dict
        self.bucket_covered_dict = all_cove_dict['bucket_covered']
        self.la_covered_dict = all_cove_dict['la_covered']
        self.ra_covered_dict = all_cove_dict['ra_covered']

        if delta is not None:
            self.current += delta
        else:
            self.current = self.coverage(all_cove_dict)

        self.current_record.append(self.current)

    def save(self, Type):
        path_every_pic_cover_record = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_every_pic_cover_record.pth"
        path_every_pic_gain_record = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_every_pic_gain_record.pth"
        path_range = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_range_dict.pth"
        path_extended_bounds = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_extended_bounds_dict.pth"
        path_bucket_covered = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_bucket_covered_dict.pth"
        path_la_covered = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_la_covered_dict.pth"
        path_ra_covered = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_ra_covered_dict.pth"
        path_la_counter = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_la_counter_dict.pth"
        path_ra_counter = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_ra_counter_dict.pth"
        path_anomaly_samples = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_anomaly_samples.pth"
        path_everytime_current = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_everytime_current.pth"

        torch.save(self.every_pic_cover_record, path_every_pic_cover_record)
        torch.save(self.every_pic_gain_record, path_every_pic_gain_record)
        torch.save(self.range_dict, path_range)
        torch.save(self.extended_bounds_dict, path_extended_bounds)
        torch.save(self.bucket_covered_dict, path_bucket_covered)
        torch.save(self.la_covered_dict, path_la_covered)
        torch.save(self.ra_covered_dict, path_ra_covered)
        torch.save(self.la_counter_dict, path_la_counter)
        torch.save(self.ra_counter_dict, path_ra_counter)
        torch.save(self.anomaly_samples, path_anomaly_samples)
        torch.save(self.current_record, path_everytime_current)

        # 保存报告
        report_path = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_report.txt"
        with open(report_path, 'w') as f:
            f.write(self.generate_report())

    def recvoery_progress(self, Type):
        path_range = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_range_dict.pth"
        path_extended_bounds = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_extended_bounds_dict.pth"
        path_bucket_covered = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_bucket_covered_dict.pth"
        path_la_covered = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_la_covered_dict.pth"
        path_ra_covered = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_ra_covered_dict.pth"
        path_la_counter = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_la_counter_dict.pth"
        path_ra_counter = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_ra_counter_dict.pth"
        path_anomaly_samples = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_anomaly_samples.pth"
        path_everytime_current = self.save_path + f"/{self.model_name}_{Type}_{self.name}_{self.k}_everytime_current.pth"

        if os.path.exists(path_range):
            self.range_dict = torch.load(path_range)
        if os.path.exists(path_extended_bounds):
            self.extended_bounds_dict = torch.load(path_extended_bounds)
        if os.path.exists(path_bucket_covered):
            self.bucket_covered_dict = torch.load(path_bucket_covered)
        if os.path.exists(path_la_covered):
            self.la_covered_dict = torch.load(path_la_covered)
        if os.path.exists(path_ra_covered):
            self.ra_covered_dict = torch.load(path_ra_covered)
        if os.path.exists(path_la_counter):
            self.la_counter_dict = torch.load(path_la_counter)
        if os.path.exists(path_ra_counter):
            self.ra_counter_dict = torch.load(path_ra_counter)
        if os.path.exists(path_anomaly_samples):
            self.anomaly_samples = torch.load(path_anomaly_samples)
        if os.path.exists(path_everytime_current):
            self.current_record = torch.load(path_everytime_current)
            self.current = self.current_record[-1]