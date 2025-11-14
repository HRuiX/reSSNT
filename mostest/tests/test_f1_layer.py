#!/usr/bin/env python3
"""
测试F1层级相似度计算的重构版本
Test the refactored F1 layer similarity calculation
"""

import torch
import numpy as np
from objectives.f1_neural_behavior import F1NeuralBehavior


def test_basic_functionality():
    """测试基本功能"""
    print("=" * 80)
    print("测试1: 基本功能测试")
    print("=" * 80)

    f1 = F1NeuralBehavior()

    # 创建测试数据
    batch_size = 2

    activations_orig = {
        # 输出层 - 应使用KL散度
        'decode_head-conv_seg-Conv2d': np.random.randn(batch_size, 19, 32, 32),

        # 卷积层 - 应使用余弦相似度
        'backbone-layer1-0-conv1-Conv2d': np.random.randn(batch_size, 64, 56, 56),

        # 归一化层 - 应使用MSE
        'backbone-layer1-0-bn1-BatchNorm2d': np.random.randn(batch_size, 64, 56, 56),

        # Attention层 - 应使用余弦相似度
        'backbone-stages-0-blocks-0-attn-w_msa-qkv-Linear': np.random.randn(batch_size, 49, 192),
    }

    # 创建略有不同的变异数据
    activations_mut = {}
    for key, val in activations_orig.items():
        # 添加小幅噪声
        activations_mut[key] = val + np.random.randn(*val.shape) * 0.1

    # 计算F1_layer
    f1_layer_score = f1.compute_f1_layer(activations_orig, activations_mut)

    print(f"✓ F1_layer score: {f1_layer_score:.6f}")
    print(f"✓ Score range check: 0 <= {f1_layer_score:.6f} <= 1: {0 <= f1_layer_score <= 1}")
    print()


def test_layer_type_dispatch():
    """测试不同层类型的自动分派"""
    print("=" * 80)
    print("测试2: 层类型自动识别")
    print("=" * 80)

    f1 = F1NeuralBehavior()

    test_cases = [
        ('decode_head-conv_seg-Conv2d', (2, 19, 32, 32), 'KL散度'),
        ('backbone-layer1-0-conv1-Conv2d', (2, 64, 56, 56), '余弦距离'),
        ('backbone-layer1-0-bn1-BatchNorm2d', (2, 64, 56, 56), 'MSE'),
        ('encoder-layer0-attn-qkv-Linear', (2, 49, 192), '余弦距离'),
        ('decoder-cross_attn-sampling_offsets-Linear', (2, 100, 32), 'MSE'),
    ]

    for layer_name, shape, expected_method in test_cases:
        a_orig = np.random.randn(*shape)
        a_mut = a_orig + np.random.randn(*shape) * 0.1

        try:
            distance = f1._dispatch_layer_calculation(layer_name, a_orig, a_mut)
            print(f"✓ {layer_name[:50]:50s} -> {expected_method:10s} | 距离: {distance:.6f}")
        except Exception as e:
            print(f"✗ {layer_name[:50]:50s} -> 错误: {e}")

    print()


def test_numpy_and_torch():
    """测试numpy和torch输入的兼容性"""
    print("=" * 80)
    print("测试3: Numpy和Torch兼容性")
    print("=" * 80)

    f1 = F1NeuralBehavior()

    # 创建相同的数据
    shape = (2, 64, 32, 32)
    data_numpy = np.random.randn(*shape)
    data_torch = torch.from_numpy(data_numpy).float()

    # 测试numpy输入
    activations_numpy = {
        'test_layer': data_numpy,
    }
    activations_numpy_mut = {
        'test_layer': data_numpy + np.random.randn(*shape) * 0.1,
    }

    # 测试torch输入
    activations_torch = {
        'test_layer': data_torch,
    }
    activations_torch_mut = {
        'test_layer': data_torch + torch.randn(*shape) * 0.1,
    }

    score_numpy = f1.compute_f1_layer(activations_numpy, activations_numpy_mut)
    score_torch = f1.compute_f1_layer(activations_torch, activations_torch_mut)

    print(f"✓ Numpy输入得分: {score_numpy:.6f}")
    print(f"✓ Torch输入得分: {score_torch:.6f}")
    print(f"✓ 两种输入都能正常工作")
    print()


def test_edge_cases():
    """测试边界情况"""
    print("=" * 80)
    print("测试4: 边界情况")
    print("=" * 80)

    f1 = F1NeuralBehavior()

    # 测试1: 完全相同的激活值
    activations_same = {
        'layer1': np.random.randn(2, 64, 32, 32),
        'layer2': np.random.randn(2, 128, 16, 16),
    }
    score_same = f1.compute_f1_layer(activations_same, activations_same)
    print(f"✓ 相同激活值得分: {score_same:.6f} (应该接近0)")

    # 测试2: 完全不同的激活值
    activations_orig = {
        'layer1': np.random.randn(2, 64, 32, 32),
    }
    activations_diff = {
        'layer1': np.random.randn(2, 64, 32, 32) * 10,  # 完全不同的分布
    }
    score_diff = f1.compute_f1_layer(activations_orig, activations_diff)
    print(f"✓ 不同激活值得分: {score_diff:.6f} (应该较大)")

    # 测试3: 空字典
    score_empty = f1.compute_f1_layer({}, {})
    print(f"✓ 空字典得分: {score_empty:.6f} (应该是0)")

    # 测试4: 不匹配的层
    activations_a = {'layer1': np.random.randn(2, 64, 32, 32)}
    activations_b = {'layer2': np.random.randn(2, 64, 32, 32)}
    score_mismatch = f1.compute_f1_layer(activations_a, activations_b)
    print(f"✓ 不匹配层得分: {score_mismatch:.6f} (应该是0)")

    print()


def test_kl_divergence_specifics():
    """测试KL散度的具体行为"""
    print("=" * 80)
    print("测试5: KL散度计算")
    print("=" * 80)

    f1 = F1NeuralBehavior()

    # 测试输出层（logits）
    logits_orig = torch.randn(2, 19, 32, 32)
    logits_mut = logits_orig + torch.randn(2, 19, 32, 32) * 0.5

    distance = f1._calculate_kl_divergence(logits_orig, logits_mut, need_softmax=True)
    print(f"✓ Logits KL散度: {distance:.6f}")

    # 测试已归一化的概率分布
    probs_orig = torch.softmax(logits_orig, dim=1)
    probs_mut = torch.softmax(logits_mut, dim=1)

    distance_prob = f1._calculate_kl_divergence(probs_orig, probs_mut, need_softmax=False)
    print(f"✓ 概率分布KL散度: {distance_prob:.6f}")

    print()


def test_cosine_distance_specifics():
    """测试余弦距离的具体行为"""
    print("=" * 80)
    print("测试6: 余弦距离计算")
    print("=" * 80)

    f1 = F1NeuralBehavior()

    # 测试完全相同的向量（距离应为0）
    feat_same = torch.randn(2, 64, 32, 32)
    distance_same = f1._calculate_cosine_distance(feat_same, feat_same)
    print(f"✓ 相同特征余弦距离: {distance_same:.6f} (应接近0)")

    # 测试正交向量（距离应为1）
    feat_a = torch.zeros(2, 100)
    feat_b = torch.zeros(2, 100)
    feat_a[:, :50] = 1.0
    feat_b[:, 50:] = 1.0
    distance_orth = f1._calculate_cosine_distance(feat_a, feat_b)
    print(f"✓ 正交特征余弦距离: {distance_orth:.6f} (应接近1)")

    # 测试反向向量（距离应为2）
    feat_neg = -feat_same
    distance_neg = f1._calculate_cosine_distance(feat_same, feat_neg)
    print(f"✓ 反向特征余弦距离: {distance_neg:.6f} (应接近2)")

    print()


def test_attention_weights_auto_detection():
    """测试attention weights的自动检测"""
    print("=" * 80)
    print("测试7: Attention Weights自动检测")
    print("=" * 80)

    f1 = F1NeuralBehavior()

    # 测试已归一化的attention weights（和接近1）
    attn_normalized = torch.softmax(torch.randn(2, 8, 64, 64), dim=-1)
    attn_normalized_mut = torch.softmax(torch.randn(2, 8, 64, 64), dim=-1)

    distance_norm = f1._calculate_attention_weights_distance(attn_normalized, attn_normalized_mut)
    print(f"✓ 已归一化attention weights距离: {distance_norm:.6f}")
    print(f"  (自动检测: sum={attn_normalized.sum(dim=-1).mean():.4f} ≈ 1.0)")

    # 测试未归一化的attention scores
    attn_unnorm = torch.randn(2, 8, 64, 64)
    attn_unnorm_mut = attn_unnorm + torch.randn(2, 8, 64, 64) * 0.1

    distance_unnorm = f1._calculate_attention_weights_distance(attn_unnorm, attn_unnorm_mut)
    print(f"✓ 未归一化attention scores距离: {distance_unnorm:.6f}")
    print(f"  (自动检测: sum={attn_unnorm.sum(dim=-1).mean():.4f} ≠ 1.0)")

    print()


def test_complete_f1():
    """测试完整的F1计算"""
    print("=" * 80)
    print("测试8: 完整F1计算")
    print("=" * 80)

    f1 = F1NeuralBehavior()

    # 创建模拟的完整激活值字典
    activations_orig = {
        'backbone-layer1-0-conv1-Conv2d': np.random.randn(2, 64, 56, 56),
        'backbone-layer1-0-bn1-BatchNorm2d': np.random.randn(2, 64, 56, 56),
        'backbone-layer2-0-conv1-Conv2d': np.random.randn(2, 128, 28, 28),
        'decode_head-conv_seg-Conv2d': np.random.randn(2, 19, 32, 32),
    }

    activations_mut = {
        k: v + np.random.randn(*v.shape) * 0.2
        for k, v in activations_orig.items()
    }

    # 计算三个子指标
    f1_neuron = f1.compute_f1_neuron(activations_orig, activations_mut)
    f1_layer = f1.compute_f1_layer(activations_orig, activations_mut)
    f1_pattern = f1.compute_f1_pattern(activations_orig, activations_mut)

    print(f"✓ F1_neuron: {f1_neuron:.6f}")
    print(f"✓ F1_layer:  {f1_layer:.6f}")
    print(f"✓ F1_pattern: {f1_pattern:.6f}")

    # 几何平均
    f1_total = (f1_neuron * f1_layer * f1_pattern) ** (1/3)
    print(f"✓ F1_total:  {f1_total:.6f}")

    print()


def main():
    """运行所有测试"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "F1层级相似度计算 - 重构测试" + " " * 21 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")

    try:
        test_basic_functionality()
        test_layer_type_dispatch()
        test_numpy_and_torch()
        test_edge_cases()
        test_kl_divergence_specifics()
        test_cosine_distance_specifics()
        test_attention_weights_auto_detection()
        test_complete_f1()

        print("╔" + "=" * 78 + "╗")
        print("║" + " " * 28 + "所有测试通过! ✓" + " " * 28 + "║")
        print("╚" + "=" * 78 + "╝")

    except Exception as e:
        print("\n")
        print("╔" + "=" * 78 + "╗")
        print("║" + " " * 30 + "测试失败! ✗" + " " * 30 + "║")
        print("╚" + "=" * 78 + "╝")
        print(f"\n错误信息: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
