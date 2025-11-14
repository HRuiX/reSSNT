"""
F1: Multi-granularity Neural Behavior Coverage Metric
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Union
from scipy.stats import entropy
from functools import lru_cache

try:
    from hdbscan import HDBSCAN
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

from scipy.optimize import linear_sum_assignment
from rich.console import Console

console = Console()


class F1NeuralBehavior:
    """
    F1 Objective: Neuron-level, Layer-level, Cross-layer Pattern Coverage
    """

    def __init__(self, top_k: int, epsilon=1e-8):
        self.top_k = top_k
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon = epsilon

    def _convert_to_tensor(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Unified tensor conversion - ensures data is on GPU"""
        if isinstance(data, torch.Tensor):
            if data.device != self.device:
                return data.to(self.device, non_blocking=True)
            return data
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(self.device, dtype=torch.float32, non_blocking=True)
        return data

    def _convert_dict_to_tensors(self, data_dict: Dict) -> Dict:
        """Convert all activations to GPU tensors in one go"""
        return {
            k: self._convert_to_tensor(v)
            for k, v in data_dict.items()
        }

    def compute_f1_neuron(
            self,
            activations_orig: Dict[str, torch.Tensor],
            activations_mut: Dict[str, torch.Tensor]
    ) -> float:
        if not activations_orig:
            return 0.0

        # Collect all layer differences in one pass
        layer_diffs = []

        for layer_name in activations_orig.keys():
            if layer_name not in activations_mut:
                continue

            a_orig = activations_orig[layer_name].flatten()
            a_mut = activations_mut[layer_name].flatten()

            # Compute difference on GPU
            neuron_diffs = torch.abs(a_orig - a_mut)

            # Get statistics
            min_diff = neuron_diffs.min()
            max_diff = neuron_diffs.max()
            diff_range = max_diff - min_diff

            # Normalize
            if diff_range < self.epsilon:
                layer_diffs.append(0.0)
            else:
                normalized_diff = (neuron_diffs.mean() - min_diff) / (diff_range + self.epsilon)
                layer_diffs.append(normalized_diff)

        if not layer_diffs:
            return 0.0

        # Convert to tensor for GPU computation, then get final result
        layer_scores = torch.tensor(layer_diffs, device=self.device, dtype=torch.float32)
        return float(layer_scores.mean().item())

    def _to_tensor(self, arr: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Deprecated: use _convert_to_tensor instead. Kept for compatibility."""
        return self._convert_to_tensor(arr)

    def _calculate_kl_divergence(self, a_orig: torch.Tensor, a_mut: torch.Tensor, need_softmax: bool = True) -> float:
        """Compute KL divergence on GPU"""
        if need_softmax:
            ndim = len(a_orig.shape)
            dim = {1: 0, 2: -1, 3: -1, 4: 1}.get(ndim, -1)

            if ndim == 2 and a_orig.shape[0] == 1:
                dim = -1
            elif ndim == 2:
                dim = 1 if a_orig.shape[1] > 1 else -1

            p_orig = F.softmax(a_orig, dim=dim)
            p_mut = F.softmax(a_mut, dim=dim)
        else:
            p_orig = a_orig + self.epsilon
            p_mut = a_mut + self.epsilon

        # Use log_softmax for numerical stability
        kl_div = F.kl_div(p_mut.log(), p_orig, reduction='batchmean')
        return float(kl_div.item())

    def _calculate_cosine_distance(self, a_orig: torch.Tensor, a_mut: torch.Tensor) -> float:
        batch_size = a_orig.shape[0]
        a_orig_flat = a_orig.reshape(batch_size, -1)
        a_mut_flat = a_mut.reshape(batch_size, -1)

        cos_sim = F.cosine_similarity(a_orig_flat, a_mut_flat, dim=1)
        cos_sim_mean = cos_sim.mean()
        distance = 1.0 - cos_sim_mean

        return float(distance.item())

    def _calculate_mse_loss(self, a_orig: torch.Tensor, a_mut: torch.Tensor) -> float:
        mse = F.mse_loss(a_orig, a_mut)
        return float(mse.item())

    def _dispatch_layer_calculation(self, layer_name: str, a_orig: torch.Tensor,
                                   a_mut: torch.Tensor) -> float:
        """
        Dispatch layer-specific calculation. Assumes tensors are on correct device.
        """
        layer_lower = layer_name.lower()

        # Use KL divergence for specific layer types
        if ('conv_seg' in layer_lower or 'cls_embed' in layer_lower or
            'mask_embed' in layer_lower or 'attention_weights' in layer_lower):
            return self._calculate_kl_divergence(a_orig, a_mut, need_softmax=True)

        # Default to cosine distance
        return self._calculate_cosine_distance(a_orig, a_mut)

    def compute_f1_layer(self, activations_orig: Dict[str, torch.Tensor],
                        activations_mut: Dict[str, torch.Tensor]) -> float:
        """
        Optimized layer-level computation using GPU.
        Assumes inputs are already converted to tensors on the correct device.
        """
        if not activations_orig:
            return 0.0

        layer_deltas = []

        for layer_name in activations_orig.keys():
            if layer_name not in activations_mut:
                continue

            a_orig = activations_orig[layer_name]
            a_mut = activations_mut[layer_name]

            # Shape check
            if a_orig.shape != a_mut.shape:
                console.print(f"  [yellow]⚠[/yellow] Shape mismatch {layer_name}: [dim]{a_orig.shape} vs {a_mut.shape}[/dim]")
                continue

            try:
                delta = self._dispatch_layer_calculation(layer_name, a_orig, a_mut)
                layer_deltas.append(delta)
            except Exception as e:
                console.print(f"  [red]✗[/red] Error {layer_name}: [dim]{e}[/dim]")
                continue

        if not layer_deltas:
            return 0.0

        # Use GPU for normalization
        layer_deltas_tensor = torch.tensor(layer_deltas, device=self.device, dtype=torch.float32)
        min_delta = layer_deltas_tensor.min()
        max_delta = layer_deltas_tensor.max()
        diff_range = max_delta - min_delta

        if diff_range < self.epsilon:
            return 0.0

        normalized = (layer_deltas_tensor.mean() - min_delta) / (diff_range + self.epsilon)
        return float(normalized.item())

    def _prepare_features_for_clustering(self, activations: Dict[str, np.ndarray]) -> np.ndarray:
        sorted_layers = sorted(activations.keys())
        feature_list = []

        max_dim = 0
        for layer_name in sorted_layers:
            activation = activations[layer_name]
            if isinstance(activation, torch.Tensor):
                activation = activation.cpu().numpy()

            if len(activation.shape) > 1:
                activation = activation.flatten()

            max_dim = max(max_dim, len(activation))
            norm = np.linalg.norm(activation)
            if norm > self.epsilon:
                activation = activation / norm

            feature_list.append(activation)

        n_features = len(feature_list)
        unified_features = np.zeros((n_features, max_dim), dtype=np.float32)

        for i, feat in enumerate(feature_list):
            feat_len = len(feat)
            unified_features[i, :feat_len] = feat

        return unified_features

    def _cluster_with_hdbscan(
            self,
            features: np.ndarray,
            min_cluster_size: int = None
    ) -> Tuple[np.ndarray, int, List[np.ndarray], List[np.ndarray]]:
        if not HDBSCAN_AVAILABLE:
            return self._fallback_clustering(features)

        n_samples = features.shape[0]

        if min_cluster_size is None:
            if n_samples < 50:
                min_cluster_size = max(3, n_samples // 10)
            elif n_samples < 500:
                min_cluster_size = max(5, n_samples // 20)
            else:
                min_cluster_size = min(100, n_samples // 50)

        min_cluster_size = min(min_cluster_size, n_samples - 1)

        if min_cluster_size < 2:
            return self._fallback_clustering(features)

        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=None,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        
        cluster_labels = clusterer.fit_predict(features)
        

        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

        cluster_centers = []
        cluster_members = []

        for cluster_id in range(n_clusters):
            mask = cluster_labels == cluster_id
            members = features[mask]
            cluster_members.append(members)
            cluster_centers.append(np.mean(members, axis=0))

        return cluster_labels, n_clusters, cluster_centers, cluster_members

    def _compute_cluster_features(
            self,
            cluster_centers: List[np.ndarray],
            cluster_members: List[np.ndarray]
    ) -> Dict:
        n_clusters = len(cluster_centers)

        if n_clusters == 0:
            return {
                'compactness': np.array([]),
                'separation': 0.0,
                'sizes': np.array([])
            }

        compactness = np.array([
            np.mean(np.linalg.norm(members - center, axis=1)) if len(members) > 0 else 0.0
            for center, members in zip(cluster_centers, cluster_members)
        ])

        if n_clusters > 1:
            centers_array = np.array(cluster_centers)
            diff = centers_array[:, np.newaxis, :] - centers_array[np.newaxis, :, :]
            distances = np.linalg.norm(diff, axis=2)
            mask = np.triu(np.ones((n_clusters, n_clusters), dtype=bool), k=1)
            separation = float(np.mean(distances[mask]))
        else:
            separation = 0.0

        sizes = np.array([len(members) for members in cluster_members])

        return {
            'compactness': compactness,
            'separation': separation,
            'sizes': sizes
        }

    def _compute_cluster_difference(
            self,
            centers_orig: List[np.ndarray],
            centers_mut: List[np.ndarray],
            features_orig: Dict,
            features_mut: Dict
    ) -> float:
        N_orig = len(centers_orig)
        N_mut = len(centers_mut)

        if N_orig == 0 or N_mut == 0:
            return 1.0

        # 1. Cluster count difference
        D_number = abs(N_orig - N_mut) / max(N_orig, N_mut)

        # 2.Vectorized cost matrix computation
        if N_orig > 0 and N_mut > 0:
            centers_orig_arr = np.array(centers_orig)
            centers_mut_arr = np.array(centers_mut)

            diff = centers_orig_arr[:, np.newaxis, :] - centers_mut_arr[np.newaxis, :, :]
            cost_matrix = np.linalg.norm(diff, axis=2)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matched_distances = cost_matrix[row_ind, col_ind]
            D_center = np.mean(matched_distances)

            avg_separation = (features_orig['separation'] + features_mut['separation']) / 2
            if avg_separation > self.epsilon:
                D_center = D_center / (avg_separation + self.epsilon)
            else:
                D_center = 0.0
        else:
            D_center = 1.0
            row_ind, col_ind = None, None

        if row_ind is not None and len(features_orig['compactness']) > 0 and len(features_mut['compactness']) > 0:
            comp_orig_matched = features_orig['compactness'][row_ind]
            comp_mut_matched = features_mut['compactness'][col_ind]
            max_comp = np.maximum(comp_orig_matched, comp_mut_matched)

            mask = max_comp > self.epsilon
            if np.any(mask):
                diffs = np.abs(comp_orig_matched - comp_mut_matched)[mask] / max_comp[mask]
                D_compactness = float(np.mean(diffs))
            else:
                D_compactness = 0.0
        else:
            D_compactness = 0.0

        sep_orig = features_orig['separation']
        sep_mut = features_mut['separation']
        max_sep = max(sep_orig, sep_mut)
        D_separation = abs(sep_orig - sep_mut) / max_sep if max_sep > self.epsilon else 0.0

        sizes_orig = features_orig['sizes']
        sizes_mut = features_mut['sizes']

        if len(sizes_orig) > 0 and len(sizes_mut) > 0:
            P_orig = sizes_orig / np.sum(sizes_orig)
            P_mut = sizes_mut / np.sum(sizes_mut)

            max_len = max(len(P_orig), len(P_mut))
            P_orig_padded = np.pad(P_orig, (0, max_len - len(P_orig)), 'constant')
            P_mut_padded = np.pad(P_mut, (0, max_len - len(P_mut)), 'constant')

            M = (P_orig_padded + P_mut_padded) / 2
            kl_pm = entropy(P_orig_padded + self.epsilon, M + self.epsilon)
            kl_qm = entropy(P_mut_padded + self.epsilon, M + self.epsilon)
            D_size = 0.5 * kl_pm + 0.5 * kl_qm
        else:
            D_size = 1.0

        components = [D_number, D_center, D_compactness, D_separation, D_size]
        components = [max(c, self.epsilon) for c in components]
        F1_pattern_cc = np.power(np.prod(components), 1.0 / len(components))

        return float(F1_pattern_cc)

    def compute_f1_pattern(
            self,
            activations_orig: Dict[str, np.ndarray],
            activations_mut: Dict[str, np.ndarray]
    ) -> float:
        if len(activations_orig) == 0 or len(activations_mut) == 0:
            return 0.0

        features_orig = self._prepare_features_for_clustering(activations_orig)
        features_mut = self._prepare_features_for_clustering(activations_mut)

        labels_orig, n_clusters_orig, centers_orig, members_orig = \
            self._cluster_with_hdbscan(features_orig)

        labels_mut, n_clusters_mut, centers_mut, members_mut = \
            self._cluster_with_hdbscan(features_mut)

        cluster_features_orig = self._compute_cluster_features(centers_orig, members_orig)
        cluster_features_mut = self._compute_cluster_features(centers_mut, members_mut)

        pattern_diff = self._compute_cluster_difference(
            centers_orig, centers_mut,
            cluster_features_orig, cluster_features_mut
        )

        return pattern_diff

    def compute(
            self,
            ori_layer_output_dict: Dict,
            muta_layer_output_dict: Dict,
            **kwargs
    ) -> float:
        """
        Compute complete F1 objective function value (GPU-optimized)

        F1 = (F1_neuron * F1_layer)^(1/2)

        Args:
            ori_layer_output_dict: {layer_name: activation}
            muta_layer_output_dict: {layer_name: activation}

        Returns:
            F1 value
        """
        # Convert all data to GPU tensors once at the beginning
        with torch.no_grad():  # Disable gradient computation for inference
            ori_layer_output_dict = self._convert_dict_to_tensors(ori_layer_output_dict)
            muta_layer_output_dict = self._convert_dict_to_tensors(muta_layer_output_dict)

            f1_neuron = self.compute_f1_neuron(ori_layer_output_dict, muta_layer_output_dict)
            f1_layer = self.compute_f1_layer(ori_layer_output_dict, muta_layer_output_dict)
            # f1_pattern = self.compute_f1_pattern(ori_layer_output_dict, muta_layer_output_dict)

            # Compute final F1 on GPU
            f1_neuron_t = torch.tensor(f1_neuron, device=self.device, dtype=torch.float32)
            f1_layer_t = torch.tensor(f1_layer, device=self.device, dtype=torch.float32)

            f1 = torch.sqrt(f1_neuron_t * f1_layer_t)
            # f1 = (f1_neuron * f1_layer * f1_pattern) ** (1 / 3)

            return float(f1.item())