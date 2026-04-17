"""Official-style NFA tree utilities for UFlow strict alignment."""

from __future__ import annotations

import itertools as it
import sys
from typing import Sequence

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from mpmath import mp
from skimage.morphology import max_tree


sys.setrecursionlimit(100000)
mp.dps = 15


def compute_nfa_anomaly_score_tree(
    z: Sequence[torch.Tensor],
    target_size: int | Sequence[int] = 256,
    upsample_mode: str = 'bilinear',
) -> torch.Tensor:
    """Compute the strict UFlow log(NFA) anomaly score.

    This mirrors the original `mtailanian/uflow` tree-based region scoring
    path. The returned value is ``-log10(NFA)`` where larger is more anomalous.
    """
    target_h, target_w = _to_hw(target_size)

    log_prob = []
    for img_idx in range(z[0].shape[0]):
        log_prob_scales = []
        for zi in z:
            nfa_tree = NFATree(zi[img_idx])
            log_prob_scales.append(nfa_tree.compute_log_prob_map())

        upsampled = []
        for log_prob_scale in log_prob_scales:
            log_prob_tensor = torch.from_numpy(log_prob_scale).nan_to_num(0.0).unsqueeze(0).unsqueeze(0)
            kwargs = {'align_corners': False} if 'nearest' not in upsample_mode else {}
            upsampled.append(
                F.interpolate(
                    log_prob_tensor,
                    size=(target_h, target_w),
                    mode=upsample_mode,
                    **kwargs,
                ))
        log_prob.append(torch.cat(upsampled, dim=1))

    log_prob = torch.cat(log_prob, dim=0)
    log_prob = log_prob.amin(dim=1, keepdim=True)

    log_n_tests = compute_number_of_tests([int(zi.shape[-2] * zi.shape[-1]) for zi in z])
    log_nfa = log_prob + log_n_tests
    return -log_nfa


class NFATree:
    """Build the UFlow NFA tree from one latent tensor."""

    def __init__(self, zi: torch.Tensor):
        self.n_channels = int(zi.shape[0])
        self.zi2_rav = zi.reshape(self.n_channels, -1).detach().cpu().numpy() ** 2

        score = torch.mean(zi ** 2, dim=0).detach().cpu().numpy()
        self.original_shape = tuple(score.shape)
        self.tree = self.build_tree(score)

    def compute_log_prob_map(self) -> np.ndarray:
        self.compute_log_prob()

        self.pfa_prune()
        keep_merging = self.pfa_merge()
        while keep_merging:
            self.pfa_prune()
            keep_merging = self.pfa_merge()
        self.pfa_prune()

        log_prob_map = np.empty(self.original_shape[0] * self.original_shape[1], dtype=np.float32)
        log_prob_map[:] = np.nan

        for log_prob, pixels in self.get_final_clusters().items():
            log_prob_map[pixels] = log_prob

        return log_prob_map.reshape(self.original_shape)

    def compute_log_prob(self) -> None:
        zi2_sum = np.sum(self.zi2_rav, axis=0)

        for node in self.tree.nodes:
            region = self.tree.nodes[node]['pixels']
            zi2_min = zi2_sum[region].min()
            ratio = zi2_min / self.n_channels
            log_prob = -(self.n_channels / 2) * (ratio - 1 - np.log(ratio)) / np.log(10)
            self.tree.nodes[node]['log_prob'] = len(region) * log_prob

    def build_tree(self, score: np.ndarray) -> nx.DiGraph:
        parents, pixel_indices = max_tree(score, connectivity=1)
        parents_rav = parents.ravel()
        score_rav = score.ravel()

        tree = nx.DiGraph()
        tree.add_nodes_from(pixel_indices)
        for node in tree.nodes():
            tree.nodes[node]['score'] = score_rav[node]
        tree.add_edges_from((node, parents_rav[node]) for node in pixel_indices[1:])

        self.prune(tree, pixel_indices[0])
        self.accumulate(tree, pixel_indices[0])
        return tree

    def prune(self, graph: nx.DiGraph, starting_node: int) -> None:
        value = graph.nodes[starting_node]['score']
        cluster_nodes = [starting_node]
        for predecessor in [node for node in graph.predecessors(starting_node)]:
            if graph.nodes[predecessor]['score'] == value:
                cluster_nodes.append(predecessor)
                graph.remove_node(predecessor)
            else:
                self.prune(graph, predecessor)
        graph.nodes[starting_node]['pixels'] = cluster_nodes

    def accumulate(self, graph: nx.DiGraph, starting_node: int) -> list[int]:
        pixels = graph.nodes[starting_node]['pixels']
        for predecessor in graph.predecessors(starting_node):
            pixels.extend(self.accumulate(graph, predecessor))
        return pixels

    def get_branch(self, starting_node: int) -> list[int]:
        branch = [starting_node]
        successors = [node for node in self.tree.successors(starting_node)]

        if len(successors) == 0:
            return branch
        if len(successors) != 1:
            raise AssertionError('Node has more than one successor')

        successor = successors[0]
        is_only_child = len([node for node in self.tree.predecessors(successor)]) == 1
        if is_only_child:
            branch.extend(self.get_branch(successor))
        return branch

    def get_final_clusters(self) -> dict[float, list[int]]:
        leaves = [node for node in self.tree.pred if len(self.tree.pred[node]) == 0]
        final_clusters = {}
        for leaf in leaves:
            branch_nodes = self.get_branch(leaf)
            branch_log_probs = [self.tree.nodes[node]['log_prob'] for node in branch_nodes]
            chosen_node = branch_nodes[int(np.argmin(branch_log_probs))]
            final_clusters[self.tree.nodes[chosen_node]['log_prob']] = self.tree.nodes[chosen_node]['pixels']
        return final_clusters

    def pfa_prune(self) -> None:
        leaves = [node for node in self.tree.pred if len(self.tree.pred[node]) == 0]
        for leaf in leaves:
            branch_nodes = self.get_branch(leaf)
            branch_log_probs = [self.tree.nodes[node]['log_prob'] for node in branch_nodes]
            chosen_index = int(np.argmin(branch_log_probs))
            for index, branch_node in enumerate(branch_nodes):
                if index == chosen_index:
                    continue
                self.tree.add_edges_from(it.product(self.tree.predecessors(branch_node), self.tree.successors(branch_node)))
                self.tree.remove_node(branch_node)

    def pfa_merge(self) -> bool:
        merged = False
        bifurcations = [node for node in self.tree.pred if len(self.tree.pred[node]) > 1]
        for bifurcation in bifurcations:
            predecessors = [node for node in self.tree.predecessors(bifurcation)]
            if np.sum([len([parent for parent in self.tree.predecessors(node)]) for node in predecessors]) > 0:
                continue
            predecessor_log_probs = [self.tree.nodes[node]['log_prob'] for node in predecessors]
            if self.tree.nodes[bifurcation]['log_prob'] <= np.min(predecessor_log_probs):
                merged = True
                for predecessor in predecessors:
                    self.tree.add_edges_from(
                        it.product(self.tree.predecessors(predecessor), self.tree.successors(predecessor)))
                    self.tree.remove_node(predecessor)
        return merged


def compute_number_of_tests(polyomino_sizes: int | list[int]) -> float:
    """Approximate the number of possible connected regions."""
    alpha = mp.mpf(0.316915)
    beta = mp.mpf(4.062570)

    if not isinstance(polyomino_sizes, list):
        polyomino_sizes = [polyomino_sizes]

    n_test = mp.mpf(0)
    for region_size in polyomino_sizes:
        n_test_i = mp.mpf(0)
        for size in range(1, region_size + 1):
            size_mp = mp.mpf(size)
            n_test_i += alpha * beta ** size_mp / size_mp
        n_test += n_test_i * region_size

    return float(np.array(mp.log10(n_test), dtype=np.float32))


def _to_hw(target_size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(target_size, int):
        return target_size, target_size
    if len(target_size) != 2:
        raise ValueError(f'Expected target_size to have length 2, got {target_size!r}')
    return int(target_size[0]), int(target_size[1])
