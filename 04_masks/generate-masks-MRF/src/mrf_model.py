"""
MRF model: unary/pairwise potential containers, energy computation,
and graph-cut MAP inference via gco-wrapper.
"""

import numpy as np
from gco import cut_general_graph


NUM_LABELS = 3


class MRFModel:
    """
    Pairwise MRF over a 2D label grid.

    Energy:
        E(X) = sum_i psi_u(x_i) + sum_(i,j) psi_p(x_i, x_j)
    """

    def __init__(self, height, width, class_freqs, pairwise_potentials,
                 target_ratio=None, lambda_ratio=1.0, toroidal=True):
        """
        Args:
            height, width: grid dimensions
            class_freqs: array of shape (NUM_LABELS,), empirical P(c)
            pairwise_potentials: dict with keys 'right','down','diag_dr','diag_dl',
                                 each a (NUM_LABELS, NUM_LABELS) cost matrix
            target_ratio: optional (NUM_LABELS,) target class proportions
            lambda_ratio: weight for the proportion constraint bias
            toroidal: whether to use toroidal boundary conditions
        """
        self.height = height
        self.width = width
        self.n_sites = height * width
        self.class_freqs = np.array(class_freqs, dtype=np.float64)
        self.pairwise = pairwise_potentials
        self.target_ratio = target_ratio
        self.lambda_ratio = lambda_ratio
        self.toroidal = toroidal

    def compute_unary(self, fixed_labels=None):
        """
        Compute unary cost matrix of shape (n_sites, NUM_LABELS).

        Args:
            fixed_labels: optional array of shape (height, width) with values
                          in {0,1,2,-1}. Sites with label >= 0 are pinned
                          (near-zero cost for that label, high cost for others).
        """
        # Base unary from class frequency
        freq_cost = -np.log(np.clip(self.class_freqs, 1e-8, 1.0))
        unary = np.tile(freq_cost, (self.n_sites, 1)).copy()

        # Proportion bias
        if self.target_ratio is not None:
            target = np.array(self.target_ratio, dtype=np.float64)
            target = np.clip(target, 1e-8, 1.0)
            bias = self.lambda_ratio * np.log(target / np.clip(self.class_freqs, 1e-8, 1.0))
            unary -= bias  # lower cost for labels we want more of

        # Pin fixed labels
        if fixed_labels is not None:
            flat = fixed_labels.ravel()
            for idx in range(self.n_sites):
                if flat[idx] >= 0:
                    # Very high cost for all labels except the fixed one
                    unary[idx, :] = 1e6
                    unary[idx, flat[idx]] = 0.0

        return unary.astype(np.float64)

    def build_edges_and_weights(self):
        """
        Build edge list and per-edge pairwise cost matrices for the grid,
        with toroidal wrapping if enabled.

        Returns:
            edges: (n_edges, 2) int32 array of node index pairs
            edge_weights: (n_edges, NUM_LABELS, NUM_LABELS) float64 cost matrices
        """
        h, w = self.height, self.width
        edges = []
        edge_costs = []

        directions = [
            ("right", 0, 1),
            ("down", 1, 0),
            ("diag_dr", 1, 1),
            ("diag_dl", 1, -1),
        ]

        for name, di, dj in directions:
            cost_matrix = self.pairwise[name]
            for i in range(h):
                for j in range(w):
                    ni = i + di
                    nj = j + dj
                    if self.toroidal:
                        ni = ni % h
                        nj = nj % w
                    else:
                        if ni < 0 or ni >= h or nj < 0 or nj >= w:
                            continue
                    idx1 = i * w + j
                    idx2 = ni * w + nj
                    if idx1 != idx2:  # avoid self-loops at corners
                        edges.append((idx1, idx2))
                        edge_costs.append(cost_matrix)

        edges = np.array(edges, dtype=np.int32)
        edge_costs = np.array(edge_costs, dtype=np.float64)
        return edges, edge_costs

    def map_inference(self, fixed_labels=None, n_iter=-1, pairwise_scale=1.0):
        """
        MAP inference via alpha-expansion graph cuts.

        Args:
            fixed_labels: optional (height, width) array with -1 for unknown
            n_iter: number of iterations (-1 = until convergence)
            pairwise_scale: global scaling for pairwise costs

        Returns:
            labels: (height, width) int array
        """
        unary = self.compute_unary(fixed_labels)
        edges, edge_costs = self.build_edges_and_weights()

        # Scale pairwise
        edge_costs = edge_costs * pairwise_scale

        # gco expects: unary (n_sites, n_labels), edges (n_edges, 2),
        # edge_weights as a single pairwise matrix or per-edge
        # For cut_general_graph we need a single pairwise + edge weights scalar
        # Use average pairwise as the label cost matrix, and edge-specific weights

        # Compute average pairwise cost matrix and symmetrize (gco requirement)
        avg_pairwise = np.mean(edge_costs, axis=0)
        avg_pairwise = (avg_pairwise + avg_pairwise.T) / 2.0

        n_edges = len(edges)
        edge_weights = np.ones(n_edges, dtype=np.float64)

        result = cut_general_graph(
            edges, edge_weights, unary, avg_pairwise,
            n_iter=n_iter, algorithm="expansion"
        )
        return result.reshape(self.height, self.width)

    def compute_energy(self, labels):
        """Compute total MRF energy for a given label assignment."""
        h, w = self.height, self.width
        flat = labels.ravel()

        # Unary
        freq_cost = -np.log(np.clip(self.class_freqs, 1e-8, 1.0))
        energy_unary = sum(freq_cost[flat[i]] for i in range(self.n_sites))

        # Pairwise
        energy_pairwise = 0.0
        directions = [
            ("right", 0, 1), ("down", 1, 0),
            ("diag_dr", 1, 1), ("diag_dl", 1, -1),
        ]
        for name, di, dj in directions:
            C = self.pairwise[name]
            for i in range(h):
                for j in range(w):
                    ni = (i + di) % h
                    nj = (j + dj) % w
                    energy_pairwise += C[labels[i, j], labels[ni, nj]]

        return energy_unary + energy_pairwise
