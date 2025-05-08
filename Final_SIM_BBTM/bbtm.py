import numpy as np
import scipy.sparse as sp
from itertools import combinations
from pypolyagamma import PyPolyaGamma
from pbcj import generate_mus_sigmas
from scipy.stats import kendalltau, spearmanr

def get_item_pairs(n_items):
    return list(combinations(range(n_items), 2))

class ComparisonGenerator:
    def __init__(self, n_items, mu_vector, sigma_vector, seed=None):
        self.n_items = n_items
        self.mu_vector = mu_vector
        self.sigma_vector = sigma_vector
        self.rng = np.random.default_rng(seed)

    def generate_winner_01(self, item_1, item_2):
        theta_1 = self.rng.normal(self.mu_vector[item_1], self.sigma_vector[item_1])
        theta_2 = self.rng.normal(self.mu_vector[item_2], self.sigma_vector[item_2])
        return 1 if theta_1 > theta_2 else 0

def build_design_matrix(comparisons, n_items):
    n_comparisons = len(comparisons)
    row_ind = []
    col_ind = []
    data = []
    for idx, (i, j, winner) in enumerate(comparisons):
        row_ind.append(idx)
        col_ind.append(i)
        data.append(1 if winner == i else -1)
        row_ind.append(idx)
        col_ind.append(j)
        data.append(-1 if winner == i else 1)
    return sp.csr_matrix((data, (row_ind, col_ind)), shape=(n_comparisons, n_items))

def round_robin_select(n_items, total_comparisons, seed=None):
    if not hasattr(round_robin_select, 'pairs'):
        rng = np.random.default_rng(seed)
        pairs = list(combinations(range(n_items), 2))
        rng.shuffle(pairs)
        round_robin_select.pairs = pairs
        round_robin_select.pair_counter = {pair: 0 for pair in pairs}
        round_robin_select.current_index = 0
        pairs_per_cycle = len(pairs)
        round_robin_select.cycles = total_comparisons // pairs_per_cycle
    while round_robin_select.current_index < len(round_robin_select.pairs):
        pair = round_robin_select.pairs[round_robin_select.current_index]
        if round_robin_select.pair_counter[pair] < round_robin_select.cycles:
            round_robin_select.pair_counter[pair] += 1
            round_robin_select.current_index += 1
            if round_robin_select.current_index >= len(round_robin_select.pairs):
                round_robin_select.current_index = 0
            return pair[0], pair[1]
        
def no_repeating_pairs_select(n_items, total_comparisons, current_comparison_idx, seed):
    pairs_per_cycle = (n_items * (n_items - 1)) // 2
    cycle_idx = current_comparison_idx // pairs_per_cycle
    pair_idx = current_comparison_idx % pairs_per_cycle
    if not hasattr(no_repeating_pairs_select, 'cycle_pairs'):
        no_repeating_pairs_select.cycle_pairs = {}
    cycle_key = cycle_idx
    if cycle_key not in no_repeating_pairs_select.cycle_pairs:
        rng = np.random.default_rng(seed + cycle_idx)
        pairs = list(combinations(range(n_items), 2))
        rng.shuffle(pairs)
        no_repeating_pairs_select.cycle_pairs[cycle_key] = pairs
    pairs = no_repeating_pairs_select.cycle_pairs[cycle_key]
    return pairs[pair_idx][0], pairs[pair_idx][1]

def gibbs_sampler_binomial_single(comparisons, n_items, prior_precision, n_samples=1000, n_burnins=500, seed=None):
    X = build_design_matrix(comparisons, n_items)
    n_comparisons = X.shape[0]
    if n_comparisons == 0:
        return np.zeros((n_samples, n_items))
    pg = PyPolyaGamma(seed=seed)
    n_total = n_samples + n_burnins
    samples = np.zeros((n_total, n_items))
    curr_theta = np.zeros(n_items)
    prior_precision = prior_precision.toarray() if sp.issparse(prior_precision) else prior_precision
    for t in range(n_total):
        omega = np.array([pg.pgdraw(1, x.dot(curr_theta)) for x in X])
        omega_diag = sp.diags(omega)
        precision = X.T @ omega_diag @ X + prior_precision
        mean = np.linalg.solve(precision, X.T @ (np.ones(n_comparisons) - 0.5))
        L = np.linalg.cholesky(precision)
        curr_theta = mean + np.linalg.solve(L.T, np.random.normal(size=n_items))
        samples[t] = curr_theta
    return samples[n_burnins:]

def compute_expected_metrics(comparisons, current_rank, n_items, prior_precision, theta_mean, theta_cov, item_pairs, seed, theta_samples):
    expected_taus = {}
    expected_spearmans = {}
    X = build_design_matrix(comparisons, n_items)
    prior_precision = prior_precision.toarray() if sp.issparse(prior_precision) else prior_precision
    for i, j in item_pairs:
        new_comparisons_i_wins = comparisons + [[i, j, i]]
        X_i_wins = build_design_matrix(new_comparisons_i_wins, n_items)
        new_comparisons_j_wins = comparisons + [[i, j, j]]
        X_j_wins = build_design_matrix(new_comparisons_j_wins, n_items)
        samples_i_wins = gibbs_sampler_binomial_single(new_comparisons_i_wins, n_items, prior_precision, n_samples=100, n_burnins=50, seed=seed)
        samples_j_wins = gibbs_sampler_binomial_single(new_comparisons_j_wins, n_items, prior_precision, n_samples=100, n_burnins=50, seed=seed+1)
        tau_i_wins = np.mean([kendalltau(np.argsort(-sample), current_rank)[0] for sample in samples_i_wins])
        tau_j_wins = np.mean([kendalltau(np.argsort(-sample), current_rank)[0] for sample in samples_j_wins])
        spearman_i_wins = np.mean([spearmanr(np.argsort(-sample), current_rank)[0] for sample in samples_i_wins])
        spearman_j_wins = np.mean([spearmanr(np.argsort(-sample), current_rank)[0] for sample in samples_j_wins])
        prob_i_wins = np.mean([1 if sample[i] > sample[j] else 0 for sample in theta_samples])
        expected_taus[(i, j)] = prob_i_wins * tau_i_wins + (1 - prob_i_wins) * tau_j_wins
        expected_spearmans[(i, j)] = prob_i_wins * spearman_i_wins + (1 - prob_i_wins) * spearman_j_wins
    return expected_taus, expected_spearmans

def compute_pair_entropy(n_items, theta_samples):
    entropies = {}
    for i, j in get_item_pairs(n_items):
        prob_i_wins = np.mean([1 if sample[i] > sample[j] else 0 for sample in theta_samples])
        prob_j_wins = 1 - prob_i_wins
        if 0 < prob_i_wins < 1:
            h = - (prob_i_wins * np.log2(prob_i_wins) + prob_j_wins * np.log2(prob_j_wins))
        else:
            h = 0
        entropies[(i, j)] = h
    return entropies

def select_item_indices(wins, losses, n_items, expected_taus=None, expected_spearmans=None, method='pair_entropy', theta_samples=None, current_entropy=None, total_comparisons=None, current_comparison_idx=None, seed=None):
    if method == 'random':
        ind_1, ind_2 = np.random.choice(np.arange(0, n_items), size=2, replace=False)
        return ind_1, ind_2
    if method == 'KG-Tau':
        ind_1, ind_2 = max(expected_taus, key=expected_taus.get)
        return ind_1, ind_2
    if method == 'KG-Rho':
        ind_1, ind_2 = max(expected_spearmans, key=expected_spearmans.get)
        return ind_1, ind_2
    if method == 'pair_entropy':
        entropies = compute_pair_entropy(n_items, theta_samples)
        ind_1, ind_2 = max(entropies, key=entropies.get)
        return ind_1, ind_2
    if method == 'round_robin':
        return round_robin_select(n_items, total_comparisons)
    if method == 'no_repeating_pairs':
        return no_repeating_pairs_select(n_items, total_comparisons, current_comparison_idx, seed=seed)
    raise ValueError(f"Unknown selection method: {method}")
