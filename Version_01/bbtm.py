import numpy as np
import scipy.sparse as sp
from itertools import combinations
from pypolyagamma import PyPolyaGamma
from utils import generate_mus_sigmas
from scipy.stats import kendalltau, spearmanr, entropy, norm

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
        np.random.seed(seed) # .default_rng/ .seed
        players = list(range(n_items))
        if n_items % 2:
            players.append(-1)  # Dummy player
        n = len(players)
        rounds = []
        discriminant = np.random.randint(0, 2)
        for round in range(n - 1):
            matchs = []
            for i in range(n // 2):
                if (round % 2) == discriminant:
                    pair = (players[i], players[n - 1 - i])
                else:
                    pair = (players[n - 1 - i], players[i])
                if pair[0] != -1 and pair[1] != -1:  # Skip dummy pairs
                    matchs.append(pair)
            players.insert(1, players.pop())
            rounds.append(matchs)
        # Flatten rounds into a list of pairs
        pairs = [pair for round in rounds for pair in round]
        # Calculate cycles needed for total_comparisons
        pairs_per_cycle = len(pairs)
        cycles = (total_comparisons + pairs_per_cycle - 1) // pairs_per_cycle
        round_robin_select.pairs = pairs * cycles  # Repeat pairs for multiple cycles
        round_robin_select.current_index = 0
        # Shuffle pairs within each cycle
        for i in range(cycles):
            cycle_pairs = round_robin_select.pairs[i * pairs_per_cycle:(i + 1) * pairs_per_cycle]
            np.random.shuffle(cycle_pairs)
            round_robin_select.pairs[i * pairs_per_cycle:(i + 1) * pairs_per_cycle] = cycle_pairs
    pair = round_robin_select.pairs[round_robin_select.current_index]
    round_robin_select.current_index += 1
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

def gibbs_sampler_binomial_single(comparisons, n_items, prior_precision, n_samples=1000, n_burnins=500, n_chains=4, seed=None):
    X = build_design_matrix(comparisons, n_items)
    n_comparisons = X.shape[0]
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

def kendall_tau_distance(rank1, rank2):
    n = len(rank1)
    pairs = list(combinations(range(n), 2))
    discordant = 0
    for i, j in pairs:
        if (rank1[i] < rank1[j] and rank2[i] > rank2[j]) or (rank1[i] > rank1[j] and rank2[i] < rank2[j]):
            discordant += 1
    return discordant / (n * (n - 1) / 2)

def compute_expected_metrics(X, comparisons, current_rank, n_items, prior_precision, theta_mean, theta_cov, item_pairs, seed=None):
    expected_taus = {}
    expected_spearmans = {}
    expected_entropies = {}
    tau_seed_seq = np.random.SeedSequence(seed)
    pair_seed_seq = tau_seed_seq.spawn(len(item_pairs) * 2)
    seed_idx = 0
    
    for i, j in item_pairs:
        # Case where j wins
        new_comparison_0 = [i, j, j]
        updated_comparisons_0 = comparisons + [new_comparison_0]
        samples_0 = gibbs_sampler_binomial_single(updated_comparisons_0, n_items, prior_precision, n_samples=1000, n_burnins=500, seed=pair_seed_seq[seed_idx].generate_state(1)[0])
        new_rank_0 = np.argsort(-np.mean(samples_0, axis=0))
        kappa_0 = kendall_tau_distance(current_rank, new_rank_0)
        rho_0 = spearmanr(current_rank, new_rank_0)[0]
        # Entropy for j wins
        rank_dist_j = np.zeros((n_items, n_items))
        for sample in samples_0:
            ranks = np.argsort(-sample)
            for rank, item in enumerate(ranks):
                rank_dist_j[item, rank] += 1
        rank_dist_j /= samples_0.shape[0]
        H_avg_j = np.mean([entropy(rank_dist_j[k], base=2) for k in range(n_items)])
        seed_idx += 1
        
        # Case where i wins
        new_comparison_1 = [i, j, i]
        updated_comparisons_1 = comparisons + [new_comparison_1]
        samples_1 = gibbs_sampler_binomial_single(updated_comparisons_1, n_items, prior_precision, n_samples=1000, n_burnins=500, seed=pair_seed_seq[seed_idx].generate_state(1)[0])
        new_rank_1 = np.argsort(-np.mean(samples_1, axis=0))
        kappa_1 = kendall_tau_distance(current_rank, new_rank_1)
        rho_1 = spearmanr(current_rank, new_rank_1)[0]
        # Entropy for i wins
        rank_dist_i = np.zeros((n_items, n_items))
        for sample in samples_1:
            ranks = np.argsort(-sample)
            for rank, item in enumerate(ranks):
                rank_dist_i[item, rank] += 1
        rank_dist_i /= samples_1.shape[0]
        H_avg_i = np.mean([entropy(rank_dist_i[k], base=2) for k in range(n_items)])
        seed_idx += 1
        
        # Probability calculation
        delta_mean = theta_mean[i] - theta_mean[j]
        delta_var = theta_cov[i, i] + theta_cov[j, j] - 2 * theta_cov[i, j]
        pr_i_gt_j = norm.cdf(delta_mean / np.sqrt(delta_var)) if delta_var > 0 else 0.5
        
        # Expected values
        expected_taus[(i, j)] = kappa_0 * (1 - pr_i_gt_j) + kappa_1 * pr_i_gt_j
        expected_spearmans[(i, j)] = rho_0 * (1 - pr_i_gt_j) + rho_1 * pr_i_gt_j
        expected_entropies[(i, j)] = H_avg_i * pr_i_gt_j + H_avg_j * (1 - pr_i_gt_j)
    
    return expected_taus, expected_spearmans, expected_entropies
    
def compute_pair_entropy(n_items, theta_mean, theta_cov):
    entropies = {}
    for i, j in get_item_pairs(n_items):
        delta_mean = theta_mean[i] - theta_mean[j]
        delta_var = theta_cov[i, i] + theta_cov[j, j] - 2 * theta_cov[i, j]
        prob_i_wins = norm.cdf(delta_mean / np.sqrt(delta_var)) if delta_var > 0 else 0.5
        prob_j_wins = 1 - prob_i_wins
        if 0 < prob_i_wins < 1:
            h = - (prob_i_wins * np.log2(prob_i_wins) + prob_j_wins * np.log2(prob_j_wins))
        else:
            h = 0
        entropies[(i, j)] = h
    return entropies

def select_item_indices(comparisons, n_items, prior_precision, theta_mean, theta_cov, method='random', total_comparisons=None, current_comparison_idx=None, seed=None, current_rank=None):

    if method == 'random':
        np.random.seed(seed) # .default_rng/.seed
        ind_1, ind_2 = np.random.choice(np.arange(0, n_items), size=2, replace=False)
        return ind_1, ind_2
    if method == 'round_robin':
        return round_robin_select(n_items, total_comparisons, seed=seed)
    if method == 'no_repeating_pairs':
        return no_repeating_pairs_select(n_items, total_comparisons, current_comparison_idx, seed=seed)
    if method == 'pair_entropy':
        entropies = compute_pair_entropy(n_items, theta_mean, theta_cov)
        ind_1, ind_2 = max(entropies, key=entropies.get)
        return ind_1, ind_2
    if method in ['KG-Tau', 'KG-Rho', 'KG-Entropy']:
        X = build_design_matrix(comparisons, n_items)
        item_pairs = get_item_pairs(n_items)
        tau_seed_seq = np.random.SeedSequence(seed)
        expected_taus, expected_spearmans, expected_entropies = compute_expected_metrics(
            X, comparisons, current_rank, n_items, prior_precision, theta_mean, theta_cov, item_pairs, seed=seed)
        if method == 'KG-Tau':
            ind_1, ind_2 = max(expected_taus, key=expected_taus.get)
            return ind_1, ind_2
        if method == 'KG-Rho':
            ind_1, ind_2 = min(expected_spearmans, key=expected_spearmans.get)
            return ind_1, ind_2
        if method == 'KG-Entropy':
            ind_1, ind_2 = min(expected_entropies, key=expected_entropies.get)
            return ind_1, ind_2
    raise ValueError(f"Unknown selection method: {method}")
