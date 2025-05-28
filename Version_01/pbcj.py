# Bayesian comparative judgment
import numpy as np
from scipy.special import beta as beta_function
from scipy.special import digamma
from scipy.stats import beta, kendalltau, spearmanr
from itertools import combinations
from utils import generate_mus_sigmas
from bbtm import kendall_tau_distance, get_item_pairs

def pbcj_cal_exp_ranks(rank_density): 
    vals = np.arange(1, len(rank_density)+1, 1)
    return np.sum(rank_density*vals, axis=1)

def pbcj_MC_exp_rank(wins, losses, alpha_init, beta_init, n_mc=1000, seed=None):
    n_items = len(wins)
    rng = np.random.default_rng(seed)
    t_alpha = alpha_init + wins
    t_beta = beta_init + losses
    random_samples = np.round(beta.rvs(t_alpha, t_beta, size=(n_mc,n_items, n_items), random_state=rng), decimals=0)
    mask = np.repeat(np.array(np.identity(n_items), dtype=bool)[np.newaxis,:,:], n_mc, axis=0)
    masked_array = np.ma.array(random_samples, mask=mask)
    data = n_items - np.sum(masked_array, axis=2).data
    return np.average(data, axis=0)

def pbcj_MC_rank_density(wins, losses, alpha_init, beta_init, n_mc=1000, seed=None):
    n_items = len(wins)
    t_alpha = alpha_init + wins
    t_beta = beta_init + losses
    random_samples = np.round(beta.rvs(t_alpha, t_beta, size=(n_mc,n_items, n_items)), decimals=0)
    mask = np.repeat(np.array(np.identity(n_items), dtype=bool)[np.newaxis,:,:], n_mc, axis=0)
    masked_array = np.ma.array(random_samples, mask=mask)
    data = n_items - np.sum(masked_array, axis=2).data
    return data

def pbcj_calc_entropy(wins, losses, alpha_init, beta_init):
    t_alpha = alpha_init + wins
    t_beta = beta_init + losses
    ent = np.zeros((len(t_alpha), len(t_alpha)))
    ent = np.log(beta_function(t_alpha, t_beta)) - (t_alpha-1)*digamma(t_alpha) \
            - (t_beta-1) * digamma(t_beta) + (t_alpha+t_beta-2)*digamma(t_alpha+t_beta)
    return ent

def pbcj_rank_entropy(wins, losses, alpha_init, beta_init, n_items, n_samples=500, seed=None):
    rng = np.random.default_rng(seed)
    rank_dist = np.zeros((n_items, n_items))
    t_alpha = alpha_init + wins
    t_beta = beta_init + losses
    for _ in range(n_samples):
        samples = beta.rvs(t_alpha, t_beta, size=(n_items, n_items), random_state=rng)
        np.fill_diagonal(samples, 0)
        scores = n_items - np.sum(samples, axis=1)
        ranks = np.argsort(-scores)
        for rank, item in enumerate(ranks):
            rank_dist[item, rank] += 1
    rank_dist /= n_samples
    entropy = 0
    for i in range(n_items):
        for r in range(n_items):
            if rank_dist[i, r] > 0:
                entropy -= rank_dist[i, r] * np.log2(rank_dist[i, r])
    return entropy / n_items  # Average entropy

def round_robin_select(n_items, total_comparisons, seed=None):
    if not hasattr(round_robin_select, 'pairs'):
        rng = np.random.default_rng(seed)
        players = list(range(n_items))
        if n_items % 2:
            players.append(-1)  # Dummy player
        n = len(players)
        rounds = []
        discriminant = rng.integers(0, 2)
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
            rng.shuffle(cycle_pairs)
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
        
def compute_expected_metrics_pbcj(wins, losses, alpha_init, beta_init, current_rank, n_items, seed=None):
    expected_taus = {}
    expected_spearmans = {}
    expected_entropies = {}
    item_pairs = get_item_pairs(n_items)
    alpha_params = wins + alpha_init
    beta_params = losses + beta_init
    rng = np.random.default_rng(seed)

    for i, j in item_pairs:
        # Case where j wins
        wins_0 = wins.copy()
        losses_0 = losses.copy()
        wins_0[j, i] += 1
        losses_0[i, j] += 1
        exp_rank_0 = pbcj_MC_exp_rank(wins_0, losses_0, alpha_init, beta_init, seed=rng.integers(1e6))
        new_rank_0 = np.argsort(-exp_rank_0)
        kappa_0 = kendall_tau_distance(current_rank, new_rank_0)
        rho_0 = spearmanr(current_rank, new_rank_0)[0]
        # Entropy where j wins
        entropy_0 = pbcj_rank_entropy(wins_0, losses_0, alpha_init, beta_init, n_items, seed=rng.integers(1e6))

        # Case where i wins
        wins_1 = wins.copy()
        losses_1 = losses.copy()
        wins_1[i, j] += 1
        losses_1[j, i] += 1
        exp_rank_1 = pbcj_MC_exp_rank(wins_1, losses_1, alpha_init, beta_init, seed=rng.integers(1e6))
        new_rank_1 = np.argsort(-exp_rank_1)
        kappa_1 = kendall_tau_distance(current_rank, new_rank_1)
        rho_1 = spearmanr(current_rank, new_rank_1)[0]
        # Entropy where i wins
        entropy_1 = pbcj_rank_entropy(wins_1, losses_1, alpha_init, beta_init, n_items, seed=rng.integers(1e6))

        # Probability using Beta CDF
        pr_i_gt_j = beta.cdf(0.5, alpha_params[i, j], beta_params[i, j]) if (alpha_params[i, j] + beta_params[i, j]) > 0 else 0.5

        # Expected values
        expected_taus[(i, j)] = kappa_0 * (1 - pr_i_gt_j) + kappa_1 * pr_i_gt_j
        expected_spearmans[(i, j)] = rho_0 * (1 - pr_i_gt_j) + rho_1 * pr_i_gt_j
        expected_entropies[(i, j)] = entropy_0 * (1 - pr_i_gt_j) + entropy_1 * pr_i_gt_j
    
    return expected_taus, expected_spearmans, expected_entropies

def pbcj_select_item_indices(wins, losses, alpha_init, beta_init, n_items, method="random", total_comparisons = None, current_comparison_idx=None, expected_taus=None, expected_rhos=None, current_rank=None, seed=None):
    if method == "random":
        ind_1, ind_2 = np.random.choice(np.arange(0, len(wins)), size=2, replace=False)
        return ind_1, ind_2
    if method == "entropy":
        ent = pbcj_calc_entropy(wins, losses, alpha_init, beta_init)
        np.fill_diagonal(ent, -1e100)
        ind_1, ind_2 = np.unravel_index(np.argmax(ent), ent.shape)
        return ind_1, ind_2
    if method == "round_robin":
        return round_robin_select(n_items, total_comparisons, seed=seed)
    if method == "no_repeating_pairs":
        return no_repeating_pairs_select(n_items, total_comparisons, current_comparison_idx, seed=seed)
    if method in ["KG-Tau", "KG-Rho", "KG-Entropy"]:
        expected_taus, expected_spearmans, expected_entropies = compute_expected_metrics_pbcj(wins, losses, alpha_init, beta_init, current_rank, n_items, seed=seed)
        if method == "KG-Tau":
            ind_1, ind_2 = max(expected_taus, key=expected_taus.get)
            return ind_1, ind_2
        if method == "KG-Rho":
            ind_1, ind_2 = min(expected_spearmans, key=expected_spearmans.get)
            return ind_1, ind_2
        if method == "KG-Entropy":
            ind_1, ind_2 = min(expected_entropies, key=expected_entropies.get)
            return ind_1, ind_2
    raise ValueError(f"Unknown selection method: {method}")