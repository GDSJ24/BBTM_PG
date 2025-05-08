# Bayesian comparative judgment
import numpy as np
from scipy.special import beta as beta_function
from scipy.special import digamma
from scipy.stats import beta, truncnorm
from itertools import combinations

def generate_mus_sigmas(N, seed, typ_mu=60, typ_sigma=5, mlb=0, mub=100, sub=None):
    np.random.seed(seed)
    lower, upper = (mlb - typ_mu) / typ_sigma, (mub - typ_mu) / typ_sigma
    trunc_normal = truncnorm(lower, upper, loc=typ_mu, scale=typ_sigma)
    mus = trunc_normal.rvs(N)
    if sub is None:
        sigmas = np.ones(N) * 5
    else:
        sigmas = np.random.random(size=N) * sub
    return mus, sigmas

def pbcj_cal_exp_ranks(rank_density): 
    vals = np.arange(1, len(rank_density)+1, 1)
    return np.sum(rank_density*vals, axis=1)

def pbcj_MC_exp_rank(wins, losses, alpha_init, beta_init, n_mc=1000):
    n_items = len(wins)
    t_alpha = alpha_init + wins
    t_beta = beta_init + losses
    random_samples = np.round(beta.rvs(t_alpha, t_beta, size=(n_mc,n_items, n_items)), decimals=0)
    mask = np.repeat(np.array(np.identity(n_items), dtype=bool)[np.newaxis,:,:], n_mc, axis=0)
    masked_array = np.ma.array(random_samples, mask=mask)
    data = n_items - np.sum(masked_array, axis=2).data
    return np.average(data, axis=0)

def pbcj_MC_rank_density(wins, losses, alpha_init, beta_init, n_mc=1000):
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

def round_robin_select(n_items, total_comparisons, seed=None):
    # Initialize round-robin state
    if not hasattr(round_robin_select, 'pairs'):
        rng = np.random.default_rng(seed)
        pairs = list(combinations(range(n_items), 2))
        rng.shuffle(pairs)  # Randomize pair order
        round_robin_select.pairs = pairs
        round_robin_select.pair_counter = {pair: 0 for pair in pairs}
        round_robin_select.current_index = 0
        # Calculate cycles (assumes total_comparisons is divisible by pairs_per_cycle)
        pairs_per_cycle = len(pairs)
        round_robin_select.cycles = total_comparisons // pairs_per_cycle

    # Select next pair if it hasn't been used 'cycles' times
    while round_robin_select.current_index < len(round_robin_select.pairs):
        pair = round_robin_select.pairs[round_robin_select.current_index]
        if round_robin_select.pair_counter[pair] < round_robin_select.cycles:
            round_robin_select.pair_counter[pair] += 1
            round_robin_select.current_index += 1
            if round_robin_select.current_index >= len(round_robin_select.pairs):
                round_robin_select.current_index = 0  # Reset for next cycle
            return pair[0], pair[1]
        
def no_repeating_pairs_select(n_students, total_comparisons, current_comparison_idx, seed):
    pairs_per_cycle = (n_students * (n_students - 1)) // 2
    cycle_idx = current_comparison_idx // pairs_per_cycle
    pair_idx = current_comparison_idx % pairs_per_cycle
    if not hasattr(no_repeating_pairs_select, 'cycle_pairs'):
        no_repeating_pairs_select.cycle_pairs = {}
    cycle_key = cycle_idx
    if cycle_key not in no_repeating_pairs_select.cycle_pairs:
        rng = np.random.default_rng(seed + cycle_idx)
        pairs = list(combinations(range(n_students), 2))
        rng.shuffle(pairs)
        no_repeating_pairs_select.cycle_pairs[cycle_key] = pairs
    pairs = no_repeating_pairs_select.cycle_pairs[cycle_key]
    return pairs[pair_idx][0], pairs[pair_idx][1]
        
def pbcj_select_item_indices(wins, losses, alpha_init, beta_init, n_items, method="random", total_comparisons = None, current_comparison_idx=None, seed=None):
    if method == "random":
        ind_1, ind_2 = np.random.choice(np.arange(0, len(wins)), size=2, replace=False)
        return ind_1, ind_2
    if method == "entropy":
        ent = pbcj_calc_entropy(wins, losses, alpha_init, beta_init)
        np.fill_diagonal(ent, -1e100)
        ind_1, ind_2 = np.unravel_index(np.argmax(ent), ent.shape)
        return ind_1, ind_2
    if method == "round_robin":
        return round_robin_select(wins.shape[0], total_comparisons)
    if method == "no_repeating_pairs":
        return no_repeating_pairs_select(n_items, total_comparisons, current_comparison_idx, seed=seed)
