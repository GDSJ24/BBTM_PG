# Imports
import numpy as np
import scipy.sparse as sp
from pypolyagamma import PyPolyaGamma
import multiprocessing as mp
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import kendalltau, spearmanr, norm
from collections import defaultdict
import pandas as pd
from itertools import combinations
import random

# Gibbs sampler - Original comp + New comparison (Binomial - handles repeated comparisons)
def gibbs_sampler_binomial_single(comparisons, n_students, prior_precision, n_samples=3000, n_burnins=1000, seed=None):
    np.random.seed(seed)
    pair_counts = defaultdict(lambda: {'n': 0, 'wins': {}})
    for i, j, k in comparisons:
        pair = (min(i, j), max(i, j))
        winner = k
        pair_counts[pair]['n'] += 1
        pair_counts[pair]['wins'][winner] = pair_counts[pair]['wins'].get(winner, 0) + 1
    
    n_pairs = len(pair_counts)
    X = sp.lil_matrix((n_pairs, n_students))
    n_trials = []
    kappa = []
    for idx, (pair, data) in enumerate(pair_counts.items()):
        i, j = pair
        n = data['n']
        wins_i = data['wins'].get(i, 0)
        wins_j = data['wins'].get(j, 0)
        if wins_i > wins_j:
            X[idx, i] = 1
            X[idx, j] = -1
            kappa.append(wins_i - n/2)
        else:
            X[idx, j] = 1
            X[idx, i] = -1
            kappa.append(wins_j - n/2)
        n_trials.append(n)
    X = X.tocsr()

    theta = np.zeros(n_students)
    omega = np.ones(n_pairs)
    pg = PyPolyaGamma()
    theta_samples = []
    for _ in range(n_samples + n_burnins):
        for idx in range(n_pairs):
            c = np.clip(X[idx].dot(theta), -15, 15).item()
            omega[idx] = sum(pg.pgdraw(1, c) for _ in range(n_trials[idx]))
        XtOmega = X.T.multiply(omega)
        XtOmegaX = XtOmega @ X + prior_precision
        XtOmegaX = XtOmegaX.toarray()
        L, lower = cho_factor(XtOmegaX)
        XtKappa = X.T @ np.array(kappa)
        mu = cho_solve((L, lower), XtKappa)
        z = np.random.randn(n_students)
        theta = mu + cho_solve((L, lower), z)
        theta -= theta.mean()
        if _ >= n_burnins:
            theta_samples.append(theta.copy())
    return np.array(theta_samples)

# Multiple chains - Parallelise the chains
def gibbs_sampler_binomial(comparisons, n_students, prior_precision, n_samples, n_burnins, n_chains, seed_seq):
    chain_seeds = seed_seq.spawn(n_chains)
    with mp.Pool(n_chains) as pool:
        chains = pool.starmap(gibbs_sampler_binomial_single, [
            (comparisons, n_students, prior_precision, n_samples, n_burnins, seed.generate_state(1)[0])
            for seed in chain_seeds
        ])
    comb_samples = np.concatenate(chains)
    theta_mean = np.mean(comb_samples, axis=0)
    return np.argsort(-theta_mean)

# Computation of expected metrics - Kendall's tau and Spearman rho
def compute_expected_metrics(X, comparisons, current_rank, n_students, prior_precision, theta_mean, theta_cov, student_pairs, tau_seed_seq):
    expected_taus = {}
    expected_spearmans = {}
    pair_seed_seq = tau_seed_seq.spawn(len(student_pairs) * 2)
    seed_idx = 0
    
    for i, j in student_pairs:
        new_comparison_0 = [i, j, j]
        updated_comparisons_0 = comparisons + [new_comparison_0]
        new_rank_0 = gibbs_sampler_binomial(updated_comparisons_0, n_students, prior_precision, 3000, 1000, 4, pair_seed_seq[seed_idx])
        kappa_0 = kendall_tau_distance(current_rank, new_rank_0)
        rho_0, _ = spearmanr(current_rank, new_rank_0)
        seed_idx += 1
        
        new_comparison_1 = [i, j, i]
        updated_comparisons_1 = comparisons + [new_comparison_1]
        new_rank_1 = gibbs_sampler_binomial(updated_comparisons_1, n_students, prior_precision, 3000, 1000, 4, pair_seed_seq[seed_idx])
        kappa_1 = kendall_tau_distance(current_rank, new_rank_1)
        rho_1, _ = spearmanr(current_rank, new_rank_1)
        seed_idx += 1
        
        delta_mean = theta_mean[i] - theta_mean[j]
        delta_var = theta_cov[i, i] + theta_cov[j, j] - 2 * theta_cov[i, j]
        pr_i_gt_j = norm.cdf(delta_mean / np.sqrt(delta_var)) if delta_var > 0 else 0.5
        
        expected_taus[(i, j)] = kappa_0 * (1 - pr_i_gt_j) + kappa_1 * pr_i_gt_j
        expected_spearmans[(i, j)] = rho_0 * (1 - pr_i_gt_j) + rho_1 * pr_i_gt_j
    
    return expected_taus, expected_spearmans