# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import seaborn as sns
import scipy.sparse as sp
from pypolyagamma import PyPolyaGamma
import multiprocessing as mp
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import kendalltau, spearmanr, norm
from collections import defaultdict

# Original comparisons
comparisons = [[1, 2, 1], [1, 3, 1], [2, 3, 2], [0, 2, 2], [0, 3, 0], [0, 1, 0]]

n_students = len(np.unique(np.array(comparisons)))
print(f'Total students: {n_students}')

# Design matrix
def build_design_matrix(comparisons, n_students):
    n_comparisons = len(comparisons)
    X = sp.lil_matrix((n_comparisons, n_students))
    for idx, (i, j, k) in enumerate(comparisons):
        X[idx, k] = 1
        loser = i + j - k
        X[idx, loser] = -1
    return X.tocsr()

# Gibbs sampler (Bernoulli)
def gibbs_sampler(X, prior_precision, n_samples, n_burnins, seed=None):
    np.random.seed(seed)
    n_students = X.shape[1]
    n_comparisons = X.shape[0]
    theta = np.zeros(n_students)
    omega = np.ones(n_comparisons)
    pg = PyPolyaGamma()
    theta_samples = []
    for i in range(n_samples + n_burnins):
        for idx in range(n_comparisons):
            c = np.clip(X[idx].dot(theta), -15, 15).item()
            omega[idx] = pg.pgdraw(1, c)
        XtOmega = X.T.multiply(omega)
        XtOmegaX = XtOmega @ X + prior_precision
        XtOmegaX = XtOmegaX.toarray()
        L, lower = cho_factor(XtOmegaX)
        XtKappa = X.T @ np.full(n_comparisons, 0.5)
        mu = cho_solve((L, lower), XtKappa)
        z = np.random.randn(n_students)
        theta = mu + cho_solve((L, lower), z)
        theta -= theta.mean()
        if i >= n_burnins:
            theta_samples.append(theta.copy())
    return np.array(theta_samples)

# Gibbs sampler (Binomial, single chain)
def gibbs_sampler_binomial_single(comparisons, n_students, prior_precision, n_samples, n_burnins, seed=None):
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

# Multi-chain Binomial sampler
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

# Kendall’s Tau distance
def kendall_tau_distance(rank1, rank2):
    tau, _ = kendalltau(rank1, rank2)
    return (1 - tau) / 2

# Compute expected Kendall’s Tau distance
def compute_expected_tau(X, comparisons, current_rank, n_students, prior_precision, theta_mean, theta_cov, student_pairs, seed_seq):
    expected_taus = {}
    pair_seed_seq = seed_seq.spawn(len(student_pairs) * 2)
    seed_idx = 0
    
    for i, j in student_pairs:
        new_comparison_0 = [i, j, j]
        updated_comparisons_0 = comparisons + [new_comparison_0]
        new_rank_0 = gibbs_sampler_binomial(updated_comparisons_0, n_students, prior_precision, 3000, 1000, 4, pair_seed_seq[seed_idx])
        kappa_0 = kendall_tau_distance(current_rank, new_rank_0)
        seed_idx += 1
        
        new_comparison_1 = [i, j, i]
        updated_comparisons_1 = comparisons + [new_comparison_1]
        new_rank_1 = gibbs_sampler_binomial(updated_comparisons_1, n_students, prior_precision, 3000, 1000, 4, pair_seed_seq[seed_idx])
        kappa_1 = kendall_tau_distance(current_rank, new_rank_1)
        seed_idx += 1
        
        delta_mean = theta_mean[i] - theta_mean[j]
        delta_var = theta_cov[i, i] + theta_cov[j, j] - 2 * theta_cov[i, j]
        pr_i_gt_j = norm.cdf(delta_mean / np.sqrt(delta_var)) if delta_var > 0 else 0.5
        
        expected_tau = kappa_0 * (1 - pr_i_gt_j) + kappa_1 * pr_i_gt_j
        expected_taus[(i, j)] = expected_tau
    return expected_taus

# Compute expected Spearman’s correlation
def compute_expected_spearman(X, comparisons, current_rank, n_students, prior_precision, theta_mean, theta_cov, student_pairs, seed_seq):
    expected_spearmans = {}
    pair_seed_seq = seed_seq.spawn(len(student_pairs) * 2)
    seed_idx = 0
    
    for i, j in student_pairs:
        new_comparison_0 = [i, j, j]
        updated_comparisons_0 = comparisons + [new_comparison_0]
        new_rank_0 = gibbs_sampler_binomial(updated_comparisons_0, n_students, prior_precision, 3000, 1000, 4, pair_seed_seq[seed_idx])
        rho_0, _ = spearmanr(current_rank, new_rank_0)
        seed_idx += 1
        
        new_comparison_1 = [i, j, i]
        updated_comparisons_1 = comparisons + [new_comparison_1]
        new_rank_1 = gibbs_sampler_binomial(updated_comparisons_1, n_students, prior_precision, 3000, 1000, 4, pair_seed_seq[seed_idx])
        rho_1, _ = spearmanr(current_rank, new_rank_1)
        seed_idx += 1
        
        delta_mean = theta_mean[i] - theta_mean[j]
        delta_var = theta_cov[i, i] + theta_cov[j, j] - 2 * theta_cov[i, j]
        pr_i_gt_j = norm.cdf(delta_mean / np.sqrt(delta_var)) if delta_var > 0 else 0.5
        
        expected_spearman = rho_0 * (1 - pr_i_gt_j) + rho_1 * pr_i_gt_j
        expected_spearmans[(i, j)] = expected_spearman
    return expected_spearmans

# Parameters
n_samples = 3000
n_burnins = 1000
n_chains = 4

# Original ranking and posterior
X = build_design_matrix(comparisons, n_students)
prior_precision = sp.eye(n_students)
seed_seq = np.random.SeedSequence(2013)
chain_seeds = seed_seq.spawn(n_chains)
with mp.Pool(n_chains) as pool:
    chains = pool.starmap(gibbs_sampler, [
        (X, prior_precision, n_samples, n_burnins, seed.generate_state(1)[0])
        for seed in chain_seeds
    ])
comb_samples = np.concatenate(chains)
theta_mean = np.mean(comb_samples, axis=0)
theta_cov = np.cov(comb_samples.T)
current_rank = np.argsort(-theta_mean)

# Student pairs
student_pairs = [(i, j) for i in range(n_students) for j in range(i + 1, n_students)]

# Compute both metrics with isolated seeds
tau_seed_seq = seed_seq.spawn(1)[0]  # Separate seed for tau
rho_seed_seq = seed_seq.spawn(2)[1]  # Separate seed for rho
expected_taus = compute_expected_tau(X, comparisons, current_rank, n_students, prior_precision, theta_mean, theta_cov, student_pairs, tau_seed_seq)
expected_spearmans = compute_expected_spearman(X, comparisons, current_rank, n_students, prior_precision, theta_mean, theta_cov, student_pairs, rho_seed_seq)

# Results
print("\nExpected Kendall’s Tau Distances:")
for pair, tau in expected_taus.items():
    print(f"Pair {pair}: {tau:.4f}")
tau_best_pair = max(expected_taus, key=expected_taus.get)
print(f"Most disruptive pair (max E(tau_distance)): {tau_best_pair} with E(tau_distance) = {expected_taus[tau_best_pair]:.4f}")

print("\nExpected Spearman’s Rank Correlations:")
for pair, spearman in expected_spearmans.items():
    print(f"Pair {pair}: {spearman:.4f}")
rho_best_pair = min(expected_spearmans, key=expected_spearmans.get)
print(f"Most disruptive pair (min E(rho)): {rho_best_pair} with E(rho) = {expected_spearmans[rho_best_pair]:.4f}")

# Comparison
print("\nComparison:")
if tau_best_pair == rho_best_pair:
    print(f"Both metrics agree: {tau_best_pair} is the most disruptive pair.")
else:
    print(f"Difference detected:")
    print(f"  Kendall’s Tau picks {tau_best_pair} with E(tau_distance) = {expected_taus[tau_best_pair]:.4f}")
    print(f"  Spearman’s Rho picks {rho_best_pair} with E(rho) = {expected_spearmans[rho_best_pair]:.4f}")