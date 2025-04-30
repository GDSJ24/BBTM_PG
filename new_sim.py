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

# Original comparisons
comparisons = [[5, 16, 5], [13, 21, 21], [5, 7, 5], [9, 11, 11], [3, 18, 18], [10, 28, 28], [14, 29, 14], [2, 7, 2], [9, 16, 16], [8, 9, 8], [1, 10, 10], [12, 14, 12], [8, 22, 8], [1, 7, 7], [6, 15, 15], [21, 22, 21], [5, 10, 5], [3, 7, 7], [6, 10, 10], [16, 20, 20], [17, 21, 21], [6, 19, 6], [17, 29, 29], [2, 11, 2], [3, 4, 4], [9, 21, 21], [3, 22, 3], [1, 8, 8], [2, 13, 2], [6, 9, 6], [14, 28, 28], [25, 29, 25], [17, 28, 28], [18, 26, 18], [15, 23, 15], [8, 14, 14], [13, 22, 22], [1, 5, 5], [6, 12, 12], [21, 25, 21], [24, 25, 25], [14, 26, 14], [5, 21, 21], [1, 25, 25], [8, 10, 10], [6, 8, 8], [14, 18, 14], [8, 15, 15], [5, 15, 15], [20, 28, 28], [3, 17, 17], [1, 29, 29], [8, 19, 8], [23, 24, 23], [22, 25, 25], [9, 24, 24], [2, 23, 2], [4, 14, 14], [12, 27, 27], [6, 27, 27], [13, 18, 18], [3, 13, 3], [4, 10, 10], [14, 27, 27], [0, 22, 22], [5, 27, 27], [25, 27, 27], [16, 19, 16], [11, 24, 11], [11, 12, 12], [7, 24, 7], [7, 9, 7], [16, 22, 16], [24, 28, 28], [3, 15, 15], [26, 27, 27], [14, 19, 14], [2, 14, 2], [9, 29, 29], [7, 28, 28], [7, 12, 12], [11, 13, 11], [9, 28, 28], [17, 20, 17], [13, 17, 17], [0, 23, 23], [8, 21, 21], [13, 25, 25], [4, 15, 15], [4, 5, 5], [21, 23, 21], [2, 8, 2], [0, 28, 28], [15, 22, 15], [9, 17, 17], [10, 23, 10], [0, 6, 6], [1, 21, 21], [2, 18, 2], [11, 28, 28], [0, 15, 15], [4, 12, 12], [3, 24, 24], [4, 21, 21], [2, 25, 2], [2, 27, 27], [15, 19, 15], [0, 12, 12], [5, 29, 5], [10, 11, 10], [18, 27, 27], [22, 24, 24], [2, 5, 5], [1, 13, 1], [19, 28, 28], [7, 18, 7], [13, 16, 16], [10, 20, 10], [1, 18, 1], [18, 19, 18], [12, 22, 12], [24, 26, 24], [1, 26, 1], [12, 15, 15], [17, 19, 17], [4, 16, 4], [6, 7, 7], [12, 23, 12], [6, 16, 6], [3, 27, 27], [0, 4, 4], [0, 10, 10], [11, 19, 11], [11, 25, 11], [19, 23, 23], [16, 24, 24], [0, 27, 27], [0, 20, 20], [18, 23, 23], [25, 26, 25], [11, 17, 11], [3, 23, 23], [4, 29, 29], [17, 26, 17], [16, 29, 29], [26, 29, 29], [20, 29, 29], [20, 26, 20],[20,12,12],[20,5,5]] #I added last 2

# Determine number of students based on maximum index
n_students = np.max(np.array(comparisons)) + 1  # Max index + 1 for 0-based indices
print(f'Total students: {n_students}')

# Design matrix with validation
def build_design_matrix(comparisons, n_students):
    n_comparisons = len(comparisons)
    X = sp.lil_matrix((n_comparisons, n_students))
    for idx, (i, j, k) in enumerate(comparisons):
        if i == j or k not in [i, j] or i >= n_students or j >= n_students or k >= n_students:
            raise ValueError(f"Invalid comparison at index {idx}: [{i}, {j}, {k}]")
        X[idx, k] = 1
        loser = i + j - k
        X[idx, loser] = -1
    return X.tocsr()

# Gibbs sampler - Original comparisons (Bernoulli - No repeated comparisons)
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

# Gibbs sampler - Original comp + New comparison (Binomial - handles repeated comparisons)
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

# Calculation of Kendalls tau distance
def kendall_tau_distance(rank1, rank2):
    tau, _ = kendalltau(rank1, rank2)
    return (1 - tau) / 2

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

# Parameters
n_samples = 3000
n_burnins = 1000
n_chains = 4

# Original ranking and posterior (using binomial for generality)
X = build_design_matrix(comparisons, n_students)
prior_precision = sp.eye(n_students)
seed_seq = np.random.SeedSequence(2013)
chain_seeds = seed_seq.spawn(n_chains)
with mp.Pool(n_chains) as pool:
    chains = pool.starmap(gibbs_sampler_binomial_single, [
        (comparisons, n_students, prior_precision, n_samples, n_burnins, seed.generate_state(1)[0])
        for seed in chain_seeds
    ])
comb_samples = np.concatenate(chains)
theta_mean = np.mean(comb_samples, axis=0)
theta_cov = np.cov(comb_samples.T)
current_rank = np.argsort(-theta_mean)

# Student pairs
student_pairs = [(i, j) for i in range(n_students) for j in range(i + 1, n_students)]

# Computation of both metrics
tau_seed_seq = seed_seq.spawn(1)[0]
expected_taus, expected_spearmans = compute_expected_metrics(X, comparisons, current_rank, n_students, prior_precision, theta_mean, theta_cov, student_pairs, tau_seed_seq)

# Final results for the metrics
print("\nExpected Kendall's Tau Distances:")
for pair, tau in expected_taus.items():
    print(f"Pair {pair}: {tau:.4f}")
tau_best_pair = max(expected_taus, key=expected_taus.get)
print(f"Most disruptive pair (max E(tau_distance)): {tau_best_pair} with E(tau_distance) = {expected_taus[tau_best_pair]:.4f}")

print("\nExpected Spearman's Rank Correlations:")
for pair, spearman in expected_spearmans.items():
    print(f"Pair {pair}: {spearman:.4f}")
rho_best_pair = min(expected_spearmans, key=expected_spearmans.get)
print(f"Most disruptive pair (min E(rho)): {rho_best_pair} with E(rho) = {expected_spearmans[rho_best_pair]:.4f}")

print("\nComparison:")
if tau_best_pair == rho_best_pair:
    print(f"Both metrics agree: {tau_best_pair} is the most disruptive pair.")
else:
    print(f"Difference detected:")
    print(f"  Kendall's Tau picks {tau_best_pair} with E(tau_distance) = {expected_taus[tau_best_pair]:.4f}")
    print(f"  Spearman's Rho picks {rho_best_pair} with E(rho) = {expected_spearmans[rho_best_pair]:.4f}")

# Simulation Output
print("\n=== Simulation Results ===")

# 1. Original Ranks - Calculated from original comparisons 
original_ranks = np.argsort(-theta_mean)
print("1. Original Ranks:", ",".join(map(str, original_ranks)))

# 2. Ability Scores based on Original Ranks
rank_map = {student_id: rank + 1 for rank, student_id in enumerate(original_ranks)}
ability_df = pd.DataFrame({
    "Student ID": range(n_students),
    "Mean Ability Score": theta_mean,
    "Rank": [rank_map[i] for i in range(n_students)]
})
print("\n2. Original Ability Scores:")
print(ability_df.to_string(index=False))

# 3. New Rank Order - Max Expected Kendalls Tau
tau_winner = tau_best_pair[0] if norm.cdf((theta_mean[tau_best_pair[0]] - theta_mean[tau_best_pair[1]]) / np.sqrt(theta_cov[tau_best_pair[0], tau_best_pair[0]] + theta_cov[tau_best_pair[1], tau_best_pair[1]] - 2 * theta_cov[tau_best_pair[0], tau_best_pair[1]])) > 0.5 else tau_best_pair[1]
tau_new_comparisons = comparisons + [[tau_best_pair[0], tau_best_pair[1], tau_winner]]
tau_new_rank = gibbs_sampler_binomial(tau_new_comparisons, n_students, prior_precision, n_samples, n_burnins, n_chains, seed_seq)
print("\n3. New Rank Order (Max Expected Kendall Tau):", ",".join(map(str, tau_new_rank)))

# 4. Ability Scores - Max Kendalls Tau
tau_chain_seeds = seed_seq.spawn(n_chains)
tau_theta_samples = np.concatenate([
    gibbs_sampler_binomial_single(tau_new_comparisons, n_students, prior_precision, n_samples, n_burnins, tau_chain_seeds[i].generate_state(1)[0])
    for i in range(n_chains)
])
tau_theta_mean = np.mean(tau_theta_samples, axis=0)
tau_rank_map = {student_id: rank + 1 for rank, student_id in enumerate(tau_new_rank)}
tau_ability_df = pd.DataFrame({
    "Student ID": range(n_students),
    "Mean Ability Score": tau_theta_mean,
    "Rank": [tau_rank_map[i] for i in range(n_students)]
})
print("\n4. Ability Scores (Max Expected Kendall Tau):")
print(tau_ability_df.to_string(index=False))

# 5. New Rank Order - Min Expected Spearmans Rho
rho_winner = rho_best_pair[0] if norm.cdf((theta_mean[rho_best_pair[0]] - theta_mean[rho_best_pair[1]]) / np.sqrt(theta_cov[rho_best_pair[0], rho_best_pair[0]] + theta_cov[rho_best_pair[1], rho_best_pair[1]] - 2 * theta_cov[rho_best_pair[0], rho_best_pair[1]])) > 0.5 else rho_best_pair[1]
rho_new_comparisons = comparisons + [[rho_best_pair[0], rho_best_pair[1], rho_winner]]
rho_new_rank = gibbs_sampler_binomial(rho_new_comparisons, n_students, prior_precision, n_samples, n_burnins, n_chains, seed_seq)
print("\n5. New Rank Order (Min Expected Spearman Rho):", ",".join(map(str, rho_new_rank)))

# 6. Ability Scores - Min Spearmans Rho
rho_chain_seeds = seed_seq.spawn(n_chains)
rho_theta_samples = np.concatenate([
    gibbs_sampler_binomial_single(rho_new_comparisons, n_students, prior_precision, n_samples, n_burnins, rho_chain_seeds[i].generate_state(1)[0])
    for i in range(n_chains)
])
rho_theta_mean = np.mean(rho_theta_samples, axis=0)
rho_rank_map = {student_id: rank + 1 for rank, student_id in enumerate(rho_new_rank)}
rho_ability_df = pd.DataFrame({
    "Student ID": range(n_students),
    "Mean Ability Score": rho_theta_mean,
    "Rank": [rho_rank_map[i] for i in range(n_students)]
})
print("\n6. Ability Scores (Min Expected Spearman Rho):")
print(rho_ability_df.to_string(index=False))