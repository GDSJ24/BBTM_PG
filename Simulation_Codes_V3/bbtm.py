import numpy as np
import pandas as pd
import scipy.sparse as sp
from itertools import combinations
from pypolyagamma import PyPolyaGamma
from scipy.stats import entropy, norm, invgamma
import scipy.linalg
from utils import generate_mus_sigmas, kendall_tau_distance, spearman_rho_distance

def get_item_pairs(n_items):
    return list(combinations(range(n_items), 2))

def matches_to_matrices(matches, t):
    wins = np.zeros((t, t), dtype=int)
    games = np.zeros((t, t), dtype=int)
    for i, j, winner in matches:
        games[i, j] += 1
        games[j, i] += 1
        if winner == i:
            wins[i, j] += 1
        elif winner == j:
            wins[j, i] += 1
    return wins, games

def gibbs_sampler_binomial_single(comparisons, n_items, n_samples=200, 
                                n_burnins=100, n_chains=4, tau=2.0,
                                seed=None):
    """
    Hierarchical BBTM Gibbs sampler implementation
    Returns: 
        ndarray of samples (n_samples * n_chains, n_items)
    """
    # Initialize Polyagamma
    pg = PyPolyaGamma(seed=seed)
    rng = np.random.default_rng(seed)
    
    # Convert comparisons to match format
    wins, games = matches_to_matrices(comparisons, n_items)
    
    # Multi-chain sampling
    all_samples = []
    
    for chain in range(n_chains):
        chain_seed = seed + chain if seed is not None else None
        chain_rng = np.random.default_rng(chain_seed)
        
        # Initialize parameters
        lam = np.zeros(n_items)
        sigma2 = 1.0
        chain_samples = []
        
        for it in range(n_samples + n_burnins):
            # 1. Sample PG variables
            omega = np.zeros((n_items, n_items))
            kappa = np.zeros((n_items, n_items))
            for i in range(n_items):
                for j in range(i+1, n_items):
                    if games[i,j] > 0:
                        psi = lam[i] - lam[j]
                        omega[i,j] = omega[j,i] = pg.pgdraw(games[i,j], psi)
                        kappa[i,j] = wins[i,j] - games[i,j]/2
                        kappa[j,i] = -kappa[i,j]

            # 2. Sample lambda
            Q = np.zeros((n_items, n_items))
            b = np.zeros(n_items)
            for i in range(n_items):
                for j in range(n_items):
                    if i != j and games[i,j] > 0:
                        Q[i,i] += omega[i,j]
                        Q[i,j] -= omega[i,j]
                        b[i] += kappa[i,j]
            
            # posterior precision matrix A
            prior_prec = np.eye(n_items) / sigma2  # Hierarchical prior
            A = Q + prior_prec
            # compute Cholesky factor of A=LL^T / lower triangular matrix L
            L = scipy.linalg.cholesky(A, lower=True)
            # solve for the posterior mean
            # A*x = b  <=> LL^T x = b
            # solve Ly = b for y (forward substitution)
            y = scipy.linalg.solve_triangular(L, b, lower=True)
            # solve L^T x = y for posterior mean x (backward substitution)
            post_mean = scipy.linalg.solve_triangular(L.T, y, lower=False)
            # sample from the posterior N(post_mean, A^-1)
            # solve Lz = N(0,I) for z
            z = chain_rng.normal(size=n_items)
            deviation = scipy.linalg.solve_triangular(L.T, z, lower=False)
            # final sample
            lam = post_mean + deviation
            lam -= lam.mean()  # Center parameters

            # 3. Sample sigma2
            xi = invgamma.rvs(0.5, scale=1.0/tau**2, random_state=chain_rng)
            shape = (n_items/2) + 0.5
            scale = max(0.5*np.sum(lam**2) + 1.0/xi, 1e-9)
            sigma2 = invgamma.rvs(shape, scale=scale, random_state=chain_rng)

            # Store samples after burn-in
            if it >= n_burnins:
                chain_samples.append(lam.copy())
        
        all_samples.append(np.array(chain_samples))
    
    # Combine chains (n_chains * n_samples, n_items)
    return np.vstack(all_samples)

def build_design_matrix(comparisons, n_items):
    """Convert comparisons to sparse design matrix"""
    n_comp = len(comparisons)
    rows = []
    cols = []
    data = []
    
    for k, (i, j, winner) in enumerate(comparisons):
        rows.append(k)
        cols.append(winner)
        data.append(1.0)
        loser = j if winner == i else i
        rows.append(k)
        cols.append(loser)
        data.append(-1.0)
    
    return sp.csr_matrix((data, (rows, cols)), shape=(n_comp, n_items))

def compute_expected_metrics(X, comparisons, current_rank, n_items, prior_precision, theta_mean, theta_cov, item_pairs, seed=None):
    expected_taus = {}
    expected_spearmans = {}
    expected_entropies = {}
    tau_seed_seq = np.random.SeedSequence(seed)
    pair_seed_seq = tau_seed_seq.spawn(len(item_pairs) * 2)
    seed_idx = 0
    
    for i, j in item_pairs:
        # Case 1: j wins
        new_comparison_0 = [i, j, j]
        updated_comparisons_0 = comparisons + [new_comparison_0]
        samples_0 = gibbs_sampler_binomial_single(comparisons=updated_comparisons_0, n_items=n_items,n_samples=100, n_burnins=50, seed=pair_seed_seq[seed_idx].generate_state(1)[0])
        new_rank_0 = np.argsort(-np.mean(samples_0, axis=0))
        kappa_0 = kendall_tau_distance(current_rank, new_rank_0)
        rho_0 = spearman_rho_distance(current_rank, new_rank_0)
        rank_dist_j = np.zeros((n_items, n_items))
        for sample in samples_0:
            ranks = np.argsort(-sample)
            for rank, item in enumerate(ranks):
                rank_dist_j[item, rank] += 1
        rank_dist_j /= samples_0.shape[0]
        H_avg_j = np.mean([entropy(rank_dist_j[k], base=2) for k in range(n_items)])
        seed_idx += 1
        
        new_comparison_1 = [i, j, i]
        # Case 2: i wins
        updated_comparisons_1 = comparisons + [new_comparison_1]
        samples_1 = gibbs_sampler_binomial_single(comparisons=updated_comparisons_1, n_items=n_items, n_samples=100, n_burnins=50, seed=pair_seed_seq[seed_idx].generate_state(1)[0])
        new_rank_1 = np.argsort(-np.mean(samples_1, axis=0))
        kappa_1 = kendall_tau_distance(current_rank, new_rank_1)
        rho_1 = spearman_rho_distance(current_rank, new_rank_1)
        rank_dist_i = np.zeros((n_items, n_items))
        for sample in samples_1:
            ranks = np.argsort(-sample)
            for rank, item in enumerate(ranks):
                rank_dist_i[item, rank] += 1
        rank_dist_i /= samples_1.shape[0]
        H_avg_i = np.mean([entropy(rank_dist_i[k], base=2) for k in range(n_items)])
        seed_idx += 1
        
        delta_mean = theta_mean[i] - theta_mean[j]
        delta_var = theta_cov[i, i] + theta_cov[j, j] - 2 * theta_cov[i, j]
        pr_i_gt_j = norm.cdf(delta_mean / np.sqrt(delta_var)) if delta_var > 0 else 0.5
        
        expected_taus[(i, j)] = kappa_0 * (1 - pr_i_gt_j) + kappa_1 * pr_i_gt_j
        expected_spearmans[(i, j)] = rho_0 * (1 - pr_i_gt_j) + rho_1 * pr_i_gt_j
        expected_entropies[(i, j)] = H_avg_i * pr_i_gt_j + H_avg_j * (1 - pr_i_gt_j)
    
    return expected_taus, expected_spearmans, expected_entropies

def compute_pair_entropy(n_items, theta_mean, theta_cov):
    if theta_mean is None or theta_cov is None:
        return {(i,j): 1.0 for i, j in get_item_pairs(n_items)}
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
        # np.random.seed(seed); this will generate same/identical pairs as PBCJ
        ind_1, ind_2 = np.random.choice(np.arange(0, n_items), size=2, replace=False)
        return ind_1, ind_2
    if method == 'entropy':
        if theta_mean is None or theta_cov is None:
            raise ValueError("Theta mean and covariance must be provided for pair entropy selection.")
        entropies = compute_pair_entropy(n_items, theta_mean, theta_cov)
        ind_1, ind_2 = max(entropies, key=entropies.get)
        return ind_1, ind_2
    if method in ['KG-Tau', 'KG-Rho', 'KG-Entropy']:
        X = build_design_matrix(comparisons, n_items)
        item_pairs = get_item_pairs(n_items)
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