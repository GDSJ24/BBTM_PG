import numpy as np
from scipy.stats import truncnorm, entropy, beta, spearmanr
from scipy.special import softmax
import pandas as pd
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

def compute_target_rank_density(mus, sigmas, n_mc=5000, seed=None):
    """
    Compute true rank probabilities based on generative parameters (mus/sigmas).
    """
    rng = np.random.default_rng(seed)
    n_items = len(mus)
    rank_dist = np.zeros((n_items, n_items))
    
    for _ in range(n_mc):
        theta = rng.normal(mus, sigmas)
        ranks = np.argsort(-theta)  # Descending order (rank 0 = best)
        for rank, item in enumerate(ranks):
            rank_dist[item, rank] += 1
    
    rank_dist /= n_mc  # Normalize to probabilities
    return rank_dist

def compute_estimated_rank_density(theta_samples):
    """
    Compute rank probabilities from MCMC samples of theta.
    """
    n_items = theta_samples.shape[1]
    rank_dist = np.zeros((n_items, n_items))
    
    for sample in theta_samples:
        ranks = np.argsort(-sample)
        for rank, item in enumerate(ranks):
            rank_dist[item, rank] += 1
    
    rank_dist /= len(theta_samples)
    return rank_dist

def compute_pbcj_rank_density(wins, losses, alpha_init, beta_init, n_mc=5000, seed=None):
    """
    Compute rank probabilities for PBCJ using Monte Carlo sampling from Beta posteriors.
    
    Args:
        wins (np.ndarray): Wins matrix (shape: [N, N]).
        losses (np.ndarray): Losses matrix (shape: [N, N]).
        alpha_init (np.ndarray): Prior alpha (shape: [N, N]).
        beta_init (np.ndarray): Prior beta (shape: [N, N]).
        n_mc (int): Number of Monte Carlo samples.
        seed (int): Random seed.
    
    Returns:
        np.ndarray: Rank density matrix (shape: [N, N]).
    """
    rng = np.random.default_rng(seed)
    n_items = wins.shape[0]
    rank_dist = np.zeros((n_items, n_items))
    
    for _ in range(n_mc):
        # Sample win probabilities from Beta posteriors
        p_win = np.zeros((n_items, n_items))
        for i in range(n_items):
            for j in range(n_items):
                if i != j:
                    p_win[i, j] = rng.beta(
                        alpha_init[i, j] + wins[i, j],
                        beta_init[i, j] + losses[i, j]
                    )
        
        # Convert to rankings (sum probabilities to get "scores")
        scores = np.sum(p_win, axis=1)  # Total win probability for each item
        ranks = np.argsort(-scores)  # Descending order
        for rank, item in enumerate(ranks):
            rank_dist[item, rank] += 1
    
    rank_dist /= n_mc
    return rank_dist

def js_divergence(P, Q, base=2):
    """
    Jensen-Shannon Divergence between two distributions.
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    # Normalize distributions
    P /= P.sum()
    Q /= Q.sum()
    M = 0.5 * (P + Q)
    # Small epsilon to avoid log(0)
    eps = 1e-12
    kl_PM = np.sum(P * np.log((P + eps) / (M + eps))) / np.log(base) # KL divergence between P&M, summation of Pi*log(Pi/Mi). np.log(base=2) convert to bits o/w nats.
    kl_QM = np.sum(Q * np.log((Q + eps) / (M + eps))) / np.log(base)
    return 0.5 * (kl_PM + kl_QM)

def kendall_tau_distance(rank_a, rank_b):
    """
    Compute the normalized Kendall's tau distance (0 to 1) between two rankings.
    
    Args:
        rank_a, rank_b (array-like).
    
    Returns:
        float: Normalized distance where 0 = identical, 1 = reversed.
    """
    assert len(rank_a) == len(rank_b), "Rankings must be of equal length."
    n = len(rank_a)
    max_distance = n * (n - 1) / 2  # Maximum possible distance
    if max_distance == 0:  # Handle n=1 case
        return 0.0
        
    discordant_pairs = 0
    for i, j in combinations(range(n), 2):
        if (rank_a[i] > rank_a[j] and rank_b[i] < rank_b[j]) or \
           (rank_a[i] < rank_a[j] and rank_b[i] > rank_b[j]):
            discordant_pairs += 1

    return discordant_pairs / max_distance

def spearman_rho_distance(rank_a, rank_b):
    """
    Compute Spearman's Rho distance (-1 to 1) between two rankings.
    
    Args:
        rank_a, rank_b (array-like)
    
    Returns:
        float: Spearman's rho where -1 = reverse, 1 = identical.
    """
    rho, _ = spearmanr(rank_a, rank_b)  # Original Spearman's Rho [-1, 1]
    return rho