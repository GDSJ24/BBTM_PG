import numpy as np
from scipy.stats import truncnorm, entropy, beta
from scipy.special import softmax
import pandas as pd

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

def js_divergence(p, q):
    """
    Jensen-Shannon Divergence between two distributions.
    """
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m, base=2) + entropy(q, m, base=2))
    