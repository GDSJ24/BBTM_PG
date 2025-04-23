import numpy as np
import scipy.sparse as sp
from pypolyagamma import PyPolyaGamma
import multiprocessing as mp
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm
from collections import defaultdict

def gibbs_sampler(X, prior_precision, n_samples, n_burnins, seed=None):
    """
    Perform Gibbs sampling to estimate student ability based on comparisons.

    Args:
        X (scipy.sparse matrix): Comparison matrix where rows represent comparisons and columns represent students.
        prior_precision (numpy.ndarray): Precision matrix for the prior distribution.
        n_samples (int): Number of samples to draw after burn-in.
        n_burnins (int): Number of burn-in iterations before sampling.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        numpy.ndarray: Array of sampled ability estimates for students.
    """
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

def gibbs_sampler_binomial_single(comparisons, n_students, prior_precision, n_samples, n_burnins, seed=None):
    """
    Perform Gibbs sampling for a single Markov chain using binomial comparisons.

    Args:
        comparisons (list of tuples): List of comparisons (i, j, winner), where i and j are students compared, 
                                      and winner indicates which student won.
        n_students (int): Total number of students being ranked.
        prior_precision (numpy.ndarray): Precision matrix for the prior distribution.
        n_samples (int): Number of samples to draw after burn-in.
        n_burnins (int): Number of burn-in iterations before sampling.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        numpy.ndarray: Array of sampled ability estimates for students.
    """
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

def gibbs_sampler_binomial(comparisons, n_students, prior_precision, n_samples, n_burnins, n_chains, seed_seq):
    """
    Perform Gibbs sampling using multiple Markov chains for ranking students based on comparisons.

    Args:
        comparisons (list of tuples): List of comparisons (i, j, winner), where i and j are students compared,
                                      and winner indicates which student won.
        n_students (int): Total number of students being ranked.
        prior_precision (numpy.ndarray): Precision matrix for the prior distribution.
        n_samples (int): Number of samples to draw after burn-in.
        n_burnins (int): Number of burn-in iterations before sampling.
        n_chains (int): Number of parallel chains for sampling.
        seed_seq (numpy.random.SeedSequence): Seed sequence for generating independent seeds per chain.

    Returns:
        numpy.ndarray: Array containing the ranking of students based on estimated ability.
    """
    chain_seeds = seed_seq.spawn(n_chains)
    with mp.Pool(n_chains) as pool:
        chains = pool.starmap(gibbs_sampler_binomial_single, [
            (comparisons, n_students, prior_precision, n_samples, n_burnins, seed.generate_state(1)[0])
            for seed in chain_seeds
        ])
    comb_samples = np.concatenate(chains)
    theta_mean = np.mean(comb_samples, axis=0)
    return np.argsort(-theta_mean)