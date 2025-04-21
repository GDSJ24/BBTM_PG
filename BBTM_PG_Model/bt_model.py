import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve
import multiprocessing as mp
from pypolyagamma import PyPolyaGamma

def gibbs_sampler(X, prior_precision, n_samples, n_burnins, seed=None):
    """
    Perform Gibbs sampling for Bayesian inference on pairwise comparison data.

    Args:
        X (scipy.sparse matrix): Design matrix for pairwise comparisons.
        prior_precision (scipy.sparse matrix): Prior precision matrix for the parameters.
        n_samples (int): Number of samples to draw after burn-in.
        n_burnins (int): Number of burn-in iterations.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        numpy.ndarray: Array of sampled parameter values (theta) after burn-in.
    """
    np.random.seed(seed)
    n_students = X.shape[1]
    n_comparisons = X.shape[0]
    theta = np.zeros(n_students)
    omega = np.ones(n_comparisons)
    pg = PyPolyaGamma()
    theta_samples = []

    for _ in range(n_samples + n_burnins):
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

        if _ >= n_burnins:
            theta_samples.append(theta.copy())

    return np.array(theta_samples)

def run_model(X, n_students, n_samples=3000, n_burnins=1000, n_chains=4):
    """
    Run the Gibbs sampler in parallel across multiple chains.

    Args:
        X (scipy.sparse matrix): Design matrix for pairwise comparisons.
        n_students (int): Number of students (parameters to estimate).
        n_samples (int, optional): Number of samples to draw after burn-in per chain. Default is 3000.
        n_burnins (int, optional): Number of burn-in iterations per chain. Default is 1000.
        n_chains (int, optional): Number of parallel chains to run. Default is 4.

    Returns:
        numpy.ndarray: Concatenated array of sampled parameter values (theta) from all chains.
    """
    prior_precision = sp.eye(n_students)
    seed_seq = np.random.SeedSequence(2013)
    chain_seeds = seed_seq.spawn(n_chains)

    with mp.Pool(n_chains) as pool:
        chains = pool.starmap(
            gibbs_sampler,
            [
                (X, prior_precision, n_samples, n_burnins, seed.generate_state(1)[0])
                for seed in chain_seeds
            ]
        )

    return np.concatenate(chains)

def summarize_results(samples, n_students):
    """
    Summarize the results of the Gibbs sampling by calculating the mean and rank of the parameters.

    Args:
        samples (numpy.ndarray): Array of sampled parameter values (theta).
        n_students (int): Number of students (parameters).

    Returns:
        pandas.DataFrame: DataFrame containing the student IDs, mean theta values, and ranks.
    """
    theta_mean = np.mean(samples, axis=0)
    rank_df = pd.DataFrame({
        'Student ID': np.arange(n_students),
        'Mean Theta': theta_mean,
        'Rank': theta_mean.argsort()[::-1].argsort() + 1
    })
    return rank_df.sort_values(by='Rank').reset_index(drop=True)