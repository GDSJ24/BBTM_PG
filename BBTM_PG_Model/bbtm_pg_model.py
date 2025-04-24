import numpy as np
import scipy.sparse
from pypolyagamma import PyPolyaGamma
from scipy.linalg import cho_factor, cho_solve
import pandas as pd

def gibbs_sampler(X, prior_precision, n_samples, n_burnins, seed=None):
    """
    Perform Gibbs sampling for the model.
    
    Args:
        X (scipy.sparse.csr_matrix): Design matrix.
        prior_precision (scipy.sparse.csr_matrix): Prior precision matrix.
        n_samples (int): Number of samples to collect.
        n_burnins (int): Number of burn-in iterations.
        seed (int, optional): Random seed for reproducibility.
    
    Returns:
        np.ndarray: Array of theta samples.
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

def create_rank_df(comb_samples, n_students):
    """
    Create a DataFrame with student rankings based on mean theta.
    
    Args:
        comb_samples (np.ndarray): Combined theta samples from all chains.
        n_students (int): Number of students.
    
    Returns:
        pd.DataFrame: DataFrame with Student ID, Mean Theta, and Rank.
    """
    theta_mean = np.mean(comb_samples, axis=0)
    rank_df = pd.DataFrame({
        'Student ID': np.arange(n_students),
        'Mean Theta': theta_mean,
        'Rank': theta_mean.argsort()[::-1].argsort() + 1
    })
    rank_df = rank_df.sort_values(by='Rank').reset_index(drop=True)
    return rank_df