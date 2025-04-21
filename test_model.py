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

# Data input & preprocessing
comparisons = [[3, 12, 12], [0, 7, 0], [18, 8, 8], [13, 5, 13], [19, 14, 14], [17, 6, 17], [10, 15, 10], [1, 2, 2], [4, 16, 4], [11, 9, 11], [12, 11, 11], [3, 6, 6], [13, 18, 13], [1, 19, 19], [10, 8, 10], [7, 14, 14], [0, 15, 0], [2, 9, 2], [17, 4, 17], [16, 5, 5], [3, 11, 11], [1, 10, 10], [0, 12, 0], [7, 13, 13], [4, 15, 15], [19, 6, 19], [18, 14, 14], [5, 16, 5], [8, 17, 17], [9, 2, 2], [16, 7, 7], [5, 2, 2], [1, 15, 15], [12, 4, 4], [6, 3, 6], [19, 11, 11], [0, 13, 13], [9, 8, 8], [10, 17, 17], [14, 18, 14], [0, 8, 8], [4, 1, 1], [14, 5, 14], [7, 6, 7], [19, 18, 19], [13, 15, 13], [16, 11, 11], [9, 12, 12], [2, 3, 2], [10, 17, 17], [11, 15, 11], [18, 3, 3], [5, 12, 5], [0, 6, 0], [19, 17, 17], [1, 2, 2], [9, 13, 13], [7, 10, 10], [4, 16, 4], [8, 14, 14], [15, 3, 15], [5, 13, 13], [16, 14, 14], [10, 17, 17], [9, 8, 8], [4, 11, 11], [0, 12, 0], [18, 1, 1], [2, 7, 2], [19, 6, 19], [3, 16, 16], [14, 7, 14], [19, 17, 17], [11, 0, 11], [6, 4, 4], [2, 13, 2], [15, 12, 15], [5, 10, 10], [8, 1, 8], [18, 9, 18], [3, 12, 12], [19, 0, 19], [2, 5, 2], [1, 13, 13], [15, 7, 7], [14, 18, 14], [8, 6, 8], [4, 10, 10], [11, 16, 11], [17, 9, 17], [17, 11, 11], [4, 18, 4], [16, 6, 6], [10, 19, 10], [0, 14, 14], [13, 7, 13], [15, 8, 8], [5, 12, 5], [2, 9, 2], [1, 3, 1]]

n_students = len(np.unique(np.array(comparisons)))
n_comparisons = len(comparisons)
print(f'Total students are {n_students},\nTotal comparisons are {n_comparisons}')

# Design matrix
X = sp.lil_matrix((n_comparisons, n_students))
for idx, (i,j,k) in enumerate(comparisons):
    X[idx, k] = 1
    loser = i+j-k
    X[idx, loser] = -1

X = X.tocsr()
print(pd.DataFrame(X.toarray()))

# Model building
# Assume student's abilities come from a normal distribution (prior)
# Polya-Gamma draws for latent variables; PG(1,c) where c = exp(theta(i)-theta(j))
# Then logistic function will transform to Gaussian likelihood, make conjugacy and able to sample using Gibbs. 

prior_precision = sp.eye(n_students)
n_samples = 3000
n_burnins = 1000
n_chains = 4

def gibbs_sampler(X, prior_precision, n_samples, n_burnins, seed=None):
    np.random.seed(seed)
    n_students = X.shape[1]
    n_comparisons = X.shape[0]
    theta = np.zeros(n_students)
    omega = np.ones(n_comparisons)
    pg = PyPolyaGamma()
    theta_samples = []

    for _ in range (n_samples + n_burnins):
        for idx in range(n_comparisons):
            c = np.clip(X[idx].dot(theta), -15,15).item()
            omega[idx] = pg.pgdraw(1,c)

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

seed_seq = np.random.SeedSequence(2013)
chain_seeds = seed_seq.spawn(n_chains)

# Run parallel chains
with mp.Pool(n_chains) as pool:
    chains = pool.starmap(gibbs_sampler,[
        (X, prior_precision, n_samples, n_burnins, seed.generate_state(1)[0])
        for seed in chain_seeds
    ])

# Posterior distribtion & Ranking data frame (ranks & mean strength)
comb_samples = np.concatenate(chains)
theta_mean = np.mean(comb_samples, axis=0)
ranking = np.argsort(-theta_mean)

rank_df = pd.DataFrame({
    'Student ID' : np.arange(n_students),
    'Mean Theta' : theta_mean,
    'Rank' : theta_mean.argsort()[::-1].argsort()+1 
})

rank_df = rank_df.sort_values(by='Rank').reset_index(drop=True)
print(rank_df)