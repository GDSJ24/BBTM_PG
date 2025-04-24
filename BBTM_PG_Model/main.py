import numpy as np
import scipy.sparse
import multiprocessing as mp
from bbtm_pg_data import load_and_preprocess_data, create_design_matrix
from bbtm_pg_model import gibbs_sampler, create_rank_df
from bbtm_pg_diagnostics import (compute_r_hat, compute_ess, plot_trace, plot_posterior, plot_forest,plot_posterior_distributions, plot_rank_probabilities)

def main():
    # Define input data and parameters
    comparisons = [[0, 1, 1], [0, 2, 2], [0, 3, 3], [0, 4, 0], [1, 2, 2],[1, 3, 3], [1, 4, 4], [2, 3, 2], [2, 4, 2],[3, 4, 3]]
    n_samples = 3000
    n_burnins = 1000
    n_chains = 4
    seed = 2013

    # Data preprocessing
    n_students, n_comparisons, comparisons_array = load_and_preprocess_data(comparisons)
    X = create_design_matrix(comparisons, n_students, n_comparisons)

    # Model setup
    prior_precision = scipy.sparse.eye(n_students)

    # Generate seeds for parallel chains
    seed_seq = np.random.SeedSequence(seed)
    chain_seeds = seed_seq.spawn(n_chains)

    # Run Gibbs sampler in parallel
    with mp.Pool(n_chains) as pool:
        chains = pool.starmap(gibbs_sampler, [
            (X, prior_precision, n_samples, n_burnins, seed.generate_state(1)[0])
            for seed in chain_seeds
        ])

    # Combine samples and create ranking DataFrame
    comb_samples = np.concatenate(chains)
    rank_df = create_rank_df(comb_samples, n_students)
    print("\nRanking DataFrame:")
    print(rank_df)

    # Convergence diagnostics
    rhat_df = compute_r_hat(chains)
    print("\nR-hat Diagnostics:")
    print(rhat_df)

    ess = compute_ess(chains)
    print(f"\nEffective Sample Size is {np.round(ess, 0)}")

    # Visualizations
    plot_trace(chains, n_students)
    plot_posterior(comb_samples, n_students)
    plot_forest(chains, n_students)
    plot_posterior_distributions(comb_samples, n_students)
    plot_rank_probabilities(comb_samples, n_students)

if __name__ == "__main__":
    main()