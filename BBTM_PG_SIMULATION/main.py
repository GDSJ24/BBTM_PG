import numpy as np
import scipy.sparse as sp
import multiprocessing as mp
from comparisons import ComparisonGenerator, build_design_matrix, get_student_pairs
from gibbs_sampling import gibbs_sampler_binomial, gibbs_sampler_binomial_single
from metrics import compute_expected_metrics
from ranking import determine_winner
from simulation import print_metrics, print_simulation_results

def main():
    # Parameters
    mu_vector = [50.0, 60.0, 70.0, 55.0, 65.0]  # 5 students
    sigma_vector = [5.0, 5.0, 5.0, 5.0, 5.0]
    n_samples = 3000
    n_burnins = 1000
    n_chains = 4
    n_students = len(mu_vector) 
    n_comparisons = 20  # Number of comparisons to generate

    # Generate comparisons
    generator = ComparisonGenerator(n_students=n_students, mu_vector=mu_vector, sigma_vector=sigma_vector, seed=2013)
    comparisons = generator.generate_comparisons(n_comparisons)
    true_mus, true_sigmas = generator.get_student_parameters()
    
    print(f'Total students: {n_students}')
    print(f'Generated {len(comparisons)} comparisons')

    print("\nPairwise comparison results in [i, j, winner]:")
    for i, j, winner in comparisons:
        print([i, j, winner])
    
    # Build design matrix
    X = build_design_matrix(comparisons, n_students)
    prior_precision = sp.eye(n_students)
    
    # Run Gibbs sampling
    seed_seq = np.random.SeedSequence(2013)
    
    with mp.Pool(n_chains) as pool:
        chains = pool.starmap(gibbs_sampler_binomial_single, [
            (comparisons, n_students, prior_precision, n_samples, n_burnins, seed.generate_state(1)[0])
            for seed in seed_seq.spawn(n_chains)
        ])
    
    comb_samples = np.concatenate(chains)
    theta_mean = np.mean(comb_samples, axis=0)
    theta_cov = np.cov(comb_samples.T)
    current_rank = np.argsort(-theta_mean)
    
    # Get student pairs
    student_pairs = get_student_pairs(n_students)
    
    # Compute metrics
    expected_taus, expected_spearmans = compute_expected_metrics(
        X, comparisons, current_rank, n_students, 
        prior_precision, theta_mean, theta_cov, 
        student_pairs, seed_seq
    )
    
    # Print metrics
    print_metrics(expected_taus, expected_spearmans)
    
    # Simulation results
    tau_best_pair = max(expected_taus, key=expected_taus.get)
    rho_best_pair = min(expected_spearmans, key=expected_spearmans.get)
    
    # Tau results
    tau_winner = determine_winner(tau_best_pair, theta_mean, theta_cov)
    tau_new_comparisons = comparisons + [[tau_best_pair[0], tau_best_pair[1], tau_winner]]
    tau_new_rank = gibbs_sampler_binomial(tau_new_comparisons, n_students, prior_precision, n_samples, n_burnins, n_chains, seed_seq)
    tau_theta_samples = np.concatenate([
        gibbs_sampler_binomial_single(tau_new_comparisons, n_students, prior_precision, n_samples, n_burnins, seed.generate_state(1)[0])
        for seed in seed_seq.spawn(n_chains)
    ])
    tau_theta_mean = np.mean(tau_theta_samples, axis=0)
    
    # Rho results
    rho_winner = determine_winner(rho_best_pair, theta_mean, theta_cov)
    rho_new_comparisons = comparisons + [[rho_best_pair[0], rho_best_pair[1], rho_winner]]
    rho_new_rank = gibbs_sampler_binomial(rho_new_comparisons, n_students, prior_precision, n_samples, n_burnins, n_chains, seed_seq)
    rho_theta_samples = np.concatenate([
        gibbs_sampler_binomial_single(rho_new_comparisons, n_students, prior_precision, n_samples, n_burnins, seed.generate_state(1)[0])
        for seed in seed_seq.spawn(n_chains)
    ])
    rho_theta_mean = np.mean(rho_theta_samples, axis=0)
    
    # Print simulation results
    print_simulation_results(
        current_rank, theta_mean, 
        tau_new_rank, tau_theta_mean,
        rho_new_rank, rho_theta_mean,
        n_students
    )

if __name__ == "__main__":
    main()