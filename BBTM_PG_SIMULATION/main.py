import numpy as np
import scipy.sparse as sp
import multiprocessing as mp
from comparisons import ComparisonGenerator, build_design_matrix, get_student_pairs
from gibbs_sampling import gibbs_sampler_binomial, gibbs_sampler_binomial_single
from metrics import compute_expected_metrics
from ranking import determine_winner
from simulation import print_metrics, print_simulation_results
import sys


# try: 
from pbcj import *
# except:
#     from .pbcj import *

from pypolyagamma import PyPolyaGamma

"""

GOAL: script for running 1 simulated experiment -- we aim to run many in parallel

simulation variations. 

- method: BBTM, PBCJ
- N: number of students/items
- K: number of comparisons/items
- selection method: random, entropy, KG, round robin
- seed: default 1234
    - for mu and sigma vectors, we will generate them randomly based on seed
    - it can work as an ID, and be potentially used for matched experiment/sampling/runs across methods

Things to save. (JSON files -- be careful of naming convention)

- mu/sigma vectors (not that important)
- expected target ranks (based on the mu ans sigma) - T
- target rank density (based on the mu and sigma) - R
- outcomes, where each row is formatted in (item index i, item index j, winner) in a 2-D array
- Estimated expected ranks - t
- Estimated rank densities - r
- history of expected tau ranks (T, t)
- history of expected spearman rho (T, t)
- history of density of JSD scores across all items (R, r) -- i th row is N JSD scores after i  comparisons. 


"""

def run_pbcj(N, K, selection_method, seed):
    # generate mus and sigmas
    # self, n_students: int, mu_vector: List[float], sigma_vector: List[float], seed: int = 2013):
    loc_mus, loc_sigmas = generate_mus_sigmas(N, seed)
    # set priors
    alpha_init = np.ones((N, N)) # prior
    beta_init = np.ones((N, N)) # prior
    # initialise matrix for wins
    wins = np.zeros((N, N))
    # initialise matrix for losses
    losses = np.zeros((N, N))
    # initialise rank density matrix
    rank_density = np.ones((N, N))/N # prior
    # initialise expected rank list
    exp_rank = pbcj_cal_exp_ranks(rank_density)
    # calculate current entropy matrix
    entropy = pbcj_calc_entropy(wins, losses, alpha_init, beta_init)
    # object for comparisons
    comp = ComparisonGenerator(N, loc_mus, loc_sigmas, seed=seed) 
    exp_hist = []
    for i in range(N*K):
        # pick item indices
        ind_1, ind_2 = pbcj_select_item_indices(wins, losses, alpha_init, beta_init, method=selection_method)
        print(ind_1, ind_2)
        # compare and get result -- update win/loss matrices
        win = comp.generate_winner_01(ind_1, ind_2)
        print(win)
        wins[ind_1, ind_2] += win
        wins[ind_2, ind_1] += (1-win)
        losses[ind_1, ind_2] += (1 - win)
        losses[ind_2, ind_1] += win
        # update entropy
        entropy = pbcj_calc_entropy(wins, losses, alpha_init, beta_init)
        # update exp rank
        exp_rank = pbcj_MC_exp_rank(wins, losses, alpha_init, beta_init)
        exp_hist.append(exp_rank)
    return np.array(exp_hist), wins, losses

def main_new():

    if '-seed' in sys.argv:
        seed = int(sys.argv[sys.argv.index('-seed')+1])
    else:
        seed = 12345

    if '-m' in sys.argv: # string
        method = sys.argv[sys.argv.index('-m')+1]
    else:
        method = "PBCJ"

    if '-n' in sys.argv: # int
        N = int(sys.argv[sys.argv.index('-n')+1])
    else:
        N = 5

    if '-k' in sys.argv:
        K = int(sys.argv[sys.argv.index('-k')+1])
    else:
        K = 5

    if '-sel' in sys.argv: # string: 
        sel = sys.argv[sys.argv.index('-sel')+1]
    else:
        sel = "random"
    
    print("Simulation settings:")
    print(seed, method, N, K, sel)


    # Run PBCJ
    if method=="PBCJ":
        res = run_pbcj(N, K, sel, seed)
        print(res[0])
        print(res[1])
        print(res[2])


def main_old():

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
    main_new()