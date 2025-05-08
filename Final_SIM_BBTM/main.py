import sys
import numpy as np
import json
from json import JSONEncoder
from pbcj import generate_mus_sigmas, pbcj_cal_exp_ranks, pbcj_MC_exp_rank, pbcj_select_item_indices, round_robin_select, no_repeating_pairs_select
from bbtm import ComparisonGenerator, select_item_indices, build_design_matrix, gibbs_sampler_binomial_single, compute_expected_metrics, get_item_pairs, round_robin_select, compute_pair_entropy, no_repeating_pairs_select
import pandas as pd
import scipy.sparse as sp
from scipy.stats import entropy

def run_pbcj(N, K, selection_method, seed):
    total_comparisons = N * K
    loc_mus, loc_sigmas = generate_mus_sigmas(N, seed)
    alpha_init = np.ones((N, N))
    beta_init = np.ones((N, N))
    wins = np.zeros((N, N))
    losses = np.zeros((N, N))
    rank_density = np.ones((N, N)) / N
    exp_rank = pbcj_cal_exp_ranks(rank_density)
    comp = ComparisonGenerator(N, loc_mus, loc_sigmas, seed=seed)
    exp_hist = []

    for i in range(N*K):
        current_seed = seed + i
        ind_1, ind_2 = pbcj_select_item_indices(
            wins=wins,
            losses=losses,
            alpha_init=alpha_init,
            beta_init=beta_init,
            method=selection_method,
            total_comparisons=total_comparisons,
            current_comparison_idx=i,
            n_items = N,
            seed = current_seed
        )
        print(ind_1, ind_2)
        win = comp.generate_winner_01(ind_1, ind_2)
        print(win)
        wins[ind_1, ind_2] += win
        wins[ind_2, ind_1] += (1 - win)
        losses[ind_1, ind_2] += (1 - win)
        losses[ind_2, ind_1] += win
        exp_rank = pbcj_MC_exp_rank(wins, losses, alpha_init, beta_init)
        exp_hist.append(exp_rank)
    return np.array(exp_hist), wins, losses

def run_BBTM(N, K, selection_method, seed):
    total_comparisons = N * K
    loc_mus, loc_sigmas = generate_mus_sigmas(N, seed)
    wins = np.zeros((N, N))
    losses = np.zeros((N, N))
    comp = ComparisonGenerator(N, loc_mus, loc_sigmas, seed=seed)
    exp_hist = []
    comparisons = []
    prior_precision = sp.eye(N)
    rank_dist = np.ones((N, N)) / N

    for i in range(N*K):
        expected_taus = {}
        expected_spearmans = {}
        theta_samples = None
        current_entropy = None
        if selection_method in ["KG-Tau", "KG-Rho", "pair_entropy", "round_robin"]:
            X = build_design_matrix(comparisons=comparisons, n_items=N) if comparisons else sp.csr_matrix((0, N))
            theta_samples = gibbs_sampler_binomial_single(
                comparisons=comparisons,
                n_items=N,
                prior_precision=prior_precision,
                n_samples=3000,
                n_burnins=1000,
                seed=seed + i
            ) if comparisons else np.zeros((3000, N))
            theta_mean = np.mean(theta_samples, axis=0)
            theta_cov = np.cov(theta_samples.T)
            current_rank = np.argsort(-theta_mean)
            item_pairs = get_item_pairs(n_items=N)
            if selection_method in ["KG-Tau", "KG-Rho"]:
                expected_taus, expected_spearmans = compute_expected_metrics(
                    comparisons=comparisons,
                    current_rank=current_rank,
                    n_items=N,
                    prior_precision=prior_precision,
                    theta_mean=theta_mean,
                    theta_cov=theta_cov,
                    item_pairs=item_pairs,
                    seed=seed + i,
                    theta_samples=theta_samples
                )
            if selection_method == "pair_entropy":
                rank_dist_temp = np.zeros((N, N))
                for sample in theta_samples:
                    ranks = np.argsort(-sample)
                    for rank, item in enumerate(ranks):
                        rank_dist_temp[item, rank] += 1
                rank_dist_temp = rank_dist_temp / theta_samples.shape[0]
                current_entropy = np.mean([entropy(rank_dist_temp[k]) for k in range(N)])
                
        current_seed = seed + i
        ind_1, ind_2 = select_item_indices(
            wins=wins,
            losses=losses,
            n_items=N,
            expected_taus=expected_taus,
            expected_spearmans=expected_spearmans,
            method=selection_method,
            theta_samples=theta_samples,
            current_entropy=current_entropy,
            total_comparisons=total_comparisons,
            current_comparison_idx=i,
            seed=current_seed
        )
        print(ind_1, ind_2)
        win = comp.generate_winner_01(ind_1, ind_2)
        print(win)
        wins[ind_1, ind_2] += win
        wins[ind_2, ind_1] += (1 - win)
        losses[ind_1, ind_2] += (1 - win)
        losses[ind_2, ind_1] += win
        winner = ind_1 if win == 1 else ind_2
        comparisons.append([ind_1, ind_2, winner])
        X = build_design_matrix(comparisons=comparisons, n_items=N)
        theta_samples = gibbs_sampler_binomial_single(
            comparisons=comparisons,
            n_items=N,
            prior_precision=prior_precision,
            n_samples=1000,
            n_burnins=500,
            seed=seed + i
        )
        n_samples = theta_samples.shape[0]
        rank_dist = np.zeros((N, N))
        for sample in theta_samples:
            ranks = np.argsort(-sample)
            for rank, item in enumerate(ranks):
                rank_dist[item, rank] += 1
        rank_dist = np.round(rank_dist / n_samples, decimals=3)
        exp_rank = np.round(np.sum(rank_dist * np.arange(1, N+1), axis=1), decimals=3)
        exp_hist.append(exp_rank)
    return np.array(exp_hist), wins, losses

def main_new():
    if '-seed' in sys.argv:
        seed = int(sys.argv[sys.argv.index('-seed')+1])
    else:
        seed = 12345

    if '-m' in sys.argv:
        method = sys.argv[sys.argv.index('-m')+1]
    else:
        method = "PBCJ"

    if '-n' in sys.argv:
        N = int(sys.argv[sys.argv.index('-n')+1])
    else:
        N = 5

    if '-k' in sys.argv:
        K = int(sys.argv[sys.argv.index('-k')+1])
    else:
        K = 5

    if '-sel' in sys.argv:
        sel = sys.argv[sys.argv.index('-sel')+1]
    else:
        sel = "random"

    if method not in ["PBCJ", "BBTM"]:
        raise ValueError("Method must be 'PBCJ' or 'BBTM'")
    if method == "PBCJ" and sel not in ["random", "entropy", "round_robin","no_repeating_pairs"]:
        raise ValueError("Selection method for PBCJ must be 'random' or 'entropy','round_robin'")
    if method == "BBTM" and sel not in ["random", "KG-Tau", "KG-Rho", "pair_entropy", "round_robin", "no_repeating_pairs"]:
        raise ValueError("Selection method for BBTM must be 'random', 'KG-Tau', 'KG-Rho', 'pair_entropy', or 'round_robin'")

    print("Simulation settings:")
    print(f"Seed: {seed}, Method: {method}, N: {N}, K: {K}, Selection: {sel}")

    if method == "PBCJ":
        exp_hist, wins, losses = run_pbcj(N, K, sel, seed)
    elif method == "BBTM":
        exp_hist, wins, losses = run_BBTM(N, K, sel, seed)
    else:
        raise ValueError("Invalid method")

    print("\nExpected Rank History:")
    print(exp_hist)
    print("\nWins Matrix:")
    print(wins)
    print("\nLosses Matrix:")
    print(losses)

    print("\n=== Final Ranking Results ===")
    mus, sigmas = generate_mus_sigmas(N, seed)
    comp_gen = ComparisonGenerator(n_items=N, mu_vector=mus, sigma_vector=sigmas, seed=seed)
    comparisons = []
    for i in range(N):
        for j in range(i+1, N):
            if wins[i,j] > 0 or losses[i,j] > 0:
                for _ in range(int(wins[i,j])):
                    comparisons.append([i, j, i])
                for _ in range(int(losses[i,j])):
                    comparisons.append([i, j, j])

    if comparisons:
        prior_precision = sp.eye(N)
        X = build_design_matrix(comparisons=comparisons, n_items=N)
        theta_samples = gibbs_sampler_binomial_single(
            comparisons=comparisons,
            n_items=N,
            prior_precision=prior_precision,
            n_samples=3000,
            n_burnins=1000,
            seed=seed
        )
        theta_mean = np.mean(theta_samples, axis=0)
        final_rank = np.argsort(-theta_mean)

        n_samples = theta_samples.shape[0]
        final_rank_dist = np.zeros((N, N))
        for sample in theta_samples:
            ranks = np.argsort(-sample)
            for rank, item in enumerate(ranks):
                final_rank_dist[item, rank] += 1
        final_rank_dist = np.round(final_rank_dist / n_samples, decimals=3)
        final_exp_rank = np.round(np.sum(final_rank_dist * np.arange(1, N+1), axis=1), decimals=3)

        print("1. Final Ranks:", ",".join(map(str, final_rank)))

        rank_map = {item_id: rank + 1 for rank, item_id in enumerate(final_rank)}
        ability_df = pd.DataFrame({
            "item ID": range(N),
            "Mean Ability Score": theta_mean,
            "Rank": [rank_map[i] for i in range(N)],
            "Expected Rank": final_exp_rank.tolist()
        })
        print("\n2. Final Ability Scores and Expected Rank:")
        print(ability_df.to_string(index=False))
    else:
        print("No comparisons generated, cannot compute final ranks.")

if __name__ == "__main__":
    main_new()
