import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import entropy, rankdata
from pbcj import pbcj_cal_exp_ranks, pbcj_MC_exp_rank, pbcj_select_item_indices, compute_expected_metrics_pbcj
from bbtm import ComparisonGenerator, select_item_indices, build_design_matrix, gibbs_sampler_binomial_single, compute_expected_metrics, get_item_pairs
from utils import generate_mus_sigmas, compute_target_rank_density, compute_estimated_rank_density, js_divergence, compute_pbcj_rank_density, kendall_tau_distance, spearman_rho_distance
import json
import os
from datetime import datetime

def numpy_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: numpy_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [numpy_to_serializable(x) for x in obj]
    return obj

def parse_args():
    args = {'-seed': 12345, '-m': "PBCJ", '-n': 5, '-k': 5, '-sel': "random"}
    for key in args:
        if key in sys.argv:
            idx = sys.argv.index(key) + 1
            args[key] = int(sys.argv[idx]) if key in ['-seed', '-n', '-k'] else sys.argv[idx]
    return args['-seed'], args['-m'], args['-n'], args['-k'], args['-sel']

def validate_params(method, sel):
    valid_pbcj = ["random", "entropy", "round_robin", "no_repeating_pairs", "KG-Tau", "KG-Rho", "KG-Entropy"]
    valid_bbtm = ["random", "KG-Tau", "KG-Rho", "pair_entropy", "round_robin", "no_repeating_pairs", "KG-Entropy"]
    if method not in ["PBCJ", "BBTM"]:
        raise ValueError("Method must be 'PBCJ' or 'BBTM'")
    if method == "PBCJ" and sel not in valid_pbcj:
        raise ValueError(f"Selection method for PBCJ must be one of {valid_pbcj}")
    if method == "BBTM" and sel not in valid_bbtm:
        raise ValueError(f"Selection method for BBTM must be one of {valid_bbtm}")

def init_history(method, sel, N, K, seed, mus, sigmas, R, T):
    return {
        "metadata": {"timestamp": datetime.now().isoformat(), "command": " ".join(sys.argv), "version": "1.0"},
        "parameters": {"method": method, "selection": sel, "N": N, "K": K, "seed": seed},
        "true_values": {"mus": mus.tolist(), "sigmas": sigmas.tolist(), 
                       "target_rank_density": R.tolist(), "expected_target_ranks": T},
        "estimated_values": {"rank_density": None, "expected_ranks": None, 
                           "final_ability_scores": None, "final_ranks": None},
        "comparison_history": {"outcomes": [], "tau_history": [], "spearman_history": [], "jsd_history": [], "expected_rank_history": []},
        "knowledge_gradient_metrics": {"expected_taus": [], "expected_spearmans": [], "expected_entropies": []},
        "metrics": {"final_jsd_scores": None}
    }

def generate_outcomes(wins, losses, N):
    outcomes = []
    for i in range(N):
        for j in range(i+1, N):
            for _ in range(int(wins[i,j])):
                outcomes.append([int(i), int(j), int(i)])
            for _ in range(int(losses[i,j])):
                outcomes.append([int(i), int(j), int(j)])
    return outcomes

def run_pbcj(N, K, selection_method, seed, history):
    mus, sigmas = generate_mus_sigmas(N, seed)
    alpha_init = beta_init = np.ones((N, N))
    wins = np.zeros((N, N))
    losses = np.zeros((N, N))
    comp = ComparisonGenerator(N, mus, sigmas, seed=seed)
    exp_hist = []
    is_kg_method = selection_method in ["KG-Tau", "KG-Rho", "KG-Entropy"]

    for i in range(N*K):
        current_seed = seed + i
        current_rank = np.argsort(-pbcj_MC_exp_rank(wins, losses, alpha_init, beta_init, seed=current_seed))
        if is_kg_method:
            expected_taus, expected_spearmans, expected_entropies = compute_expected_metrics_pbcj(
                wins, losses, alpha_init, beta_init, current_rank, N, seed=current_seed
            )
        ind_1, ind_2 = pbcj_select_item_indices(
            wins=wins, losses=losses, alpha_init=alpha_init, beta_init=beta_init,
            method=selection_method, total_comparisons=N*K, current_comparison_idx=i,
            current_rank=current_rank, n_items=N, seed=current_seed
        )
        if is_kg_method:
            pair = (ind_1, ind_2)
            history["knowledge_gradient_metrics"]["expected_taus"].append(
                expected_taus.get(pair, None)
            )
            history["knowledge_gradient_metrics"]["expected_spearmans"].append(
                expected_spearmans.get(pair, None)
            )
            history["knowledge_gradient_metrics"]["expected_entropies"].append(
                expected_entropies.get(pair, None)
            )
        print(ind_1, ind_2)
        win = comp.generate_winner_01(ind_1, ind_2)
        print(win)
        wins[ind_1, ind_2] += win
        wins[ind_2, ind_1] += (1 - win)
        losses[ind_1, ind_2] += (1 - win)
        losses[ind_2, ind_1] += win
        exp_rank = pbcj_MC_exp_rank(wins, losses, alpha_init, beta_init, seed=current_seed)
        exp_hist.append(exp_rank)

    return np.array(exp_hist), wins, losses

def run_bbtm(N, K, selection_method, seed, history):
    mus, sigmas = generate_mus_sigmas(N, seed)
    wins = np.zeros((N, N))
    losses = np.zeros((N, N))
    comp = ComparisonGenerator(N, mus, sigmas, seed=seed)
    exp_hist, comparisons = [], []
    rank_densities = []
    prior_precision = sp.eye(N)
    is_kg_method = selection_method in ["KG-Tau", "KG-Rho", "KG-Entropy"]

    for i in range(N*K):
        current_seed = seed + i
        expected_taus = expected_spearmans = expected_entropies = {}
        theta_mean = np.zeros(N)
        theta_cov = np.zeros((N, N)) 
        current_rank = np.argsort(-theta_mean)
        if comparisons:
            X = build_design_matrix(comparisons, n_items=N)
            theta_samples = gibbs_sampler_binomial_single(
                comparisons=comparisons, n_items=N, n_samples=100, n_burnins=50, n_chains=4, seed=current_seed
            )
            if theta_samples is not None:
                theta_mean = np.mean(theta_samples, axis=0)
                theta_cov = np.cov(theta_samples.T)
                current_rank = np.argsort(-theta_mean)
                
        if is_kg_method or selection_method == "pair_entropy" or i > 0:
            X = build_design_matrix(comparisons, n_items=N) if comparisons else sp.csr_matrix((0, N))
            theta_samples = gibbs_sampler_binomial_single(
                comparisons=comparisons, n_items=N, n_samples=100, n_burnins=50, n_chains=4, seed=current_seed
            ) if comparisons else np.zeros((1000, N))
            
            if is_kg_method:
                theta_mean = np.mean(theta_samples, axis=0)
                theta_cov = np.cov(theta_samples.T)
                current_rank = np.argsort(-theta_mean)
                item_pairs = get_item_pairs(n_items=N)
                expected_taus, expected_spearmans, expected_entropies = compute_expected_metrics(
                    X=X, comparisons=comparisons, current_rank=current_rank, n_items=N,
                    prior_precision=prior_precision, theta_mean=theta_mean, theta_cov=theta_cov,
                    item_pairs=item_pairs, seed=current_seed
                )

        ind_1, ind_2 = select_item_indices(
            comparisons=comparisons, n_items=N, prior_precision=prior_precision,
            theta_mean=theta_mean, theta_cov=theta_cov, method=selection_method,
            total_comparisons=N*K, current_comparison_idx=i, seed=current_seed,
            current_rank=current_rank
        )
        
        if is_kg_method:
            pair = (ind_1, ind_2)
            history["knowledge_gradient_metrics"]["expected_taus"].append(
                expected_taus.get(pair, None)
            )
            history["knowledge_gradient_metrics"]["expected_spearmans"].append(
                expected_spearmans.get(pair, None)
            )
            history["knowledge_gradient_metrics"]["expected_entropies"].append(
                expected_entropies.get(pair, None)
            )
        
        print(ind_1, ind_2)
        win = comp.generate_winner_01(ind_1, ind_2)
        print(win)
        wins[ind_1, ind_2] += win
        wins[ind_2, ind_1] += (1 - win)
        losses[ind_1, ind_2] += (1 - win)  
        losses[ind_2, ind_1] += win
        comparisons.append([ind_1, ind_2, ind_1 if win == 1 else ind_2])
        
        theta_samples = gibbs_sampler_binomial_single(
            comparisons=comparisons, n_items=N, n_samples=3000, n_burnins=1000, n_chains=4, seed=current_seed
        )
        current_rank_density = compute_estimated_rank_density(theta_samples)
        rank_densities.append(current_rank_density)
        exp_rank = np.sum(current_rank_density * np.arange(1, N+1), axis=1)
        exp_hist.append(exp_rank)

    return np.array(exp_hist), wins, losses, theta_samples, rank_densities

def compute_history_metrics(exp_hist, T, R, method, wins, losses, alpha_init, beta_init, rank_densities, N, seed):
    tau_hist, spearman_hist, jsd_hist = [], [], []

    target_ranks = np.array(T)

    for i, current_t in enumerate(exp_hist):
        # Convert current_t (expected ranks) to rank POSITIONS (like T) Match T's format
        current_rank_positions = rankdata(current_t, method='min')
        
        # Now compute distances with consistent formats
        tau_distance = kendall_tau_distance(target_ranks, current_rank_positions)
        rho_distance = spearman_rho_distance(target_ranks, current_rank_positions)
        
        tau_hist.append(float(tau_distance))
        spearman_hist.append(float(rho_distance))
        
        # JSD calculation (unchanged)
        if method == "PBCJ":
            current_r = compute_pbcj_rank_density(wins, losses, alpha_init, beta_init, n_mc=1000, seed=seed+i)
        else:
            current_r = rank_densities[i] if i < len(rank_densities) else np.ones((N, N))/N
        jsd_scores = [js_divergence(R[k], current_r[k], base=2) for k in range(N)]
        jsd_hist.append([float(x) for x in jsd_scores])
    return tau_hist, spearman_hist, jsd_hist

def compute_final_results(wins, losses, N, seed):
    comparisons = []
    for i in range(N):
        for j in range(i+1, N):
            for _ in range(int(wins[i,j])):
                comparisons.append([i, j, i])
            for _ in range(int(losses[i,j])):
                comparisons.append([i, j, j])
    
    if not comparisons:
        return None, None, None, None
        
    prior_precision = sp.eye(N)
    theta_samples = gibbs_sampler_binomial_single(
        comparisons=comparisons, n_items=N, n_samples=1000, n_burnins=500, n_chains=4, seed=seed
    )
    theta_mean = np.mean(theta_samples, axis=0)
    
    # Fix: Convert final_rank to rank POSITIONS (like T)
    final_rank_item_order = np.argsort(-theta_mean)
    final_rank_positions = np.zeros(N, dtype=int)
    for rank_pos, item_id in enumerate(final_rank_item_order):
        final_rank_positions[item_id] = rank_pos + 1
    
    final_rank_dist = np.zeros((N, N))
    for sample in theta_samples:
        ranks = np.argsort(-sample)
        for rank, item in enumerate(ranks):
            final_rank_dist[item, rank] += 1
    final_rank_dist = np.round(final_rank_dist / theta_samples.shape[0], decimals=3)
    # Now final_rank_positions matches T
    return theta_mean, final_rank_positions, final_rank_dist, theta_samples  

def main_new():
    seed, method, N, K, sel = parse_args()
    validate_params(method, sel)
    
    print(f"Simulation settings:\nSeed: {seed}, Method: {method}, N: {N}, K: {K}, Selection: {sel}")
    
    mus, sigmas = generate_mus_sigmas(N, seed)
    R = compute_target_rank_density(mus, sigmas, n_mc=5000, seed=seed)
    T = rankdata(-mus, method='min').tolist()
    history = init_history(method, sel, N, K, seed, mus, sigmas, R, T)
    
    if method == "PBCJ":
        exp_hist, wins, losses = run_pbcj(N, K, sel, seed, history)
        alpha_init = beta_init = np.ones((N, N))
        final_seed = seed + N*K - 1
        r = compute_pbcj_rank_density(wins, losses, alpha_init, beta_init, n_mc=1000, seed=final_seed)
        rank_densities = None
        final_theta_samples = None
    else:
        exp_hist, wins, losses, final_theta_samples, rank_densities = run_bbtm(N, K, sel, seed, history)
        r = compute_estimated_rank_density(final_theta_samples)
        alpha_init = beta_init = None
    
    history["estimated_values"]["rank_density"] = r.tolist()
    history["estimated_values"]["expected_ranks"] = np.sum(r * np.arange(1, N+1), axis=1).tolist()
    history["comparison_history"]["outcomes"] = generate_outcomes(wins, losses, N) 
    history["comparison_history"]["expected_rank_history"] = [
        [round(float(rank), 3) for rank in iteration_ranks] 
        for iteration_ranks in exp_hist
    ]
    
    tau_hist, spearman_hist, jsd_hist = compute_history_metrics(
        exp_hist, T, R, method, wins, losses, alpha_init, beta_init, rank_densities, N, seed
    )
    history["comparison_history"].update({
        "tau_history": tau_hist, "spearman_history": spearman_hist, "jsd_history": jsd_hist
    })
    
    print("\nExpected Rank History:")
    for i, ranks in enumerate(exp_hist):
        formatted_ranks = [round(x, 3) for x in ranks]
        print(f"{i+1}: {formatted_ranks}")
    
    print(f"\nWins Matrix:\n{wins}")
    print(f"\nLosses Matrix:\n{losses}")
    
    jsd_scores = np.array([js_divergence(R[i], r[i], base=2) for i in range(N)])
    quality_df = pd.DataFrame({
        "Item no": range(N), "Mus": mus, "Sigmas": sigmas, "Rank": rankdata(-mus, method='min')
    })
    
    print(f"\nTrue Quality Scores and Target Rank:\n{quality_df.to_string(index=False)}")
    print(f"\nTarget Rank Density (R):\n{np.round(R, 3)}")
    print(f"\nEstimated Rank Density (r):\n{np.round(r, 3)}")
    print(f"\nJSD Scores (R vs r):\n{np.round(jsd_scores, 4)}")
    
    final_jsd_scores = [js_divergence(R[k], r[k], base=2) for k in range(N)]
    history["metrics"]["final_jsd_scores"] = [float(x) for x in final_jsd_scores]
    
    print("\n=== Final Ranking results ===")
    theta_mean, final_rank, final_rank_dist, _ = compute_final_results(wins, losses, N, seed)
    
    if theta_mean is not None:
        final_exp_rank = np.sum(final_rank_dist * np.arange(1, N+1), axis=1)
        ability_df = pd.DataFrame({
            "item ID": range(N),
            "Mean Ability Score": theta_mean,
            "Rank": final_rank,  # Now correctly uses rank positions
            "Expected Rank": final_exp_rank
        })
        
        history["estimated_values"].update({
            "final_ability_scores": theta_mean.tolist(),
            "final_ranks": final_rank.tolist(),  # Rank positions
            "final_rank_distribution": final_rank_dist.tolist()
        })
        
        print(f"1. Final Ranks: {','.join(map(str, final_rank))}")
        print(f"\n2. Final Ability Scores and Expected Rank:\n{ability_df.to_string(index=False)}")
    else:
        print("No comparisons generated, cannot compute final ranks.")
    
    print(f"\nFinal JSD Scores: {[round(x, 6) for x in final_jsd_scores]}")
    if jsd_hist:
        print(f"Last History JSD: {[round(x, 6) for x in jsd_hist[-1]]}")
        print(f"Are they equal? {np.allclose(final_jsd_scores, jsd_hist[-1])}")

    os.makedirs("history_n20_k20_bbtm", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"history_n20_k20_bbtm/{method}_{sel}_N{N}_K{K}_seed{seed}_{timestamp}.json"
    
    try:
        with open(filename, 'w') as f:
            json.dump(numpy_to_serializable(history), f, indent=4, ensure_ascii=False)
        print(f"\nhistory successfully saved to {filename}")
    except Exception as e:
        print(f"\nError saving history: {str(e)}")

if __name__ == "__main__":
    main_new()