import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import entropy, rankdata, kendalltau, spearmanr
from pbcj import pbcj_cal_exp_ranks, pbcj_MC_exp_rank, pbcj_select_item_indices, compute_expected_metrics_pbcj
from bbtm import ComparisonGenerator, select_item_indices, build_design_matrix, gibbs_sampler_binomial_single, compute_expected_metrics, get_item_pairs
from utils import generate_mus_sigmas, compute_target_rank_density, compute_estimated_rank_density, js_divergence, compute_pbcj_rank_density
import json
import os
from datetime import datetime

def numpy_to_serializable(obj):
    """Convert numpy objects to JSON-serializable formats"""
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
    """Parse command line arguments"""
    args = {'-seed': 12345, '-m': "PBCJ", '-n': 5, '-k': 5, '-sel': "random"}
    for key in args:
        if key in sys.argv:
            idx = sys.argv.index(key) + 1
            args[key] = int(sys.argv[idx]) if key in ['-seed', '-n', '-k'] else sys.argv[idx]
    return args['-seed'], args['-m'], args['-n'], args['-k'], args['-sel']

def validate_params(method, sel):
    """Validate method and selection parameters"""
    valid_pbcj = ["random", "entropy", "round_robin", "no_repeating_pairs", "KG-Tau", "KG-Rho", "KG-Entropy"]
    valid_bbtm = ["random", "KG-Tau", "KG-Rho", "pair_entropy", "round_robin", "no_repeating_pairs", "KG-Entropy"]
    
    if method not in ["PBCJ", "BBTM"]:
        raise ValueError("Method must be 'PBCJ' or 'BBTM'")
    if method == "PBCJ" and sel not in valid_pbcj:
        raise ValueError(f"Selection method for PBCJ must be one of {valid_pbcj}")
    if method == "BBTM" and sel not in valid_bbtm:
        raise ValueError(f"Selection method for BBTM must be one of {valid_bbtm}")

def init_history(method, sel, N, K, seed, mus, sigmas, R, T):
    """Initialize history dictionary"""
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
    """Generate outcome list from wins/losses matrices"""
    outcomes = []
    for i in range(N):
        for j in range(i+1, N):
            for _ in range(int(wins[i,j])):
                outcomes.append([int(i), int(j), int(i)])
            for _ in range(int(losses[i,j])):
                outcomes.append([int(i), int(j), int(j)])
    return outcomes

def run_pbcj(N, K, selection_method, seed, history):
    """Run PBCJ method with KG metrics tracking"""
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
        
        # Compute KG metrics if using KG method
        if is_kg_method:
            expected_taus, expected_spearmans, expected_entropies = compute_expected_metrics_pbcj(
                wins, losses, alpha_init, beta_init, current_rank, N, seed=current_seed
            )
        
        ind_1, ind_2 = pbcj_select_item_indices(
            wins=wins, losses=losses, alpha_init=alpha_init, beta_init=beta_init,
            method=selection_method, total_comparisons=N*K, current_comparison_idx=i,
            current_rank=current_rank, n_items=N, seed=current_seed
        )
        
        # Store KG metrics for the selected pair
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
    """Run BBTM method with KG metrics tracking"""
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
        theta_mean = theta_cov = current_rank = None
        
        # Generate theta samples for KG methods or after first comparison
        if is_kg_method or selection_method == "pair_entropy" or i > 0:
            X = build_design_matrix(comparisons, n_items=N) if comparisons else sp.csr_matrix((0, N))
            theta_samples = gibbs_sampler_binomial_single(
                comparisons=comparisons, n_items=N, prior_precision=prior_precision,
                n_samples=100, n_burnins=50, n_chains=4, seed=current_seed
            ) if comparisons else np.zeros((1000, N))
            
            if is_kg_method:
                theta_mean = np.mean(theta_samples, axis=0)
                theta_cov = np.cov(theta_samples.T)
                current_rank = np.argsort(-theta_mean)
                item_pairs = get_item_pairs(n_items=N)
                
                # Compute KG metrics
                expected_taus, expected_spearmans, expected_entropies = compute_expected_metrics(
                    X=X,
                    comparisons=comparisons,
                    current_rank=current_rank,
                    n_items=N,
                    prior_precision=prior_precision,
                    theta_mean=theta_mean,
                    theta_cov=theta_cov,
                    item_pairs=item_pairs,
                    seed=current_seed
                )

        ind_1, ind_2 = select_item_indices(
            comparisons=comparisons, n_items=N, prior_precision=prior_precision,
            theta_mean=theta_mean, theta_cov=theta_cov, method=selection_method,
            total_comparisons=N*K, current_comparison_idx=i, seed=current_seed,
            current_rank=current_rank
        )
        
        # Store KG metrics for the selected pair
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
        
        # Final theta sampling for rank calculation
        theta_samples = gibbs_sampler_binomial_single(
            comparisons=comparisons, n_items=N, prior_precision=prior_precision,
            n_samples=3000, n_burnins=1000, n_chains=4, seed=current_seed
        )
        current_rank_density = compute_estimated_rank_density(theta_samples)
        rank_densities.append(current_rank_density)
        exp_rank = np.sum(current_rank_density * np.arange(1, N+1), axis=1)
        exp_hist.append(exp_rank)

    return np.array(exp_hist), wins, losses, theta_samples, rank_densities

def compute_history_metrics(exp_hist, T, R, method, wins, losses, alpha_init, beta_init, rank_densities, N, seed):
    """Compute tau, spearman, and JSD history"""
    tau_hist, spearman_hist, jsd_hist = [], [], []
    
    for i, current_t in enumerate(exp_hist):
        tau, _ = kendalltau(T, current_t)
        rho, _ = spearmanr(T, current_t)
        tau_hist.append(float(tau))
        spearman_hist.append(float(rho))
        
        if method == "PBCJ":
            current_r = compute_pbcj_rank_density(wins, losses, alpha_init, beta_init, n_mc=1000, seed=seed+i)
        else:
            current_r = rank_densities[i] if i < len(rank_densities) else np.ones((N, N))/N
        
        jsd_scores = [js_divergence(R[k], current_r[k]) for k in range(N)]
        jsd_hist.append([float(x) for x in jsd_scores])
    
    return tau_hist, spearman_hist, jsd_hist

def compute_final_results(wins, losses, N, seed):
    """Compute final ranking results"""
    comparisons = []
    for i in range(N):
        for j in range(i+1, N):
            # Add wins for item i against item j
            for _ in range(int(wins[i,j])):
                comparisons.append([i, j, i])
            # Add losses for item i against item j (i.e., wins for j against i)
            for _ in range(int(losses[i,j])):
                comparisons.append([i, j, j])
    
    if not comparisons:
        return None, None, None, None
        
    prior_precision = sp.eye(N)
    theta_samples = gibbs_sampler_binomial_single(
        comparisons=comparisons, n_items=N, prior_precision=prior_precision,
        n_samples=100, n_burnins=50, n_chains=4, seed=seed
    )
    theta_mean = np.mean(theta_samples, axis=0)
    final_rank = np.argsort(-theta_mean)
    
    # Calculate final rank distribution
    n_samples = theta_samples.shape[0]
    final_rank_dist = np.zeros((N, N))
    for sample in theta_samples:
        ranks = np.argsort(-sample)
        for rank, item in enumerate(ranks):
            final_rank_dist[item, rank] += 1
    final_rank_dist = np.round(final_rank_dist / n_samples, decimals=3)
    
    return theta_mean, final_rank, final_rank_dist, theta_samples

def main_new():
    seed, method, N, K, sel = parse_args()
    validate_params(method, sel)
    
    print(f"Simulation settings:\nSeed: {seed}, Method: {method}, N: {N}, K: {K}, Selection: {sel}")
    
    # Initialize true parameters and history
    mus, sigmas = generate_mus_sigmas(N, seed)
    R = compute_target_rank_density(mus, sigmas, n_mc=5000, seed=seed)
    T = rankdata(-mus, method='min').tolist()
    history = init_history(method, sel, N, K, seed, mus, sigmas, R, T)
    
    # Run simulation with KG metrics tracking
    if method == "PBCJ":
        exp_hist, wins, losses = run_pbcj(N, K, sel, seed, history)
        alpha_init = beta_init = np.ones((N, N))
        # Use the same seed for final rank density computation
        final_seed = seed + N*K - 1  # Same seed as last iteration
        r = compute_pbcj_rank_density(wins, losses, alpha_init, beta_init, n_mc=1000, seed=final_seed)
        rank_densities = None
        final_theta_samples = None
    else:  # BBTM
        exp_hist, wins, losses, final_theta_samples, rank_densities = run_bbtm(N, K, sel, seed, history)
        r = compute_estimated_rank_density(final_theta_samples)  # Use the final theta samples
        alpha_init = beta_init = None
    
    # Store history
    history["estimated_values"]["rank_density"] = r.tolist()
    history["estimated_values"]["expected_ranks"] = np.sum(r * np.arange(1, N+1), axis=1).tolist()
    history["comparison_history"]["outcomes"] = generate_outcomes(wins, losses, N) 
    history["comparison_history"]["expected_rank_history"] = [
        [round(float(rank), 3) for rank in iteration_ranks] 
        for iteration_ranks in exp_hist
    ]
    
    # Compute history metrics
    tau_hist, spearman_hist, jsd_hist = compute_history_metrics(
        exp_hist, T, R, method, wins, losses, alpha_init, beta_init, rank_densities, N, seed
    )
    history["comparison_history"].update({
        "tau_history": tau_hist, "spearman_history": spearman_hist, "jsd_history": jsd_hist
    })
    
    # Print Expected Rank History in the same format as main.py
    print("\nExpected Rank History:")
    for i, ranks in enumerate(exp_hist):
        formatted_ranks = [round(x, 3) for x in ranks]
        print(f"{i+1}: {formatted_ranks}")
    
    print(f"\nWins Matrix:\n{wins}")
    print(f"\nLosses Matrix:\n{losses}")
    
    # Display analysis
    jsd_scores = np.array([js_divergence(R[i], r[i]) for i in range(N)])
    quality_df = pd.DataFrame({
        "Item no": range(N), "Mus": mus, "Sigmas": sigmas, "Rank": rankdata(-mus, method='min')
    })
    
    print(f"\nTrue Quality Scores and Target Rank:\n{quality_df.to_string(index=False)}")
    print(f"\nTarget Rank Density (R):\n{np.round(R, 3)}")
    print(f"\nEstimated Rank Density (r):\n{np.round(r, 3)}")
    print(f"\nJSD Scores (R vs r):\n{np.round(jsd_scores, 4)}")
    
    # FIXED: Use the same rank density for final JSD scores as in history
    # This ensures consistency between final_jsd_scores and the last entry in jsd_history
    final_jsd_scores = [js_divergence(R[k], r[k]) for k in range(N)]
    history["metrics"]["final_jsd_scores"] = [float(x) for x in final_jsd_scores]
    
    # Final results
    print("\n=== Final Ranking results ===")
    theta_mean, final_rank, final_rank_dist, _ = compute_final_results(wins, losses, N, seed)
    
    if theta_mean is not None:
        history["estimated_values"].update({
            "final_ability_scores": theta_mean.tolist(),
            "final_ranks": final_rank.tolist(),
            "final_rank_distribution": final_rank_dist.tolist()
        })
        
        print(f"1. Final Ranks: {','.join(map(str, final_rank))}")
        
        rank_map = {item_id: rank + 1 for rank, item_id in enumerate(final_rank)}
        final_exp_rank = np.sum(final_rank_dist * np.arange(1, N+1), axis=1)
        ability_df = pd.DataFrame({
            "item ID": range(N), "Mean Ability Score": theta_mean,
            "Rank": [rank_map[i] for i in range(N)], "Expected Rank": final_exp_rank
        })
        print(f"\n2. Final Ability Scores and Expected Rank:\n{ability_df.to_string(index=False)}")
    else:
        print("No comparisons generated, cannot compute final ranks.")
    
    # VERIFICATION: Print both final and last history JSD scores for comparison
    print(f"\nFinal JSD Scores: {[round(x, 6) for x in final_jsd_scores]}")
    if jsd_hist:
        print(f"Last History JSD: {[round(x, 6) for x in jsd_hist[-1]]}")
        print(f"Are they equal? {np.allclose(final_jsd_scores, jsd_hist[-1])}")

    # Save history results
    os.makedirs("history", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"history/{method}_{sel}_N{N}_K{K}_seed{seed}_{timestamp}.json"
    
    try:
        with open(filename, 'w') as f:
            json.dump(numpy_to_serializable(history), f, indent=4, ensure_ascii=False)
        print(f"\nhistory successfully saved to {filename}")
    except Exception as e:
        print(f"\nError saving history: {str(e)}")

if __name__ == "__main__":
    main_new()