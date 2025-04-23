import numpy as np
from scipy.stats import kendalltau, spearmanr, norm
from gibbs_sampling import gibbs_sampler_binomial

def kendall_tau_distance(rank1, rank2):
    """
    Compute the normalized Kendall Tau distance between two rankings.

    Args:
        rank1 (numpy.ndarray): First ranking of students/items.
        rank2 (numpy.ndarray): Second ranking of students/items.

    Returns:
        float: Normalized Kendall Tau distance in the range [0,1].
    """
    tau, _ = kendalltau(rank1, rank2)
    return (1 - tau) / 2 

def compute_expected_metrics(X, comparisons, current_rank, n_students, prior_precision, 
                           theta_mean, theta_cov, student_pairs, tau_seed_seq):
    """
    Compute expected Kendall Tau and Spearman correlation metrics for student rankings.

    This function estimates the impact of potential pairwise comparisons on the ranking 
    using Gibbs sampling and computes weighted expectations based on model probabilities.

    Args:
        X (scipy.sparse matrix): Feature matrix for ranking estimation.
        comparisons (list of tuples): List of comparisons (i, j, winner), where i and j are students compared.
        current_rank (numpy.ndarray): Current ranking of students.
        n_students (int): Number of students being ranked.
        prior_precision (numpy.ndarray): Prior precision matrix.
        theta_mean (numpy.ndarray): Mean ability estimates for students.
        theta_cov (numpy.ndarray): Covariance matrix of student abilities.
        student_pairs (list of tuples): List of student pairs for which metrics will be computed.
        tau_seed_seq (numpy.random.SeedSequence): Seed sequence for sampling.

    Returns:
        dict: Expected Kendall Tau distances for each student pair.
        dict: Expected Spearman correlation values for each student pair.
    """

    expected_taus = {}
    expected_spearmans = {}
    pair_seed_seq = tau_seed_seq.spawn(len(student_pairs) * 2)
    seed_idx = 0
    
    for i, j in student_pairs:
        # Case where j wins
        new_comparison_0 = [i, j, j]
        updated_comparisons_0 = comparisons + [new_comparison_0]
        new_rank_0 = gibbs_sampler_binomial(updated_comparisons_0, n_students, 
                                           prior_precision, 3000, 1000, 4, 
                                           pair_seed_seq[seed_idx])
        kappa_0 = kendall_tau_distance(current_rank, new_rank_0)
        rho_0 = spearmanr(current_rank, new_rank_0)[0]  # Take correlation value only
        seed_idx += 1
        
        # Case where i wins
        new_comparison_1 = [i, j, i]
        updated_comparisons_1 = comparisons + [new_comparison_1]
        new_rank_1 = gibbs_sampler_binomial(updated_comparisons_1, n_students, 
                                           prior_precision, 3000, 1000, 4, 
                                           pair_seed_seq[seed_idx])
        kappa_1 = kendall_tau_distance(current_rank, new_rank_1)
        rho_1 = spearmanr(current_rank, new_rank_1)[0]  # Take correlation value only
        seed_idx += 1
        
        # Probability calculation
        delta_mean = theta_mean[i] - theta_mean[j]
        delta_var = theta_cov[i, i] + theta_cov[j, j] - 2 * theta_cov[i, j]
        pr_i_gt_j = norm.cdf(delta_mean / np.sqrt(delta_var)) if delta_var > 0 else 0.5
        
        # Expected values
        expected_taus[(i, j)] = kappa_0 * (1 - pr_i_gt_j) + kappa_1 * pr_i_gt_j
        expected_spearmans[(i, j)] = rho_0 * (1 - pr_i_gt_j) + rho_1 * pr_i_gt_j
    
    return expected_taus, expected_spearmans