import numpy as np
from itertools import combinations

def simulate_pairwise_comparisons(mu_vector, sigma_vector, num_trials=1, seed=42):
    """
    Simulate all pairwise comparisons between students.

    Args:
        mu_vector (list): Mean skill level of each student.
        sigma_vector (list): Standard deviation of each student.
        num_trials (int): Number of comparisons per pair.
        seed (int): Random seed for reproducibility.

    Returns:
        results (list): List of tuples (i, j, winner) for each trial.
    """
    np.random.seed(seed)
    num_students = len(mu_vector)
    all_pairs = list(combinations(range(num_students), 2))
    results = []

    for i, j in all_pairs:
        for _ in range(num_trials):
            sample_i = np.random.normal(mu_vector[i], sigma_vector[i])
            sample_j = np.random.normal(mu_vector[j], sigma_vector[j])
            winner = i if sample_i > sample_j else j
            results.append((i, j, winner,sample_i,sample_j))

    return results
