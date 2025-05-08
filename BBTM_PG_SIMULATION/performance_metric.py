from scipy.stats import kendalltau, spearmanr, norm

def kendall_tau_distance(rank1, rank2):
    """Calculate Kendall's tau distance between two rankings.

    Args:
        rank1 (array-like): The first ranking.
        rank2 (array-like): The second ranking.

    Returns:
        float: The normalized Kendall's tau distance.

    Example:
        rank1 = [1, 2, 3, 4]
        rank2 = [4, 3, 2, 1]
        distance = kendall_tau_distance(rank1, rank2)
        print(distance)  # Output: Normalized Kendall's tau distance
    """
    # Calculate Kendall's tau correlation coefficient and ignore the p-value
    tau, _ = kendalltau(rank1, rank2)
    
    # Normalize the tau value to get the distance
    return (1 - tau) / 2

def spearman_rho_distance(rank1, rank2):
    """Calculate Spearman's rho distance between two rankings.

    Args:
        rank1 (array-like): The first ranking.
        rank2 (array-like): The second ranking.

    Returns:
        float: The normalized Spearman's rho distance.

    Example:
        rank1 = [1, 2, 3, 4]
        rank2 = [4, 3, 2, 1]
        distance = spearman_rho_distance(rank1, rank2)
        print(distance)  # Output: Spearman's rho distance
    """
    # Calculate Spearman's rho correlation coefficient and ignore the p-value
    rho, _ = spearmanr(rank1, rank2)
    
    # Normalize the rho value to get the distance
    return rho

