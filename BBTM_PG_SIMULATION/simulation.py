from ranking import create_ability_df

def print_metrics(expected_taus, expected_spearmans):
    """
    Print the expected Kendall's Tau distances and Spearman's Rank correlations for student rankings.

    This function identifies the most disruptive student pair based on Kendall's Tau distance 
    and Spearman's Rank correlation, providing a comparison between the two metrics.

    Args:
        expected_taus (dict): Dictionary mapping student pairs to their expected Kendall's Tau distances.
        expected_spearmans (dict): Dictionary mapping student pairs to their expected Spearman's Rank correlations.

    Returns:
        None: Prints the computed metrics and comparisons.
    """
    print("\nExpected Kendall's Tau Distances:")
    for pair, tau in expected_taus.items():
        print(f"Pair {pair}: {tau:.4f}")
    tau_best_pair = max(expected_taus, key=expected_taus.get)
    print(f"Most disruptive pair (max E(tau_distance)): {tau_best_pair} with E(tau_distance) = {expected_taus[tau_best_pair]:.4f}")

    print("\nExpected Spearman's Rank Correlations:")
    for pair, spearman in expected_spearmans.items():
        print(f"Pair {pair}: {spearman:.4f}")
    rho_best_pair = min(expected_spearmans, key=expected_spearmans.get)
    print(f"Most disruptive pair (min E(rho)): {rho_best_pair} with E(rho) = {expected_spearmans[rho_best_pair]:.4f}")

    print("\nComparison:")
    if tau_best_pair == rho_best_pair:
        print(f"Both metrics agree: {tau_best_pair} is the most disruptive pair.")
    else:
        print(f"Difference detected:")
        print(f"  Kendall's Tau picks {tau_best_pair} with E(tau_distance) = {expected_taus[tau_best_pair]:.4f}")
        print(f"  Spearman's Rho picks {rho_best_pair} with E(rho) = {expected_spearmans[rho_best_pair]:.4f}")

def print_simulation_results(original_ranks, theta_mean, tau_new_rank, tau_theta_mean, 
                           rho_new_rank, rho_theta_mean, n_students):
    """
    Print simulation results for ranking changes based on expected metrics.

    This function prints the original rankings, ability scores, and updated rankings 
    based on maximum expected Kendall's Tau distance and minimum expected Spearman's Rho.

    Args:
        original_ranks (list of int): Original ranking order of students.
        theta_mean (numpy.ndarray): Mean ability scores of students.
        tau_new_rank (list of int): New ranking order based on maximum expected Kendall's Tau.
        tau_theta_mean (numpy.ndarray): Mean ability scores based on Kendall's Tau ranking.
        rho_new_rank (list of int): New ranking order based on minimum expected Spearman's Rho.
        rho_theta_mean (numpy.ndarray): Mean ability scores based on Spearman's Rho ranking.
        n_students (int): Total number of students being ranked.

    Returns:
        None: Prints formatted results including rankings and ability scores.
    """

    print("\n=== Simulation Results ===")
    
    # 1. Original Ranks
    print("1. Original Ranks:", ",".join(map(str, original_ranks)))
    
    # 2. Ability Scores based on Original Ranks
    print("\n2. Original Ability Scores:")
    original_ability_df = create_ability_df(range(n_students), theta_mean, original_ranks)
    print(original_ability_df.to_string(index=False))
    
    # 3. New Rank Order - Max Expected Kendalls Tau
    print("\n3. New Rank Order (Max Expected Kendall Tau):", ",".join(map(str, tau_new_rank)))
    
    # 4. Ability Scores - Max Kendalls Tau
    print("\n4. Ability Scores (Max Expected Kendall Tau):")
    tau_ability_df = create_ability_df(range(n_students), tau_theta_mean, tau_new_rank)
    print(tau_ability_df.to_string(index=False))
    
    # 5. New Rank Order - Min Expected Spearmans Rho
    print("\n5. New Rank Order (Min Expected Spearman Rho):", ",".join(map(str, rho_new_rank)))
    
    # 6. Ability Scores - Min Spearmans Rho
    print("\n6. Ability Scores (Min Expected Spearman Rho):")
    rho_ability_df = create_ability_df(range(n_students), rho_theta_mean, rho_new_rank)
    print(rho_ability_df.to_string(index=False))