import numpy as np
import pandas as pd
from scipy.stats import norm

def get_rank_map(rank_order):

    """
    Generate a mapping of student IDs to their respective ranks.

    Args:
        rank_order (list of int): List of student IDs sorted by rank.

    Returns:
        dict: Dictionary mapping student IDs to their ranks (starting from 1).
    """
    return {student_id: rank + 1 for rank, student_id in enumerate(rank_order)}

def create_ability_df(student_ids, theta_mean, rank_order):
    """
    Create a DataFrame containing student ability scores and rankings.

    Args:
        student_ids (list of int): List of student IDs.
        theta_mean (numpy.ndarray): Array of mean ability scores for students.
        rank_order (list of int): List of student IDs sorted by rank.

    Returns:
        pandas.DataFrame: DataFrame with columns "Student ID", "Mean Ability Score", and "Rank".
    """
    rank_map = get_rank_map(rank_order)
    return pd.DataFrame({
        "Student ID": student_ids,
        "Mean Ability Score": theta_mean,
        "Rank": [rank_map[i] for i in student_ids]
    })

def determine_winner(pair, theta_mean, theta_cov):
    """
    Determine the winner between two students based on ability estimates.

    The winner is decided probabilistically using a normal distribution based on 
    the difference in mean ability and covariance.

    Args:
        pair (tuple of int): Pair of student IDs being compared.
        theta_mean (numpy.ndarray): Array of mean ability scores for students.
        theta_cov (numpy.ndarray): Covariance matrix of student abilities.

    Returns:
        int: Student ID of the predicted winner.
    """
    i, j = pair
    delta_mean = theta_mean[i] - theta_mean[j]
    delta_var = theta_cov[i, i] + theta_cov[j, j] - 2 * theta_cov[i, j]
    return i if norm.cdf(delta_mean / np.sqrt(delta_var)) > 0.5 else j