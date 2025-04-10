import numpy as np

def generate_winner_from_pair(mu_vector, sigma_vector, seed=42):
    """Generate the index of the best sample from a pair of normal distributions.

    Args:
        mu_vector (array-like): Mean values for the normal distributions.
        sigma_vector (array-like): Standard deviations for the normal distributions.
        seed (int, optional): Seed for the random number generator. Defaults to 42.

    Returns:
        int: Index of the best sample.

    Example:
        mu_vector = [0, 1]
        sigma_vector = [1, 1.5]
        seed = 42
        best_index = generate_winner_from_pair(mu_vector, sigma_vector, seed)
        print(best_index)  # Output: Index of the sample with the highest value
    """
    # Ensure the input vectors have a length of 2
    assert len(mu_vector) == 2 and len(sigma_vector) == 2, "mu_vector and sigma_vector must each have a length of 2"
    
    # Set the random seed for reproducibility
    np.random.seed(seed)
    
    # Generate samples from a normal distribution using the provided mean (mu_vector) and standard deviation (sigma_vector)
    samples = np.random.normal(mu_vector, sigma_vector)
    
    # Find the index of the maximum value in the samples array
    best_ind = np.argmax(samples)
    
    # Return the index of the best sample
    return best_ind