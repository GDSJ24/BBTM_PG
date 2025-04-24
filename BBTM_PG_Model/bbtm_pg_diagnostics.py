import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az

def compute_r_hat(chains):
    """
    Compute R-hat convergence diagnostic.
    
    Args:
        chains (list): List of np.ndarray, each containing samples from a chain.
    
    Returns:
        pd.DataFrame: DataFrame with Student ID and R-hat values.
    """
    n_chains = len(chains)
    n_samples = chains[0].shape[0]
    n_students = chains[0].shape[1]

    chain_mean = np.array([np.mean(chain, axis=0) for chain in chains])
    overall_mean = np.mean(chain_mean, axis=0)
    B = (n_samples / (n_chains - 1)) * np.sum((chain_mean - overall_mean) ** 2, axis=0)

    W = np.mean([np.var(chain, axis=0) for chain in chains], axis=0)

    theta_var = (n_samples - 1) / n_samples * W + B / n_samples
    r_hat = np.sqrt(theta_var / W)

    rhat_df = pd.DataFrame({
        'Student ID': np.arange(n_students),
        'R hat': r_hat
    })
    return rhat_df

def compute_ess(chains):
    """
    Compute Effective Sample Size (ESS).
    
    Args:
        chains (list): List of np.ndarray, each containing samples from a chain.
    
    Returns:
        float: Effective Sample Size.
    """
    acfs = []
    for chain in chains:
        for param in range(chain.shape[1]):
            x = chain[:, param] - np.mean(chain[:, param])
            acf = np.correlate(x, x, mode='full')[-len(x):]
            acf = acf / acf[0]
            acfs.append(acf)

    max_lag = next((i for i, val in enumerate(np.mean(acfs, 0)) if val < 0), 10)
    ess = len(chains) * chains[0].shape[0] / (1 + 2 * np.sum(np.mean(acfs, 0)[:max_lag]))
    return ess

def plot_trace(chains, n_students):
    """
    Generate trace plots for each student.
    
    Args:
        chains (list): List of np.ndarray, each containing samples from a chain.
        n_students (int): Number of students.
    """
    plt.figure(figsize=(12, 12))
    for i in range(n_students):
        plt.subplot(6, 4, i + 1)
        for chain in chains:
            plt.plot(chain[:, i], alpha=0.5)
        plt.title(f'Trace plot of student {i}')
        plt.xlabel('Iterations')
        plt.ylabel(f'$\\theta_{i}$')
    plt.tight_layout()
    plt.savefig('trace_plots.png')
    plt.close()

def plot_posterior(comb_samples, n_students):
    """
    Generate posterior plots.
    
    Args:
        comb_samples (np.ndarray): Combined theta samples from all chains.
        n_students (int): Number of students.
    """
    posterior = {f'theta_{i}': comb_samples[:, i] for i in range(n_students)}
    inference_data = az.from_dict(posterior=posterior)
    az.plot_posterior(inference_data, figsize=(14, 8))
    plt.savefig('posterior_plots.png')
    plt.close()

def plot_forest(chains, n_students):
    """
    Generate forest plots.
    
    Args:
        chains (list): List of np.ndarray, each containing samples from a chain.
        n_students (int): Number of students.
    """
    posterior = {f'theta_{i}': np.array([chain[:, i] for chain in chains]) for i in range(n_students)}
    inference_data = az.from_dict(posterior=posterior)
    az.plot_forest(inference_data, combined=False, figsize=(10, 12))
    plt.xlabel('Theta_mean')
    plt.ylabel('Student')
    plt.savefig('forest_plots.png')
    plt.close()

def plot_posterior_distributions(comb_samples, n_students):
    """
    Generate posterior distribution plots with 95% credible intervals.
    
    Args:
        comb_samples (np.ndarray): Combined theta samples from all chains.
        n_students (int): Number of students.
    """
    plt.figure(figsize=(16, 12))
    for i in range(n_students):
        plt.subplot(6, 4, i + 1)
        sns.histplot(comb_samples[:, i], kde=True, color='blue', bins=30)
        lower, upper = np.percentile(comb_samples[:, i], [2.5, 97.5])
        plt.axvline(lower, color='red', linestyle='--', label='95% CI')
        plt.axvline(upper, color='red', linestyle='--')
        plt.title(f'Posterior distribution of student {i}')
        plt.xlabel(f'$\\theta_{i}$')
        plt.ylabel('Density')
        plt.legend()
    plt.tight_layout()
    plt.savefig('posterior_distributions.png')
    plt.close()

def plot_rank_probabilities(comb_samples, n_students):
    """
    Generate heatmap of rank probabilities.
    
    Args:
        comb_samples (np.ndarray): Combined theta samples from all chains.
        n_students (int): Number of students.
    """
    rank_counts = np.zeros((n_students, n_students), dtype=int)
    for theta in comb_samples:
        rank_order = np.argsort(-theta)
        ranks = np.argsort(rank_order) + 1
        for i in range(n_students):
            rank = ranks[i]
            rank_counts[i, rank - 1] += 1

    rank_prob = rank_counts / len(comb_samples)
    rank_labels = [f'Rank {r + 1}' for r in range(n_students)]
    prob_df = pd.DataFrame(
        rank_prob,
        columns=rank_labels,
        index=[f'Student {i}' for i in range(n_students)]
    )

    plt.figure(figsize=(16, 8))
    sns.heatmap(prob_df, annot=True, fmt='.2f', cmap='Blues')
    plt.title('Posterior Rank Probabilities')
    plt.xlabel('Rank')
    plt.ylabel('Students')
    plt.savefig('rank_probabilities.png')
    plt.close()