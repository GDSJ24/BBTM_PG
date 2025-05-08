# Bayesian comparative judgment
import numpy as np
from scipy.special import beta as beta_function
from scipy.special import digamma
from scipy.stats import beta, truncnorm
# from BBTM_PG_SIMULATION.performance_metric import augmented_tchebychev, comparison_result

def generate_mus_sigmas(N, seed, typ_mu=60, typ_sigma=5, mlb=0, mub=100, sub=None):
    
    np.random.seed(seed)

    # Calculate the lower and upper bounds in terms of the standard normal distribution
    lower, upper = (mlb - typ_mu) / typ_sigma, (mub - typ_mu) / typ_sigma

    # Create the truncated normal distribution
    trunc_normal = truncnorm(lower, upper, loc=typ_mu, scale=typ_sigma)

    # Generate random samples
    mus = trunc_normal.rvs(N)

    # sigma samples
    if sub is None:
        sigmas = np.ones(N) * 5 # std dev of 5 - default same across items
    else:
        sigmas = np.random.random(size=N) * sub

    return mus, sigmas


def pbcj_cal_exp_ranks(rank_density): 
    # with equal probability gets the median value
    vals = np.arange(1, len(rank_density)+1, 1)
    return np.sum(rank_density*vals, axis=1)

def pbcj_MC_exp_rank(wins, losses, alpha_init, beta_init, n_mc=1000):
    n_items = len(wins)
    t_alpha = alpha_init + wins
    t_beta = beta_init + losses
    random_samples = np.round(beta.rvs(t_alpha, t_beta, size=(n_mc,n_items, n_items)), decimals=0)
    mask = np.repeat(np.array(np.identity(n_items), dtype=bool)[np.newaxis,:,:], n_mc, axis=0)
    masked_array = np.ma.array(random_samples, mask=mask)
    data = n_items - np.sum(masked_array, axis=2).data
    return np.average(data, axis=0)

def pbcj_MC_rank_density(wins, losses, alpha_init, beta_init, n_mc=1000):
    n_items = len(wins)
    t_alpha = alpha_init + wins
    t_beta = beta_init + losses
    random_samples = np.round(beta.rvs(t_alpha, t_beta, size=(n_mc,n_items, n_items)), decimals=0)
    mask = np.repeat(np.array(np.identity(n_items), dtype=bool)[np.newaxis,:,:], n_mc, axis=0)
    masked_array = np.ma.array(random_samples, mask=mask)
    data = n_items - np.sum(masked_array, axis=2).data
    return data

def pbcj_calc_entropy(wins, losses, alpha_init, beta_init):
    t_alpha = alpha_init + wins
    t_beta = beta_init + losses
    ent = np.zeros((len(t_alpha), len(t_alpha)))
    ent = np.log(beta_function(t_alpha, t_beta)) - (t_alpha-1)*digamma(t_alpha) \
            - (t_beta-1) * digamma(t_beta) + (t_alpha+t_beta-2)*digamma(t_alpha+t_beta)
    return ent

def pbcj_select_item_indices(wins, losses, alpha_init, beta_init, method="random"):
    if method == "random":
        ind_1, ind_2 = np.random.choice(np.arange(0, len(wins)), size=2, replace=False)
        return ind_1, ind_2
    if method == "entropy":
        ent = pbcj_calc_entropy(wins, losses, alpha_init, beta_init)
        np.fill_diagonal(ent, -1e100)
        ind_1, ind_2 = np.unravel_index(np.argmax(ent), ent.shape)
        return ind_1, ind_2
