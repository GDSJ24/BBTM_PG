# Bayesian comparative judgment
import numpy as np
from scipy.special import beta as beta_function
from scipy.special import digamma
from scipy.stats import beta
from performance_metric import augmented_tchebychev, comparison_result

def cal_exp_ranks(rank_density): 
    # with equal probability gets the median value
    vals = np.arange(1, len(rank_density)+1, 1)
    return np.sum(rank_density*vals, axis=1)

def MC_exp_rank(wins, losses, alpha_init, beta_init, n_mc=10000):
    n_items = len(wins)
    t_alpha = alpha_init + wins
    t_beta = beta_init + losses
    random_samples = np.round(beta.rvs(t_alpha, t_beta, size=(n_mc,n_items, n_items)), decimals=0)
    mask = np.repeat(np.array(np.identity(n_items), dtype=bool)[np.newaxis,:,:], n_mc, axis=0)
    masked_array = np.ma.array(random_samples, mask=mask)
    data = n_items - np.sum(masked_array, axis=2).data
    return np.average(data, axis=0)

def MC_rank_density(wins, losses, alpha_init, beta_init, n_mc=10000):
    n_items = len(wins)
    t_alpha = alpha_init + wins
    t_beta = beta_init + losses
    random_samples = np.round(beta.rvs(t_alpha, t_beta, size=(n_mc,n_items, n_items)), decimals=0)
    mask = np.repeat(np.array(np.identity(n_items), dtype=bool)[np.newaxis,:,:], n_mc, axis=0)
    masked_array = np.ma.array(random_samples, mask=mask)
    data = n_items - np.sum(masked_array, axis=2).data
    return data

def calc_entropy(wins, losses, alpha_init, beta_init):
    t_alpha = alpha_init + wins
    t_beta = beta_init + losses
    ent = np.zeros((len(t_alpha), len(t_alpha)))
    ent = np.log(beta_function(t_alpha, t_beta)) - (t_alpha-1)*digamma(t_alpha) \
            - (t_beta-1) * digamma(t_beta) + (t_alpha+t_beta-2)*digamma(t_alpha+t_beta)
    return ent

def select_item_indices(expected_rank, entropy, method="greedy", epsilon=0.1):
    if method=="greedy":
        ind_1 = np.argmin(expected_rank)
        masked_ent = np.ma.masked_array(entropy[ind_1], mask=False)
        masked_ent.mask[ind_1] = True
        ind_2 = np.argmax(masked_ent)
        # import pdb; pdb.set_trace()
        return ind_1, ind_2
    elif method=="all":
        mask = np.array(np.identity(len(entropy)), dtype=bool)
        masked_ent = np.ma.masked_array(entropy, mask=mask)
        inds = np.unravel_index(np.argmax(masked_ent), shape=(len(entropy), len(entropy)))
        return inds
    elif method=="egreedy":
        r = np.random.random()
        if r >= epsilon: # select best
            ind_1 = np.argmin(expected_rank)
        else:
            # ind_1 = np.random.randint(len(expected_rank))
            ind_1, ind_2 = np.random.choice(np.arange(0, len(expected_rank)), size=2, replace=False)
            return ind_1, ind_2
        masked_ent = np.ma.masked_array(entropy[ind_1], mask=False)
        masked_ent.mask[ind_1] = True
        ind_2 = np.argmax(masked_ent)
        # import pdb; pdb.set_trace()
        return ind_1, ind_2

def bcj(items, n_rounds, weight_vector, rho, sel_method="all"):
    n_items = len(items)
    alpha_init = np.ones((n_items, n_items)) # prior
    beta_init = np.ones((n_items, n_items)) # prior
    # initialise matrix for wins
    wins = np.zeros((n_items, n_items))
    # initialise matrix for losses
    losses = np.zeros((n_items, n_items))
    # initialise rank density matrix
    rank_density = np.ones((n_items, n_items))/n_items # prior
    # initialise expected rank list
    exp_rank = cal_exp_ranks(rank_density)
    # calculate current entropy matrix
    entropy = calc_entropy(wins, losses, alpha_init, beta_init)
    hist = []
    ind_hist = []
    exp_hist = []
    for i in range(n_rounds):
        # pick item indices
        ind_1, ind_2 = select_item_indices(exp_rank, entropy, sel_method)
        # print(ind_1, ind_2)
        # import pdb; pdb.set_trace()
        # compare and get result -- update win/loss matrices
        win = comparison_result(items, ind_1, ind_2, weight_vector, rho)
        # print(win)
        wins[ind_1, ind_2] += win
        wins[ind_2, ind_1] += (1-win)
        losses[ind_1, ind_2] += (1 - win)
        losses[ind_2, ind_1] += win
        # update entropy
        entropy = calc_entropy(wins, losses, alpha_init, beta_init)
        # update rank density
        # pass
        # update exp rank
        exp_rank = cal_exp_ranks(rank_density)
        # exp_rank = 
        # print(np.argmin(exp_rank))
        hist.append(np.argmin(exp_rank))
        # ind_hist.append([ind_1, ind_2, win])
        exp_hist.append(exp_rank)
        # stopping criterion?
    # return rank density and exp rank
    # return exp_rank, np.array(hist), np.array(ind_hist), np.array(exp_hist)
    return np.array(hist), np.array(exp_hist), wins, losses 


def bcj_with_prior(items, n_rounds, weight_vector, rho, wins, losses):
    n_items = len(items)
    alpha_init = np.ones((n_items, n_items)) # prior
    beta_init = np.ones((n_items, n_items)) # prior
    # initialise expected rank list
    exp_rank = MC_exp_rank(wins, losses, alpha_init, beta_init)
    # calculate current entropy matrix
    entropy = calc_entropy(wins, losses, alpha_init, beta_init)
    hist = []
    ind_hist = []
    exp_hist = []
    for i in range(n_rounds):
        # pick item indices
        ind_1, ind_2 = select_item_indices(exp_rank, entropy, sel_method)
        # print(ind_1, ind_2)
        # import pdb; pdb.set_trace()
        # compare and get result -- update win/loss matrices
        win = comparison_result(items, ind_1, ind_2, weight_vector, rho)
        # print(win)
        wins[ind_1, ind_2] += win
        wins[ind_2, ind_1] += (1-win)
        losses[ind_1, ind_2] += (1 - win)
        losses[ind_2, ind_1] += win
        # update entropy
        entropy = calc_entropy(wins, losses, alpha_init, beta_init)
        # update rank density
        # pass
        # update exp rank
        # exp_rank = cal_exp_ranks(rank_density)
        exp_rank = MC_exp_rank(wins, losses, alpha_init, beta_init)
        # print(np.argmin(exp_rank))
        hist.append(np.argmin(exp_rank))
        # ind_hist.append([ind_1, ind_2, win])
        exp_hist.append(exp_rank)
        # stopping criterion?
    # return rank density and exp rank
    # return exp_rank, np.array(hist), np.array(ind_hist), np.array(exp_hist)
    return np.array(hist), np.array(exp_hist), wins, losses 

def bcj_one_step(items, n_rounds, weight_vector, rho, wins, losses):
    n_items = len(items)
    alpha_init = np.ones((n_items, n_items)) # prior
    beta_init = np.ones((n_items, n_items)) # prior
    # initialise expected rank list
    exp_rank = MC_exp_rank(wins, losses, alpha_init, beta_init)
    # calculate current entropy matrix
    entropy = calc_entropy(wins, losses, alpha_init, beta_init)
    hist = []
    ind_hist = []
    exp_hist = []
    for i in range(n_rounds):
        # pick item indices
        ind_1, ind_2 = select_item_indices(exp_rank, entropy, sel_method)
        # print(ind_1, ind_2)
        # import pdb; pdb.set_trace()
        # compare and get result -- update win/loss matrices
        win = comparison_result(items, ind_1, ind_2, weight_vector, rho)
        # print(win)
        wins[ind_1, ind_2] += win
        wins[ind_2, ind_1] += (1-win)
        losses[ind_1, ind_2] += (1 - win)
        losses[ind_2, ind_1] += win
        # update entropy
        entropy = calc_entropy(wins, losses, alpha_init, beta_init)
        # update rank density
        # pass
        # update exp rank
        # exp_rank = cal_exp_ranks(rank_density)
        exp_rank = MC_exp_rank(wins, losses, alpha_init, beta_init)
        # print(np.argmin(exp_rank))
        hist.append(np.argmin(exp_rank))
        # ind_hist.append([ind_1, ind_2, win])
        exp_hist.append(exp_rank)
        # stopping criterion?
    # return rank density and exp rank
    # return exp_rank, np.array(hist), np.array(ind_hist), np.array(exp_hist)
    return np.array(hist), np.array(exp_hist), wins, losses 