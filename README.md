# Bayesian Bradley-Terry Model with Polya-Gamma Data Augmentation for Ranking Paired Comparisons 

## 1. Model Setup

### Bradley-Terry Likelihood

For a comparison between individuals *i* and *j* where *i* wins, the probability is:

```math
P(i \text{ beats } j) = \frac{e^{\theta_i}}{e^{\theta_i} + e^{\theta_j}} = \sigma(\theta_i - \theta_j)
```

where $\sigma$ is the logistic function, and $\theta_i$ is the latent strength of individual *i*.

### Polya-Gamma Augmentation

Introduce a latent variable $\omega_{ij} \sim \text{PG}(1, \theta_i - \theta_j)$ for each comparison.  
This allows Gibbs sampling with conjugate Gaussian updates for $\theta$.

---

## 2. Data Preparation


```python
comparisons = [[i,j, winner]]
```

### Design Matrix

Each row corresponds to a comparison and encodes `+1` for the winner and `-1` for the loser.

---

## 3. Gibbs Sampling Algorithm

### Steps

**Initialize Parameters:**

```python
theta = [0, 0, 0, 0, 0]  # latent strengths
omega = [1, 1, ..., 1]   # PG variables
```

**Sample $\omega_{ij}$:**

For each comparison $(i, j, k)$, compute:

```math
c = \theta_i - \theta_j
\quad \text{then draw} \quad \omega_{ij} \sim \text{PG}(1, c)
```

**Update $\theta$:**

1. Construct the posterior precision matrix:

```math
V^{-1} = X^T \Omega X + I
```

where:
- $X$ is the design matrix  
- $\Omega = \text{diag}(\omega)$  
- $I$ is the identity matrix

2. Compute the posterior mean:

```math
m = V \cdot X^T \kappa, \quad \text{where } \kappa = y - 0.5 = 0.5
```

3. Draw:

```math
\theta \sim \mathcal{N}(m, V)
```

**Center $\theta$ for identifiability:**  
Subtract the mean of $\theta$ at each iteration.

---

