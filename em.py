"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture
from scipy.stats import multivariate_normal

def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    K = mixture.mu.shape[0]
    mu = mixture.mu
    var = mixture.var
    p = mixture.p

    def normal_dist(x, mean, sd):
        prob_density = np.zeros(x.shape[0])
        d = mean.shape[0]
        for i in range(x.shape[0]):
            exponent = -0.5 * np.sum(((x[i] - mean)**2)/sd)
            prefactor = 1 / ((2 * np.pi*np.prod(sd)) ** (d/2))
            prob_density[i] = prefactor * np.exp(exponent)
        return prob_density


    # Compute the probabilities of each data point belonging to each Gaussian component
    likelihood = np.zeros((X.shape[0], K))
    for k in range(K):
        likelihood[:, k] = normal_dist(X, mu[k], var[k])
    numerator = likelihood * p
    denominator = numerator.sum(axis=1, keepdims=True)
    post = numerator / denominator

    # Compute the log-likelihood of the assignment
    ll = np.log(denominator).sum()

    return post, ll

    
    raise NotImplementedError


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    N, D = X.shape
    K = post.shape[1]

    # Update the prior probabilities
    p = post.mean(axis=0)

    # Update the means
    mu = np.zeros((K, D))
    for k in range(K):
        mu[k] = (post[:, k] @ X).ravel() / post[:, k].sum()

    # Update the variances
    var = np.zeros((K,))
    for k in range(K):
        diff = X - mu[k]
        var[k] = np.matmul(post[:, k], np.sum(diff**2, axis=1)) / (post[:, k]*D).sum()

    return GaussianMixture(mu, var, p)
    raise NotImplementedError


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    ll_old = -np.inf
    ll_new = None
    eps = 1e-6
    while (ll_new - ll_old <= eps*np.abs(ll_new)):
        # E-step
        post, ll_new = estep(X, mixture)

        # M-step
        mixture = mstep(X, post)

        # Log-likelihood
        ll_old = ll_new

    return mixture, post, ll_new
    raise NotImplementedError


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError
