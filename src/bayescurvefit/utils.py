import warnings
from typing import Callable, List, Tuple

import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.stats import gaussian_kde, truncnorm
from sklearn.mixture import GaussianMixture


def show_warning(message: str, verbose_level: int, verbose: int):
    if verbose_level <= verbose:
        warnings.warn(message, stacklevel=999)


def ols_fitting(
    x_data: np.ndarray,
    y_data: np.ndarray,
    fit_func: Callable,
    bounds: List[Tuple[List[float], List[float]]],
    init_guess: List[float] = None,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform OLS fitting.

    Args:
        x_data : Array of x values corresponding to the y data.
        y_data : Array of y data to be fitted.
        fit_func : The function to fit to the data.
        bounds : Bounds for the parameters as a tuple (lower_bounds, upper_bounds), one pair for each parameter.
        init_guess : Initial guess for the parameters. If None, guess from the mean of bounds.

    Returns:
        Array of fitted parameters, fitted predicted y values, and fitting errors.
    """

    kwargs.setdefault("maxfev", 10000)
    kwargs.setdefault("method", "trf")

    if init_guess is None:
        init_guess = [np.mean(b) for b in bounds]

    # reorg for fitting
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])

    valid_mask = ~np.isnan(y_data)
    x_fit = x_data[valid_mask]
    y_fit = y_data[valid_mask]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        fit_params, _ = curve_fit(
            fit_func,
            x_fit,
            y_fit,
            p0=init_guess,
            bounds=[lower_bounds, upper_bounds],
            **kwargs
        )
    y_preds = fit_func(x_fit, *fit_params)
    fit_errors = np.array(y_preds - y_fit)

    return fit_params, y_preds, fit_errors


def check_none(value, message):
    if value is None:
        raise ValueError(message)
    return value


def geweke_diag(x: np.ndarray, frac1: float = 0.1, frac2: float = 0.5) -> np.ndarray:
    """
    Geweke diagnostic to assess convergence of MCMC chains; it is useful for estimating burnin length for MCMC starting from random starting locations. Converted from R package coda https://cran.r-project.org/web/packages/coda/

    Args:
        x: MCMC chain.
        frac1 : Fraction of the beginning of the chain.
        frac2 : Fraction of the end of the chain.

    Returns:
        Geweke z-scores.
    """
    if not (0 <= frac1 <= 1):
        raise ValueError("frac1 must be between 0 and 1.")
    if not (0 <= frac2 <= 1):
        raise ValueError("frac2 must be between 0 and 1.")
    if frac1 + frac2 > 1:
        raise ValueError("start and end sequences are overlapping.")

    n_samples = x.shape[0]
    end1 = int(frac1 * n_samples)
    start2 = int((1 - frac2) * n_samples)

    segments = [x[:end1], x[start2:]]
    means = [np.mean(segment, axis=0) for segment in segments]
    variances = [np.var(segment, axis=0, ddof=1) for segment in segments]

    z = (means[0] - means[1]) / np.sqrt(
        variances[0] / len(segments[0]) + variances[1] / len(segments[1])
    )
    return z


def split_chains(chains: np.ndarray):
    """
    Split each MCMC chain into two halves as recommede by Gelman in Bayesian Data Analysis (3rd edition). If the number of samples is odd, ignore the first sample.

    Args:
        chains: MCMC chains with shape (n_samples, n_chains, n_parameters).

    Returns:
        splited MCMC chains.
    """
    n_samples = chains.shape[0]
    if n_samples % 2 != 0:
        chains = chains[
            1:, :, :
        ]  # Ignore the first sample if the number of samples is odd
    half = chains.shape[0] // 2
    return np.concatenate([chains[:half, :, :], chains[half:, :, :]], axis=1)


def gelman_rubin(chains: np.ndarray, return_tot_var: bool = False) -> np.ndarray:
    """
    Gelman-Rubin diagnostic to assess convergence of MCMC chains. Based on the updated methodology from Bayesian Data Analysis (3rd edition).

    Args:
        chains: MCMC chains.
        return_tot_var: If True, return the variance estimate. Otherwise, return the R-hat values.

    Returns:
        Gelman-Rubin R-hat values for each parameter, or the total variance estimate if return_tot_var is True.
    """
    if not return_tot_var:
        chains = split_chains(
            chains
        )  # Gelman recommended to split each chain in half for r_hat calculation so the within chain mixing at different time is better represented
    n = chains.shape[0]
    chain_means = np.mean(chains, axis=0)
    b = n * np.var(chain_means, axis=0, ddof=1)
    w = np.mean(np.var(chains, axis=0, ddof=1), axis=0)
    var_hat_plus = (n - 1) * w / n + b / n
    return var_hat_plus if return_tot_var else np.sqrt(var_hat_plus / w)


def variogram(chains: np.ndarray, t: int) -> np.ndarray:
    """
    Calculate the variogram V_t at each lag t for MCMC chains.

    Args:
        chains: MCMC chains with shape (n_samples, n_chains, n_parameters).
        t: lag t to calculate the variogram.

    Returns:
        np.ndarray: Variogram values V_t for each lag t, with shape (n_parameters).
    """
    n, m, _ = chains.shape
    diffs = chains[t:, :, :] - chains[:-t, :, :]
    squared_diffs = np.sum(diffs**2, axis=(0, 1))
    variograms = squared_diffs / (m * (n - t))
    return variograms


def calculate_effective_size(chains: np.ndarray, threshold: float = 0.0):
    """
    Calculate the effective sample size (ESS) for MCMC chains using the partial sum method.

    Args:
        chains: MCMC chains with shape (n_samples, n_chains, n_parameters).
        max_lag: Maximum number of lags to compute ACF for.

    Returns:
        np.ndarray: Effective sample sizes for each parameter.
    """

    n_samples, n_chains, n_parameters = chains.shape

    max_lag = int(n_samples / 2)
    jump = max(int(max_lag / 1000), 1)
    ess = np.zeros(n_parameters)
    p_hats = [
        1 - (variogram(chains, i) / (2 * gelman_rubin(chains, return_tot_var=True)))
        for i in np.arange(1, max_lag)[::jump]
    ]
    T = (
        np.argmax(np.transpose(p_hats).mean(axis=0) < threshold) * jump
    )  # Gelman suggest T should be the first odd positive integer for which p^_T+1 + p^_T+2 is negative, but here we just take the first mean of all params < threshold as an estimate
    if T > 0:
        p_hat_sum = np.sum(
            [
                1
                - (
                    variogram(chains, t)
                    / (2 * gelman_rubin(chains, return_tot_var=True))
                )
                for t in range(1, T)
            ],
            axis=0,
        )
        ess = n_samples * n_chains / (1 + 2 * (p_hat_sum))
    return ess


def calculate_bic(log_likelihood, num_params, num_data_points):
    return num_params * np.log(num_data_points) - 2 * log_likelihood


def fit_posterior(data: np.ndarray, max_components: int = 10, bw_method="scott") -> GaussianMixture:
    """
    Fit posterior distribution using a Gaussian Mixture Model based on KDE resampling.

    Args:
        data: 2D array of shape (n_params, n_samples) representing samples from the posterior.
        max_components: Maximum number of GMM components to try.
        bw_method: Bandwidth method for KDE ("scott", "silverman", or float).

    Returns:
        best_gmm: The best-fit GaussianMixture model selected via BIC.
    """
    if data.ndim != 2:
        raise ValueError("Input data must be 2D with shape (n_params, n_samples)")

    # KDE fit and sample
    kde = gaussian_kde(data, bw_method=bw_method)
    kde_samples = kde.resample(size=10000).T  # shape (10000, n_params)

    # Fit GMMs
    gmms = [
        GaussianMixture(n_components=n, covariance_type="full", max_iter=1000, random_state=0).fit(kde_samples)
        for n in range(1, max_components + 1)
    ]

    # Select best GMM via BIC
    bics = [gmm.bic(kde_samples) for gmm in gmms]
    best_gmm = gmms[np.argmin(bics)]
    return best_gmm


def calc_bma(best_gmm: GaussianMixture):
    """
    Calculate the Bayesian Model Averaging (BMA) mean and covariance from a multivariate GMM.

    Args:
        best_gmm: GaussianMixture instance from fit_posterior.

    Returns:
        bma_mean: Weighted average of means across components (shape: [n_params]).
        bma_cov: Full BMA covariance matrix (shape: [n_params, n_params]).
    """
    means = best_gmm.means_  # shape: (n_components, n_params)
    covariances = best_gmm.covariances_  # shape: (n_components, n_params, n_params)
    weights = best_gmm.weights_  # shape: (n_components,)

    # Compute BMA mean
    bma_mean = np.average(means, axis=0, weights=weights)

    # Compute BMA covariance using law of total variance
    bma_cov = sum(
        w * (cov + np.outer(mean - bma_mean, mean - bma_mean))
        for w, mean, cov in zip(weights, means, covariances)
    )

    return bma_mean, bma_cov


def compute_pep(bic0: float, bic1: float):
    """
    Compute prosterior error probability

    Args:
        bic0: BIC of null model
        bic1: BIC of alternative model

    Returns:
        null model probability
    """
    delta0 = bic0 - min(bic0, bic1)
    delta1 = bic1 - min(bic0, bic1)
    return np.exp(-0.5 * delta0) / (np.exp(-0.5 * delta0) + np.exp(-0.5 * delta1))


def truncated_normal(loc, scale, lower, upper, num_sim_samples, seed=42):
    """
    Generate samples from a truncated normal distribution with specific bounds.

    Args:
        loc (float): Mean of the normal distribution.
        scale (float): Standard deviation of the normal distribution.
        lower (float): Lower bound of the truncated distribution.
        upper (float): Upper bound of the truncated distribution.
        num_sim_samples (int): Number of samples to generate.

    Returns:
        np.ndarray: Samples from the truncated normal distribution.
    """
    a, b = (lower - loc) / scale, (upper - loc) / scale
    samples = truncnorm.rvs(
        a,
        b,
        loc=loc,
        scale=scale,
        size=num_sim_samples,
        random_state=np.random.default_rng(seed),
    )
    return samples
