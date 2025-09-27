from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
from sklearn.mixture import GaussianMixture

from .utils import calculate_bic, check_none, fit_posterior


@dataclass
class SA_Results:
    SAMPLES_: dict = None
    LogP_: dict = None
    RESULTS_: dict = None

    @property
    def samples(self):
        return check_none(
            self.SAMPLES_,
            "No recorded samples, please run run_simulated_annealing first.",
        )

    @property
    def logp(self):
        return check_none(
            self.LogP_,
            "No recorded log probability values, please run run_simulated_annealing first.",
        )

    @property
    def results(self):
        return check_none(
            self.RESULTS_,
            "No recorded results, please run run_simulated_annealing first.",
        )

    def get_optimal(self):
        max_logp_index = np.argmax(list(self.logp))
        return self.SAMPLES_[max_logp_index]


@dataclass
class MCMC_Results:
    CONVERGE_: bool = False
    SAMPLES_: np.ndarray = None
    LogP_: np.ndarray = None
    R_hat_: np.ndarray = None
    EFFECTIVE_SIZES_: np.ndarray = None
    best_gmms: List[GaussianMixture] = None

    def get_samples(self, flat=False):
        samples = (
            self.SAMPLES_.reshape(-1, self.SAMPLES_.shape[-1])
            if flat
            else self.SAMPLES_
        )
        return check_none(samples, "No recorded samples, please run run_mcmc first.")

    @property
    def logp(self):
        return check_none(
            self.LogP_, "No recorded log probability values, please run run_mcmc first."
        )

    @property
    def effective_sizes(self):
        return check_none(
            self.EFFECTIVE_SIZES_,
            "No recorded effective sizes, please run run_mcmc first.",
        )

    @property
    def r_hat(self):
        return check_none(
            self.R_hat_, "No recorded Gelman Rubin STATs, please run run_mcmc first."
        )

    def get_posterior_dist(self, max_components: int = 5):
        return fit_posterior(
            self.get_samples(flat=True), max_components=max_components
        )


@dataclass
class OLS_Results:
    PARAMS_: np.ndarray = None
    LogP_: np.ndarray = None


@dataclass
class Data:
    x_data: np.ndarray
    y_data: np.ndarray
    bounds: List[Tuple[float, float]]
    param_names: Optional[List[str]] = None
    ndim: int = 0
    n_nuisance: int = 0
    sa_results: SA_Results = field(default_factory=SA_Results)
    mcmc_results: MCMC_Results = field(default_factory=MCMC_Results)
    ols_results: OLS_Results = field(default_factory=OLS_Results)
    OPTIMAL_LogP_: float = -np.inf
    OPTIMAL_PARAM_: np.ndarray = None
    OPTIMAL_PARAM_STD_: np.ndarray = None
    NUISANCE_: np.ndarray = None
    fit_func: Callable = None

    @property
    def optimal_params(self):
        return check_none(
            self.OPTIMAL_PARAM_,
            "Optimal parameters have not been set, please run optimization first.",
        )

    @property
    def optimal_logp(self):
        return self.OPTIMAL_LogP_

    def get_bic(self):
        return calculate_bic(self.optimal_logp, self.ndim, len(self.y_data))
