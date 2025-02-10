import itertools
from typing import Callable, List, Optional, Tuple, Union
import warnings
import emcee
import numpy as np
from scipy.optimize import dual_annealing
from scipy.stats.qmc import LatinHypercube

from .data_classes import Data
from .utils import (
    calc_bma,
    calculate_effective_size,
    fit_prosterior,
    gelman_rubin,
    ols_fitting,
    show_warning,
)

warnings.simplefilter("ignore", RuntimeWarning)


class SamplesCallback:
    """
    Custom callback function to allow early termination once delta threshold is reached
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html
    A callback function with signature callback(x, f, context), which will be called for all minima found. x and f are the coordinates and function value of the latest minimum found, and context has value in [0, 1, 2], with the following meaning:
        0: minimum detected in the annealing process.
        1: detection occurred in the local search process.
        2: detection done in the dual annealing process.
        If the callback implementation returns True, the algorithm will stop.

    """

    def __init__(self, delta_threshold):
        self.delta_threshold = delta_threshold
        self.reset()

    def reset(self):
        self.samples = []
        self.neg_logp = []
        self.previous_min = None
        self.current_min = None
        self.success = False

    def __call__(self, x, f, context):
        self.samples.append(list(x.copy()))
        self.neg_logp.append(f)
        self.previous_min = self.current_min
        self.current_min = f
        if self.previous_min is None:
            return False
        if abs(self.current_min - self.previous_min) < self.delta_threshold:
            self.success = True
            return True
        return False


class BayesFit:
    def __init__(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        fit_func: Callable,
        pdf_function: Callable,
        bounds: List[Tuple[float, float]],
        param_names: Optional[List[str]] = None,
        n_nuisance: int = 0,
        seed: int = 42,
    ):
        """Initializes the BayesFit object.

        Args:
            x_data: The independent variable data.
            y_data: The dependent variable data.
            fit_func: The function to fit the data.
            pdf_function: The probability density function.
            bounds: A list of tuples specifying the lower and upper bounds for each parameter, including nuisance parameters. The bounds for each parameter must follow the order of the function's parameters, with the bounds for nuisance parameters following those of the function parameters.
            param_names: A list of parameter names. If not specified, the parameters will be named param_1, param_2, and so on.
            n_nuisance: number of nuisance parameters.
            seed: Random seed for reproducibility. Defaults to 42.
        """
        self.fit_func = fit_func
        self.pdf_function = pdf_function
        if param_names is None:
            param_names = [f"param_{x+1}" for x in range(len(bounds))]
        assert len(param_names) == len(
            bounds
        ), "Ensure param_names have same number of elements as bounds"
        self.seed = seed
        self.load_data(x_data, y_data, bounds, param_names, n_nuisance, fit_func)

    def load_data(self, x_data, y_data, bounds, param_names, n_nuisance, fit_func):
        ndim = len(bounds) - n_nuisance
        self.data = Data(
            x_data=x_data,
            y_data=y_data,
            bounds=bounds,
            param_names=param_names,
            ndim=ndim,
            n_nuisance=n_nuisance,
            fit_func=fit_func,
        )
        self.update_nuisance = False

    def error_function(self, params):
        """Calculate the error between the observed data and the model
        predictions."""
        return self.data.y_data - self.fit_func(self.data.x_data, *params)

    def log_likelihood(self, params):
        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            errors = self.error_function(params[: self.data.ndim])
            if self.data.n_nuisance > 0 and not self.update_nuisance:
                assert (
                    self.data.NUISANCE_ is not None
                ), "To use nuisance parameters you need to first finish appoximation!"
                return np.log(
                    self.pdf_function(pdf_params=self.data.NUISANCE_, data=errors)
                )

            elif self.data.n_nuisance > 0 and self.update_nuisance:
                return np.log(
                    self.pdf_function(
                        pdf_params=params[-self.data.n_nuisance :], data=errors
                    )
                )

            else:
                return np.log(self.pdf_function(data=errors))

    def sum_log_likelihood(self, params):
        return np.sum(self.log_likelihood(params))

    def log_probability_mcmc(self, params: np.ndarray) -> np.ndarray:
        """Calculate the sum of log probabilities for the emcee mover.

        Args:
            params: Array of parameter sets.

        Returns:
            Array of log probabilities for each parameter set.
        """
        invalid_mask = np.any(params < self.lower_bounds, axis=1) | np.any(
            params > self.upper_bounds, axis=1
        )
        log_probs = np.full(params.shape[0], -np.inf)
        valid_params = params[~invalid_mask]

        if valid_params.size > 0:
            log_probs[~invalid_mask] = np.array(
                [
                    self.sum_log_likelihood(valid_params[i])
                    for i in range(valid_params.shape[0])
                ]
            )

            if (
                self.data.OPTIMAL_PARAM_ is not None and self.local_sigma > 0.0
            ):  # Add panelty to walkers when moving too far from local peak
                distances = np.linalg.norm(valid_params - self.init_pos, axis=1)
                penalties = -0.5 * (distances / self.local_sigma) ** 2
                log_probs[~invalid_mask] += penalties

        return log_probs

    def generate_random_init_params(self, samples: int, **kwargs) -> np.ndarray:
        """Generate random initial parameter sets where walkers start.

        Args:
            samples (int): Number of parameter sets to generate.
            **kwargs: Additional keyword arguments for customization.

        Returns:
            np.ndarray: An array of randomly generated starting positions.
        """
        kwargs.setdefault("seed", self.seed)
        kwargs.setdefault("optimization", "lloyd")
        if self.update_nuisance:
            lhs_engine = LatinHypercube(d=len(self.data.bounds), **kwargs)
            lhs_samples = lhs_engine.random(n=samples)
        else:
            if self.data.ndim >= 2:
                lhs_engine = LatinHypercube(
                    d=len(self.data.bounds[: self.data.ndim]), **kwargs
                )
                lhs_samples = lhs_engine.random(n=samples)
            else:
                lhs_samples = np.random.rand(samples, 1)

        return self.lower_bounds + lhs_samples * (self.upper_bounds - self.lower_bounds)

    def run_ols(self, run_correction=False, verbose: int = 0, **kwargs):
        """Run OLS fitting.

        Args:
            run_correction: If Ture, update global optima if OLS yields higher sum log likelihoods.
            verbose: 0 for no message, 1 for critical info and 2 for al info.
        """
        try:
            fit_params = ols_fitting(
                x_data=np.array(self.data.x_data),
                y_data=np.array(self.data.y_data),
                fit_func=self.fit_func,
                bounds=self.data.bounds[: self.data.ndim],
                **kwargs,
            )[0]
        except (ValueError, RuntimeError, OverflowError) as e:
            show_warning(
                f"First fitting attempt failed: {e}", verbose_level=2, verbose=verbose
            )
            try:
                fit_params = ols_fitting(
                    x_data=np.array(self.data.x_data),
                    y_data=np.array(self.data.y_data),
                    fit_func=self.fit_func,
                    bounds=self.data.bounds[: self.data.ndim],
                    init_guess=self.data.OPTIMAL_PARAM_,
                    **kwargs,
                )[0]
            except (ValueError, RuntimeError, OverflowError) as e:
                show_warning(
                    f"Second fitting attempt failed: {e}",
                    verbose_level=2,
                    verbose=verbose,
                )
                fit_params = np.array([np.nan for i in range(self.data.ndim)])
        self.data.ols_results.PARAMS_ = fit_params
        if (
            run_correction
        ):  # After SA, check if OLS with SA found optimal parameters as init might generate lower sum_log, if so, update the optima
            self.data.ols_results.LogP_ = self.sum_log_likelihood(fit_params)
            if self.data.OPTIMAL_LogP_ < self.data.ols_results.LogP_:
                self.data.OPTIMAL_LogP_ = self.data.ols_results.LogP_
                self.data.OPTIMAL_PARAM_ = fit_params
                show_warning(
                    f"Params updated with OLS fitting: {self.data.OPTIMAL_PARAM_}",
                    verbose_level=2,
                    verbose=verbose,
                )
            self.loocv_adjustment(verbose=verbose)

    def loocv_adjustment(self, verbose: int = 0, **kwargs):
        """
        Perform leave one out cross validation for adjusting log P values when OLS correction is used
        """
        x_data_train = self.data.x_data
        y_data_train = self.data.y_data
        loo_log_likelihoods = []

        for i in range(len(x_data_train)):
            x_train = np.delete(x_data_train, i)
            y_train = np.delete(y_data_train, i)
            try:
                fit_params = ols_fitting(
                    x_data=x_train,
                    y_data=y_train,
                    fit_func=self.fit_func,
                    bounds=self.data.bounds[: self.data.ndim],
                    init_guess=self.data.OPTIMAL_PARAM_,
                    **kwargs,
                )[0]

                loo_log_likelihood = self.sum_log_likelihood(fit_params)
                loo_log_likelihoods.append(loo_log_likelihood)
            except (ValueError, RuntimeError, OverflowError):
                pass

        if len(loo_log_likelihoods) > 0:
            self.data.OPTIMAL_LogP_ = np.log(np.mean(np.exp(loo_log_likelihoods)))
        else:
            show_warning(
                "No loocv log P adjustment due to failed OLS convergence",
                verbose_level=2,
                verbose=verbose,
            )

    def run_simulated_annealing(
        self,
        ensemble_size: int = None,
        maxiter: int = 5000,
        delta_threshold: float = 1e-3,
        minimizer_kwargs: Optional[dict] = None,
        solve_nuisance: bool = True,
        verbose: int = 0,
    ) -> None:
        """Run simulated annealing to optimize parameters and update the
        optimal parameters and LogP if a better solution is found.

        Args:
            ensemble_size: The number of initial parameter sets to generate and use for annealing, if None then 5 ensembles per dimension. .
            maxiter: The maximum number of iterations for the annealing process.
            delta_threshold: The threshold for determining convergence based on changes in the objective function.
            minimizer_kwargs: Additional keyword arguments to pass to the minimizer.
            solve_nuisance: If True, solves for nuisance parameters.
            verbose: 0 for no message, 1 for critical info and 2 for al info.
        """
        if minimizer_kwargs is None:
            minimizer_kwargs = {"method": "L-BFGS-B"}

        self.update_nuisance = solve_nuisance
        if solve_nuisance:
            assert self.data.n_nuisance > 0, "No nuisance parameters specified"
            self.lower_bounds, self.upper_bounds = np.transpose(self.data.bounds)
            sa_dim = len(self.data.bounds)
        else:
            if self.data.n_nuisance > 0:
                assert (
                    self.data.NUISANCE_ is not None
                ), "Nuisance parameters specified but not solved "
            self.lower_bounds, self.upper_bounds = np.transpose(
                self.data.bounds[: self.data.ndim]
            )
            sa_dim = self.data.ndim

        if ensemble_size is None:
            ensemble_size = int(sa_dim * 5)
        samples_callback = SamplesCallback(delta_threshold=delta_threshold)
        initial_positions = self.generate_random_init_params(ensemble_size)

        for i in range(ensemble_size):
            samples_callback.reset()
            result = dual_annealing(
                lambda params: -1 * self.sum_log_likelihood(params),
                self.data.bounds[:sa_dim],
                maxiter=maxiter,
                no_local_search=True,
                minimizer_kwargs=minimizer_kwargs,
                callback=samples_callback,
                x0=initial_positions[i],
                seed=self.seed,
            )
            result.success = samples_callback.success
            if result.success:
                result.message = ["Optimal condition reached!"]

            if self.data.OPTIMAL_LogP_ < -1 * result.fun:
                self.data.OPTIMAL_LogP_ = -1 * result.fun
                self.data.OPTIMAL_PARAM_ = result.x[: self.data.ndim]
                if self.update_nuisance:
                    assert self.data.n_nuisance > 0, "No nuisance parameters specified"
                    self.data.NUISANCE_ = result.x[-self.data.n_nuisance :]
                self.data.sa_results.SAMPLES_ = np.array(samples_callback.samples)
                self.data.sa_results.LogP_ = -1 * np.array(samples_callback.neg_logp)
                self.data.sa_results.RESULTS_ = result
                show_warning(
                    f"Params updated to {self.data.OPTIMAL_PARAM_}",
                    verbose_level=2,
                    verbose=verbose,
                )
        self.update_nuisance = False  # Ensure reset global default value

    def run_mcmc(
        self,
        num_walkers: int = None,
        chain_length: int = 1000,
        burn_in_length: int = 10,
        starting_pos: str = "optimal",
        moves: emcee.moves.Move = None,
        expected_local_sigma: float = 3.0,
        maximal_rhat: float = 1.1,
        minimal_sample_sizes: int = 100,
        converge_attempts: int = 10,
        solve_nuisance: bool = False,
        max_components: int = 5,
        bw_method: Union[str, float] = "scott",
        verbose: int = 0,
    ) -> None:
        """Run Markov Chain Monte Carlo (MCMC) sampling to optimize parameters.
        If the auto_converge is enabled, MCMC will be executed in loops until
        all desired convergence criteria are met. Note that the automated
        algorithm is optimized for detecting a local peak after the global
        optimal approximation has been completed; advanced users running
        standalone MCMC with the automated algorithm should proceed with
        caution.

        Args:
            num_walkers: Number of walkers for running MCMC, if None then 5 chains per dimension.
            chain_length: Number of steps for the main MCMC chain.
            burn_in_length: Number of steps for the burn-in period.
            starting_pos: Starting position for walkers, either "optimal" or "random".
            moves: emcee moves function, see https://emcee.readthedocs.io/en/stable/user/sampler/.
            expected_local_sigma: Expected sigma when searching a local peak.
            maximal_rhat: Maximum allowable Gelman Rubin test R_hat value for convergence.
            minimal_sample_sizes: Minimum effective sample size required for convergence. According to Bayesian Data Analysis 3rd edition, 100 draws are enough for reasonable prosterior summaries. Nontheless, more draws can provide better accuracy as needed.
            converge_attempts: Maximum number of attempts to convergence.
            solve_nuisance: If True, solves for nuisance parameters.
            max_components: Maximal number of components allowed for fitting prosterior distribution.
            verbose: 0 for no message, 1 for critical info and 2 for al info.

        Raises:
            ValueError: If auto converge is set to True and not enough chains converge after multiple attempts.
        """
        self.update_nuisance = solve_nuisance
        np.random.seed(self.seed)

        if solve_nuisance:
            assert self.data.n_nuisance > 0, "No nuisance parameters specified"
            self.lower_bounds, self.upper_bounds = np.transpose(self.data.bounds)
            mcmc_dim = len(self.data.bounds)
        else:
            if self.data.n_nuisance > 0:
                assert (
                    self.data.NUISANCE_ is not None
                ), "Nuisance parameters specified but not solved "
            self.lower_bounds, self.upper_bounds = np.transpose(
                self.data.bounds[: self.data.ndim]
            )
            mcmc_dim = self.data.ndim

        if num_walkers is None:
            num_walkers = int(mcmc_dim * 5)

        if starting_pos == "optimal":
            self.local_sigma = expected_local_sigma
            if solve_nuisance:
                self.init_pos = np.concatenate(
                    [self.data.OPTIMAL_PARAM_, self.data.NUISANCE_]
                )
            else:
                self.init_pos = self.data.OPTIMAL_PARAM_
            initial_positions = np.clip(
                np.random.normal(
                    loc=np.tile(self.init_pos, (num_walkers, 1)),
                    scale=abs(np.tile(self.init_pos, (num_walkers, 1))) * 1e-2,
                ),
                self.lower_bounds,
                self.upper_bounds,
            )
        elif starting_pos == "random":
            initial_positions = self.generate_random_init_params(num_walkers)
            self.local_sigma = 0.0  # Ensure no panelty in loglikelihood estimation
        else:
            raise ValueError("Invalid starting_pos value. Use 'optimal' or 'random'.")

        sampler = emcee.EnsembleSampler(
            num_walkers,
            mcmc_dim,
            self.log_probability_mcmc,
            moves=moves,
            vectorize=True
        )

        sampler.run_mcmc(initial_positions, burn_in_length, progress=(verbose > 0))

        r_hat = np.inf
        attempt = 0
        effective_sizes = 0
        while attempt < converge_attempts and (
            np.any(r_hat > maximal_rhat)
            or np.any(effective_sizes < minimal_sample_sizes)
        ):
            last_positions = sampler.get_last_sample().coords
            if attempt == 0:
                sampler.reset()  # Do not save burn in samples
            sampler.run_mcmc(last_positions, chain_length, progress=(verbose > 0))
            chains = sampler.get_chain(flat=False)
            r_hat = gelman_rubin(chains)
            if all(r_hat < maximal_rhat):
                effective_sizes = calculate_effective_size(chains)
                if not np.all(effective_sizes > minimal_sample_sizes):
                    show_warning(
                        f"Insurfficient sample sizes {effective_sizes}; continue to run mcmc.",
                        verbose_level=1,
                        verbose=verbose,
                    )
            else:
                show_warning(
                    f"Failed Gelman Rubin test {r_hat}; continue to run mcmc.",
                    verbose_level=1,
                    verbose=verbose,
                )
            attempt += 1

        if attempt < converge_attempts:
            self.data.mcmc_results.CONVERGE_ = True

        self.data.mcmc_results.SAMPLES_ = chains
        self.data.mcmc_results.EFFECTIVE_SIZES_ = calculate_effective_size(
            self.data.mcmc_results.SAMPLES_
        )
        self.data.mcmc_results.R_hat_ = gelman_rubin(self.data.mcmc_results.SAMPLES_)
        self.data.mcmc_results.LogP_ = np.array(
            list(
                map(
                    self.sum_log_likelihood,
                    self.data.mcmc_results.get_samples(flat=True),
                )
            )
        )

        self.data.mcmc_results.best_gmms = [
            fit_prosterior(
                self.data.mcmc_results.get_samples(flat=True)[:, i],
                max_components=max_components,
                bw_method=bw_method,
            )
            for i in range(self.data.ndim)
        ]
        optima = np.array([calc_bma(gmm) for gmm in self.data.mcmc_results.best_gmms])
        self.data.OPTIMAL_PARAM_ = optima[:, 0]
        self.data.OPTIMAL_PARAM_STD_ = optima[:, 1]

        opt_params = list(
            itertools.product(
                *[gmm.means_.flatten() for gmm in self.data.mcmc_results.best_gmms]
            )
        )
        opt_weights = np.prod(
            np.array(
                list(
                    itertools.product(
                        *[gmm.weights_ for gmm in self.data.mcmc_results.best_gmms]
                    )
                )
            ),
            axis=1,
        )
        weighted_probs = np.array(
            [
                weight * np.exp(self.sum_log_likelihood(param))
                for weight, param in zip(opt_weights, opt_params)
            ]
        )
        self.data.OPTIMAL_LogP_ = np.log(np.sum(weighted_probs))

        self.update_nuisance = False  # Ensure to reset global default value
