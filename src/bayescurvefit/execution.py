from typing import Callable, List, Optional, Tuple, Union

import emcee
import numpy as np
import pandas as pd
from emcee.moves import DESnookerMove, StretchMove

from .analysis import BayesAnalysis
from .bayesfit import BayesFit
from .distribution import SimplePDF
from .utils import calculate_bic, compute_pep

PDF_FUNCTION = SimplePDF(fixed_pdf_params=[1, 0.05]).mixture_pdf


class BayesFitModel:
    def __init__(
        self,
        x_data: np.array,
        y_data: np.array,
        fit_function: Callable,
        params_range: List[Tuple[float, float]],
        param_names: Optional[List[str]] = None,
        run_mcmc: bool = False,
        mcmc_sampler: Optional[emcee.moves.Move] = None,
        bw_method: Union[str, float] = 1e-3,
        verbose: int = 0,
    ):
        """
        General Bayesian workflow that is useful for most biological data fitting applicaitons.

        Args:
            x_data: The independent variable data.
            y_data: The dependent variable data.
            fit_function: The function to fit the data.
            params_range: The ranges for the parameters.
            param_names: The names of the parameters.
            run_mcmc: Whether to compute parameter uncertainties by running mcmc.
            mcmc_sampler: The MCMC sampler to use.
            verbose: Verbosity level, 0 for silence, 1 for critical, and 2 for all.
        """
        self.x_data = x_data[~np.isnan(y_data)]
        self.y_data = y_data[~np.isnan(y_data)]
        self.fit_function = fit_function
        self.params_range = params_range
        self.param_names = param_names
        self.run_mcmc = run_mcmc
        self.bw_method = bw_method
        self.verbose = verbose
        self.null_function = lambda x, c: np.full_like(x, c)
        self.nuisance_names = [
            "est_std"
        ]  # estimated standard deviation of the residual

        if (
            np.std(self.y_data) == 0.0
        ):  # For exeteme cases when all ys equal; introduce random 1% error so simulated annealing has space to work
            self.y_data = self.y_data + np.random.uniform(
                low=self.y_data.mean() * 0.99,
                high=self.y_data * 1.01,
                size=self.y_data.shape,
            )

        if (
            mcmc_sampler is None
        ):  # Best estimated sampler based on dimension if user did not specify
            if len(params_range) <= 2:
                self.mcmc_sampler = StretchMove()
            else:
                self.mcmc_sampler = DESnookerMove()
        else:
            self.mcmc_sampler = mcmc_sampler

        if param_names is None:
            self.all_param_names = [
                f"params_{x+1}" for x in range(len(params_range))
            ] + self.nuisance_names
        else:
            self.all_param_names = param_names + self.nuisance_names
        self._run()

    def _run(self):
        """
        Perform Bayesian fitting on the provided data.
        """
        self.bayes_fit_pipe = BayesFit(
            fit_func=self.fit_function,
            pdf_function=PDF_FUNCTION,
            x_data=self.x_data,
            y_data=self.y_data,
            bounds=self.params_range + [(1e-6, np.std(self.y_data))],
            param_names=self.all_param_names,
            n_nuisance=1,
        )
        self.bayes_fit_pipe.run_simulated_annealing(verbose=self.verbose)
        # correction for error estimation; when sample size is small the estimated SE might be underestimated
        y_bayes_fit = self.fit_function(
            self.x_data, *self.bayes_fit_pipe.data.OPTIMAL_PARAM_
        )
        bayes_fit_errors = self.y_data - y_bayes_fit
        self.bayes_fit_pipe.data.NUISANCE_ = np.array(
            [
                np.mean(
                    [
                        self.bayes_fit_pipe.data.NUISANCE_[0],
                        np.sqrt(np.mean(bayes_fit_errors**2)),
                    ]
                )
            ]
        )
        self.bayes_fit_pipe.run_ols(
            run_correction=(not self.run_mcmc), verbose=self.verbose
        )  # Use OLS for correction if MCMC is not desired
        # Rerun SA to ensure best performance
        self.bayes_fit_pipe.run_simulated_annealing(
            solve_nuisance=False, verbose=self.verbose
        )

        if self.run_mcmc:
            self.bayes_fit_pipe.run_mcmc(
                moves=self.mcmc_sampler, bw_method=self.bw_method, verbose=self.verbose
            )

        # Null model consider y is independent of x, and its error = stdev of y
        self.bayes_fit_null = BayesFit(
            fit_func=self.null_function,
            pdf_function=PDF_FUNCTION,
            x_data=self.x_data,
            y_data=self.y_data,
            bounds=[
                (min(self.y_data) * 0.99, max(self.y_data) * 1.01),
                (np.std(self.y_data) * 0.99, np.std(self.y_data) * 1.01),
            ],
            n_nuisance=1,
        )
        self.bayes_fit_null.run_simulated_annealing(verbose=self.verbose)

    def get_result(self):
        """
        Return the fitting results
        """
        bic0 = calculate_bic(
            self.bayes_fit_null.data.OPTIMAL_LogP_,
            self.bayes_fit_null.data.ndim,
            len(self.y_data),
        )
        bic1 = calculate_bic(
            self.bayes_fit_pipe.data.OPTIMAL_LogP_,
            self.bayes_fit_pipe.data.ndim,
            len(self.y_data),
        )
        pep = compute_pep(bic0, bic1)

        rmse = np.sqrt(
            np.mean(
                (
                    self.bayes_fit_pipe.data.y_data
                    - self.bayes_fit_pipe.fit_func(
                        self.bayes_fit_pipe.data.x_data,
                        *self.bayes_fit_pipe.data.OPTIMAL_PARAM_,
                    )
                )
                ** 2
            )
        )
        null_mean = self.bayes_fit_null.data.OPTIMAL_PARAM_[0]

        if self.run_mcmc:
            output_values = (
                list(self.bayes_fit_pipe.data.OPTIMAL_PARAM_)
                + list(self.bayes_fit_pipe.data.OPTIMAL_PARAM_STD_)
                + list(self.bayes_fit_pipe.data.NUISANCE_)
                + [null_mean,
                    rmse,
                    pep,
                    not self.bayes_fit_pipe.data.mcmc_results.CONVERGE_ and pep > 0.01,
                ]
            )
            output_names = (
                [f"fit_{v}" for v in self.param_names]
                + [f"std_{v}" for v in self.param_names]
                + self.nuisance_names
                + ["null_mean", "rmse", "pep", "convergence_warning"]
            )
        else:
            output_values = (
                list(self.bayes_fit_pipe.data.OPTIMAL_PARAM_)
                + list(self.bayes_fit_pipe.data.NUISANCE_)
                + [null_mean, rmse, pep]
            )
            output_names = (
                [f"fit_{v}" for v in self.param_names]
                + self.nuisance_names
                + ["null_mean", "rmse", "pep"]
            )
        return pd.Series(output_values, index=output_names)

    @property
    def analysis(self):
        """
        run_ols: whether to run OLS for comparison in analysis
        """
        return BayesAnalysis(self.bayes_fit_pipe.data)
