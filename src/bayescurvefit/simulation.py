from typing import Callable, List, Tuple

import numpy as np
import pandas as pd


class Simulation:
    def __init__(
        self,
        fit_function: Callable,
        X: np.ndarray,
        sim_params_bounds: List[Tuple[float, float]],
        pdf_function: Callable,
        n_samples: int = 10,
        n_replicates: int = 1,
        param_names: List[str] = None,
        positive_obs: bool = True,
        seed: int = 42,
    ):

        """
        Simulating data with noise sampled and potential outliers for curve fitting from user specificed probability density function and fitting function .

        args:
            fit_function: The fitting function.
            X: The independent variable values.
            sim_params_bounds: The bounds for the simulation parameters.
            tech_error: The standard deviations for the technical noise.
            n_samples: The number of sample sets to simulate.
            n_replicates: The number of replicates for each parameter set.
            pdf_function: the distribution function for sampling.
            pdf_params: The distribution function parameters.
            param_names: The names of the fitting parameters.
            positive_obs: If True, ensures all observations are non-negative.
            seed: The random seed for reproducibility.
        """
        self.fit_function = fit_function
        self.X = X.repeat(n_replicates)
        self.sim_params_bounds = sim_params_bounds
        self.n_samples = n_samples
        self.pdf = pdf_function
        self.seed = seed
        if param_names is None:
            param_names = [f"param_{i+1}" for i in range(len(sim_params_bounds))]
        self.columns = param_names
        self.positive_obs = positive_obs
        self.df_sim_params = None
        self.df_sim_data = None
        np.random.seed(self.seed)
        self.run_simulation()

    def generate_sim_params(self):
        num_params = len(self.sim_params_bounds)
        self.sim_params = np.empty((self.n_samples, num_params))
        for i, (low, high) in enumerate(self.sim_params_bounds):
            self.sim_params[:, i] = np.random.uniform(low, high, self.n_samples)

        self.df_sim_params = pd.DataFrame(
            [list(param) for param in self.sim_params],
            index=[f"m{i+1:03d}" for i in range(len(self.sim_params))],
            columns=self.columns,
        )

    def generate_error(self):
        errors = self.pdf.sample(size=len(self.X) * len(self.sim_params))
        errors_list = errors.reshape(len(self.sim_params), len(self.X))
        index_labels = self.df_sim_params.index
        column_labels = [f"s{i+1:03d}" for i in range(len(self.X))]
        all_errors = np.concatenate(errors_list).reshape(
            len(index_labels), len(column_labels)
        )
        self.sim_errors = pd.DataFrame(
            all_errors, index=index_labels, columns=column_labels
        )

    def create_simulation_data(self):
        data_list = [
            self.fit_function(self.X, *params)
            for _, params in self.df_sim_params.iterrows()
        ]
        self.df_sim_data = pd.DataFrame(
            data_list, index=self.sim_errors.index, columns=self.sim_errors.columns
        )
        self.df_sim_data_w_error = self.df_sim_data + self.sim_errors

        if self.positive_obs:
            self.df_sim_data_w_error = self.df_sim_data_w_error.map(
                lambda x: np.nan if x < 0 else x
            )

    def run_simulation(self):
        self.generate_sim_params()
        self.generate_error()
        self.create_simulation_data()
