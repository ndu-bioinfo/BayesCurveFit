import itertools
import math
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import griddata
from scipy.stats import norm

from .data_classes import Data
from .utils import show_warning, truncated_normal


class BayesAnalysis:
    def __init__(self, data_instance: Data):
        self.data = data_instance
        self.sa_params = (
            self.sa_results.SAMPLES_ if self.sa_results.SAMPLES_ is not None else None
        )
        self.sa_optima = (
            self.sa_results.get_optimal()
            if self.sa_results.SAMPLES_ is not None
            else None
        )
        mcmc_samples = (
            self.mcmc_results.get_samples(flat=True)
            if self.mcmc_results.SAMPLES_ is not None
            else None
        )
        if mcmc_samples is not None:
            k = max(int(np.prod(mcmc_samples.shape) / 5000), 1)
        self.mcmc_params = (
            self.mcmc_results.get_samples(flat=True)[::k, :]
            if self.mcmc_results.SAMPLES_ is not None
            else None
        )
        self.mcmc_logps = (
            self.mcmc_results.logp[::k]
            if self.mcmc_results.SAMPLES_ is not None
            else None
        )

    def __getattr__(self, name):
        return getattr(self.data, name)

    @staticmethod
    def _plot_sa_path(ax, x_param, y_param, sa_color, sa_path_size):
        ax.scatter(
            x=x_param,
            y=y_param,
            s=sa_path_size,
            c=sa_color,
            label="Simulated Annealing",
        )
        for k in range(1, len(x_param)):
            prev_x = x_param[k - 1]
            prev_y = y_param[k - 1]
            curr_x = x_param[k]
            curr_y = y_param[k]
            ax.arrow(
                prev_x,
                prev_y,
                curr_x - prev_x,
                curr_y - prev_y,
                fc=sa_color,
                color=sa_color,
                length_includes_head=True,
            )

    @staticmethod
    def _plot_samplings(
        ax,
        x_param,
        y_param,
        logps,
        sampling_color,
        contour_cmap,
        contour_levels,
        contour_method,
        label,
    ):
        X, Y = np.meshgrid(
            np.linspace(x_param.min(), x_param.max(), 100),
            np.linspace(y_param.min(), y_param.max(), 100),
        )
        Z = griddata((x_param, y_param), logps, (X, Y), method=contour_method)
        ax.scatter(x=x_param, y=y_param, color=sampling_color, s=1, label=label)
        ax.contourf(X, Y, Z, cmap=contour_cmap, alpha=0.3, levels=contour_levels)

    def plot_pairwise_comparison(self, figsize=(8, 6), show_sa_path=True, **kwargs):
        default_kwargs = {
            "sampling_color": "orange",
            "contour_cmap": "Oranges_r",
            "contour_levels": 10,
            "contour_method": "linear",
            "sa_color": "blue",
            "sa_path_size": 5,
            "optima_size": 8,
            "optima_color": "red",
        }
        for key, value in default_kwargs.items():
            kwargs.setdefault(key, value)

        def _make_plots(ax, i, j):
            if self.mcmc_params is not None:
                logps = self.mcmc_logps
                self._plot_samplings(
                    ax=ax,
                    x_param=self.mcmc_params[:, j],
                    y_param=self.mcmc_params[:, i],
                    logps=logps,
                    sampling_color=kwargs["sampling_color"],
                    contour_cmap=kwargs["contour_cmap"],
                    contour_levels=kwargs["contour_levels"],
                    contour_method=kwargs["contour_method"],
                    label="MCMC samples",
                )
            if self.sa_params is not None and show_sa_path:
                sa_optima = self.sa_optima
                self._plot_sa_path(
                    ax,
                    x_param=self.sa_params[:, j],
                    y_param=self.sa_params[:, i],
                    sa_color=kwargs["sa_color"],
                    sa_path_size=kwargs["sa_path_size"],
                )
                ax.plot(
                    sa_optima[j],
                    sa_optima[i],
                    "o",
                    markersize=kwargs["optima_size"],
                    color=kwargs["sa_color"],
                    label="SA optima",
                )
            ax.plot(
                self.OPTIMAL_PARAM_[j],
                self.OPTIMAL_PARAM_[i],
                "+",
                markersize=kwargs["optima_size"],
                label="Global optima",
                color=kwargs["optima_color"],
            )
            ax.set_xlabel(self.param_names[j])
            ax.set_ylabel(self.param_names[i])
            ax.legend()
            ax.set_xlim(self.bounds[j])
            ax.set_ylim(self.bounds[i])

        num_params = self.ndim
        if num_params == 1:
            show_warning(
                "Pairwise comparison is not possible with only 1 parameter. Use plot_param_dist instead."
            )
        if num_params == 2:
            fig, ax = plt.subplots(figsize=figsize)
            _make_plots(ax, i=1, j=0)

        if num_params > 2:
            fig, axes = plt.subplots(num_params - 1, num_params - 1, figsize=figsize)
            for i, j in itertools.combinations(range(num_params), 2):
                ax = axes[i, j - 1]
                ax.clear()
                _make_plots(ax, i=i, j=j)
            for i in range(num_params - 1):
                for j in range(i):
                    fig.delaxes(axes[i, j])

        plt.tight_layout()

    def plot_param_dist(self, figsize=(8, 6), bins=10, hist_color="orange"):
        num_params = self.ndim
        num_cols = min(math.ceil(math.sqrt(num_params)), 4)
        num_rows = math.ceil(num_params / num_cols)
        samples = self.mcmc_params
        _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        if self.ndim > 1:
            axes = axes.flatten()
        for i in range(num_params):
            if self.ndim > 1:
                ax = axes[i]
            else:
                ax = axes
            x_data = np.linspace(*self.bounds[i], 1000)
            # For multivariate GMM, we need to create a 2D array with all parameters
            # and then extract the marginal distribution for parameter i
            if self.data.mcmc_results.multivariate_gmm is not None:
                # Create a grid for all parameters, using optimal values for others
                param_grid = np.tile(self.data.OPTIMAL_PARAM_, (len(x_data), 1))
                param_grid[:, i] = x_data
                y_data = np.exp(
                    self.data.mcmc_results.multivariate_gmm.score_samples(param_grid)
                )
            else:
                # Fallback: use normal distribution approximation
                y_data = norm(
                    loc=self.data.OPTIMAL_PARAM_[i],
                    scale=self.data.OPTIMAL_PARAM_STD_[i],
                ).pdf(x_data)
            sns.histplot(
                samples[:, i],
                bins=bins,
                ax=ax,
                alpha=0.3,
                stat="density",
                color=hist_color,
            )
            valid_indices = y_data > 1e-3
            y_data = y_data[valid_indices]
            x_data = x_data[valid_indices]
            y_data /= np.sum(y_data * (x_data[1] - x_data[0]))
            sns.lineplot(
                x=x_data,
                y=y_data,
                label="GaussianMixture fitted distribution",
                ax=ax,
                color="red",
            )
            sns.lineplot(
                x=x_data,
                y=norm(
                    loc=self.data.OPTIMAL_PARAM_[i],
                    scale=self.data.OPTIMAL_PARAM_STD_[i],
                ).pdf(x_data),
                label="BMA estimated distribution",
                ax=ax,
                color="blue",
                dashes=[3, 1],
            )
            ax.set_title(f"{self.param_names[i]} Density")
            ax.set_xlabel(f"{self.param_names[i]}")
            ax.set_ylabel("Density")
        ax.legend()
        plt.tight_layout()

    def plot_fitted_curve(
        self,
        figsize=(8, 5),
        num_sim_samples=1000,
        show_ols_fit=True,
        title="",
        max_label_length=30,
        show_fitting_params=False,
        truth_params=None,
        show_ci: float = 0.95,
        ax=None,
    ):
        """
        Plot the fitted curve with confidence intervals, observations, and optional OLS and truth fits.

        Args:
            figsize (tuple): Figure size for the plot.
            num_sim_samples (int): Number of samples for simulation.
            show_ols_fit (bool): Whether to show the OLS fit.
            title (str): Title of the plot.
            max_label_length (int): Maximum length of labels in the legend.
            truth_params (list): True parameter values for comparison.
        """
        s_range = np.linspace(min(self.x_data), max(self.x_data), 100)
        label_bayes_params = ", ".join(
            [
                f"{param}={round(value, 2)}"
                for param, value in zip(self.param_names, self.OPTIMAL_PARAM_)
            ]
        )

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        if self.mcmc_results.SAMPLES_ is not None and show_ci > 0:
            sim_parameters = np.zeros((num_sim_samples, self.ndim))
            for i in range(self.ndim):
                loc, scale = (
                    self.data.OPTIMAL_PARAM_[i],
                    self.data.OPTIMAL_PARAM_STD_[i],
                )
                lower, upper = self.bounds[i]
                random_values = truncated_normal(
                    loc=loc,
                    scale=scale,
                    lower=lower,
                    upper=upper,
                    num_sim_samples=num_sim_samples,
                )
                np.random.shuffle(random_values)
                sim_parameters[:, i] = random_values
            sim_ys = np.array(
                [self.fit_func(s_range, *params) for params in sim_parameters]
            )
            y_lower = np.percentile(sim_ys, 100 * ((1 - show_ci) / 2), axis=0)
            y_upper = np.percentile(sim_ys, 100 * (1 - (1 - show_ci) / 2), axis=0)
            ax.fill_between(
                s_range,
                y_lower,
                y_upper,
                color="#add8e6",
                alpha=0.5,
                label=f"{show_ci * 100}% PI",
            )

        # Plot the observations
        sns.scatterplot(x=self.x_data, y=self.y_data, label="Observation", ax=ax)

        # Plot the BayesFitModel curve
        sns.lineplot(
            x=s_range,
            y=self.fit_func(s_range, *self.OPTIMAL_PARAM_),
            color="blue",
            label=f"BayesCurveFit: {label_bayes_params}"
            if show_fitting_params
            else "BayesCurveFit",
            ax=ax,
        )

        # Optionally plot the OLS fit
        if show_ols_fit and self.ols_results.PARAMS_ is not None:
            fit_params = self.ols_results.PARAMS_
            label_fit_params = ", ".join(
                [
                    f"{param}={round(value, 2)}"
                    for param, value in zip(self.param_names[: self.ndim], fit_params)
                ]
            )
            sns.lineplot(
                x=s_range,
                y=self.fit_func(s_range, *fit_params),
                color="red",
                dashes=[5, 1],
                label=f"OLS: {label_fit_params}" if show_fitting_params else "OLS",
                ax=ax,
            )

        # Optionally plot the truth parameters
        if truth_params is not None:
            label_true_params = ", ".join(
                [
                    f"{param}={round(value, 2)}"
                    for param, value in zip(self.param_names[: self.ndim], truth_params)
                ]
            )
            sns.lineplot(
                x=s_range,
                y=self.fit_func(s_range, *truth_params),
                color="green",
                dashes=[3, 1],
                label=f"Truth: {label_true_params}" if show_fitting_params else "Truth",
                ax=ax,
            )

        # Wrap long labels and set the legend
        handles, labels = ax.get_legend_handles_labels()
        wrapped_labels = [textwrap.fill(label, max_label_length) for label in labels]
        ax.legend(handles, wrapped_labels)
        ax.set_title(title)
        if ax is None:
            return ax
        else:
            return None
