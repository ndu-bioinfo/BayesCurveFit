import os
import unittest
import numpy as np
import pickle

from biobayesfit.utils import (
    ols_fitting,
    split_chains,
    gelman_rubin,
    variogram,
    geweke_diag,
    calculate_effective_size,
    calculate_bic,
    fit_prosterior,
    calc_bma,
    compute_fdr,
    truncated_normal,
)


def load_output(filename):
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "files", filename)
    with open(file_path, "rb") as f:
        return pickle.load(f)


def load_text(filename):
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "files", filename)
    with open(file_path, "r") as f:
        return float(f.read())


class TestUtilsFromFiles(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.x_data = np.linspace(0, 10, 100)
        self.y_data = 3 * np.sin(self.x_data) + np.random.normal(0, 0.5, 100)
        self.mcmc_chain = np.random.randn(1000, 5)
        self.gamma_samples = np.random.gamma(2, 2, size=(100, 1))

    def test_ols_fitting(self):
        def fit_func(x, a, b):
            return a * np.sin(x) + b

        bounds = [(0, 10), (0, 10)]
        params, y_preds, errors = ols_fitting(
            self.x_data, self.y_data, fit_func, bounds
        )
        saved_output = load_output("ols_fitting_output.pkl")
        saved_params, saved_y_preds, saved_errors = (
            saved_output["params"],
            saved_output["y_preds"],
            saved_output["errors"],
        )
        np.testing.assert_allclose(params, saved_params, rtol=0.1)
        np.testing.assert_allclose(y_preds, saved_y_preds, rtol=0.1)
        np.testing.assert_allclose(errors, saved_errors, rtol=0.1)

    def test_split_chains(self):
        chains = self.mcmc_chain[:, np.newaxis, :]
        split_chains_result = split_chains(chains)
        saved_split_chains = load_output("split_chains_output.pkl")
        np.testing.assert_array_equal(split_chains_result, saved_split_chains)

    def test_gelman_rubin(self):
        chains = self.mcmc_chain[:, np.newaxis, :]
        r_hat = gelman_rubin(chains)
        saved_r_hat = load_output("gelman_rubin_output.pkl")
        np.testing.assert_allclose(r_hat, saved_r_hat, rtol=0.1)

    def test_variogram(self):
        chains = self.mcmc_chain[:, np.newaxis, :]
        variogram_result = variogram(chains, 1)
        saved_variogram_result = load_output("variogram_output.pkl")
        np.testing.assert_allclose(variogram_result, saved_variogram_result, rtol=0.1)

    def test_geweke_diag(self):
        geweke_scores = geweke_diag(self.mcmc_chain[:, 0])
        saved_geweke_scores = load_output("geweke_diag_output.pkl")
        np.testing.assert_allclose(geweke_scores, saved_geweke_scores, rtol=0.1)

    def test_calculate_effective_size(self):
        chains = self.mcmc_chain[:, np.newaxis, :]
        ess = calculate_effective_size(chains)
        saved_ess = load_output("effective_size_output.pkl")
        np.testing.assert_allclose(ess, saved_ess, rtol=0.1)

    def test_calculate_bic(self):
        log_likelihood = -500
        num_params = 5
        num_data_points = 100
        bic = calculate_bic(log_likelihood, num_params, num_data_points)
        saved_bic = load_text("bic_output.txt")
        self.assertAlmostEqual(bic, saved_bic, places=5)

    def test_fit_prosterior(self):
        gmm = fit_prosterior(self.gamma_samples.flatten(), max_components=5)
        saved_gmm = load_output("fit_prosterior_output.pkl")
        np.testing.assert_allclose(gmm.means_, saved_gmm.means_, rtol=0.1)
        np.testing.assert_allclose(gmm.covariances_, saved_gmm.covariances_, rtol=0.1)
        np.testing.assert_allclose(gmm.weights_, saved_gmm.weights_, rtol=0.1)

    def test_calc_bma(self):
        gmm = fit_prosterior(self.gamma_samples.flatten(), max_components=5)
        bma_mean, bma_std = calc_bma(gmm)
        saved_bma = load_output("calc_bma_output.pkl")
        saved_bma_mean, saved_bma_std = saved_bma["bma_mean"], saved_bma["bma_std"]
        self.assertAlmostEqual(bma_mean, saved_bma_mean, places=5)
        self.assertAlmostEqual(bma_std, saved_bma_std, places=5)

    def test_compute_fdr(self):
        bic0 = 100
        bic1 = 150
        fdr = compute_fdr(bic0, bic1)
        saved_fdr = load_text("fdr_output.txt")
        self.assertAlmostEqual(fdr, saved_fdr, places=5)

    def test_truncated_normal(self):
        loc = 0
        scale = 1
        lower = -2
        upper = 2
        num_samples = 1000
        samples = truncated_normal(loc, scale, lower, upper, num_samples)
        saved_samples = load_output("truncated_normal_output.pkl")
        np.testing.assert_allclose(np.mean(samples), np.mean(saved_samples), rtol=0.1)
        np.testing.assert_allclose(np.std(samples), np.std(saved_samples), rtol=0.1)


if __name__ == "__main__":
    unittest.main()
