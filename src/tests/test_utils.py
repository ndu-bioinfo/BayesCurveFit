import os
import pickle
import unittest

import numpy as np

from bayescurvefit.utils import (
    calc_bma,
    calculate_bic,
    calculate_effective_size,
    compute_pep,
    fit_posterior,
    gelman_rubin,
    geweke_diag,
    ols_fitting,
    split_chains,
    truncated_normal,
    variogram,
)


def load_output(filename):
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "files/utils", filename)
    with open(file_path, "rb") as f:
        return pickle.load(f)


class TestUtilsFromFiles(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.x_data = np.linspace(0, 10, 100)
        self.y_data = 3 * np.sin(self.x_data) + np.random.normal(0, 0.5, 100)
        self.mcmc_chain = load_output("mcmc_chains.pkl")
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
        split_chains_result = split_chains(self.mcmc_chain)
        saved_split_chains = load_output("split_chains_output.pkl")
        np.testing.assert_array_equal(split_chains_result, saved_split_chains)

    def test_gelman_rubin(self):
        expected_r_hat = np.array([1.03122, 1.04002])
        r_hat = gelman_rubin(self.mcmc_chain)
        np.testing.assert_allclose(r_hat, expected_r_hat, rtol=0.1)

    def test_variogram(self):
        expected_variogram = np.array([0.01436, 0.33768])
        variogram_result = variogram(self.mcmc_chain, 100)
        np.testing.assert_allclose(variogram_result, expected_variogram, rtol=0.1)

    def test_geweke_diag(self):
        expected_geweke_scores = np.array(
            [
                [-0.27642318, -1.56015775],
                [1.57414811, 0.45452033],
                [-5.85555905, -16.41391957],
                [-2.29344027, 4.84707257],
                [9.38815234, 16.44230356],
                [-2.77470165, -7.50552885],
                [5.92067096, 6.81474101],
                [3.03861873, 5.96154778],
                [2.96977574, 2.8423133],
                [-8.48320535, 2.98517782],
            ]
        )
        geweke_scores = geweke_diag(self.mcmc_chain)
        np.testing.assert_allclose(geweke_scores, expected_geweke_scores, rtol=0.1)

    def test_calculate_effective_size(self):
        expected_ess = np.array([309.88196, 270.92933])
        ess = calculate_effective_size(self.mcmc_chain)
        np.testing.assert_allclose(ess, expected_ess, rtol=0.1)

    def test_calculate_bic(self):
        log_likelihood = -1.5
        num_params = 4
        num_data_points = 10
        bic = calculate_bic(log_likelihood, num_params, num_data_points)
        self.assertAlmostEqual(bic, 12.21034, places=5)

    def test_fit_posterior(self):
        gmm = fit_posterior(self.gamma_samples.T, max_components=5)
        # Test that the GMM has the expected structure and properties
        self.assertEqual(gmm.means_.shape[1], 1)  # Should have 1 parameter
        self.assertGreaterEqual(
            gmm.means_.shape[0], 1
        )  # Should have at least 1 component
        self.assertLessEqual(gmm.means_.shape[0], 5)  # Should have at most 5 components
        self.assertTrue(
            np.allclose(np.sum(gmm.weights_), 1.0, rtol=1e-10)
        )  # Weights should sum to 1
        self.assertTrue(np.all(gmm.weights_ >= 0))  # All weights should be non-negative

    def test_calc_bma(self):
        gmm = fit_posterior(self.gamma_samples.T, max_components=5)
        bma_result = calc_bma(gmm)
        bma_mean = bma_result[0]
        bma_std = bma_result[1]
        expected_bma_mean = 4.052290
        expected_bma_std = 2.987765
        self.assertAlmostEqual(bma_mean[0], expected_bma_mean, places=5)
        self.assertAlmostEqual(bma_std[0], expected_bma_std, places=5)

    def test_compute_pep(self):
        bic0 = 1.0
        bic1 = 1.5
        pep = compute_pep(bic0, bic1)
        self.assertAlmostEqual(pep, 0.562176, places=5)

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
