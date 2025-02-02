import unittest
import numpy as np
from bayescurvefit.data_classes import SA_Results, MCMC_Results, Data


class TestDataClasses(unittest.TestCase):
    def setUp(self):
        self.sa_samples = {0: [0.1, 0.2], 1: [0.3, 0.4]}
        self.sa_logp = {0: -1.0, 1: -0.5}
        self.mcmc_samples = np.random.rand(10, 2, 2)
        self.mcmc_logp = np.random.rand(10)
        self.mcmc_rhat = np.array([1.1, 1.05])
        self.mcmc_effective_sizes = np.array([100, 120])

    def test_sa_results(self):
        sa_results = SA_Results(SAMPLES_=self.sa_samples, LogP_=self.sa_logp)
        self.assertEqual(sa_results.samples, self.sa_samples)
        self.assertEqual(sa_results.logp, self.sa_logp)
        self.assertEqual(sa_results.get_optimal(), self.sa_samples[1])

    def test_mcmc_results(self):
        mcmc_results = MCMC_Results(
            SAMPLES_=self.mcmc_samples,
            LogP_=self.mcmc_logp,
            R_hat_=self.mcmc_rhat,
            EFFECTIVE_SIZES_=self.mcmc_effective_sizes,
        )
        self.assertTrue(
            np.array_equal(mcmc_results.get_samples(flat=False), self.mcmc_samples)
        )
        self.assertTrue(
            np.array_equal(
                mcmc_results.get_samples(flat=True),
                self.mcmc_samples.reshape(-1, self.mcmc_samples.shape[-1]),
            )
        )
        self.assertTrue(np.array_equal(mcmc_results.logp, self.mcmc_logp))
        self.assertTrue(np.array_equal(mcmc_results.r_hat, self.mcmc_rhat))
        self.assertTrue(
            np.array_equal(mcmc_results.effective_sizes, self.mcmc_effective_sizes)
        )

    def test_data_class(self):
        x_data = np.linspace(0, 10, 100)
        y_data = 3 * np.sin(x_data) + np.random.normal(0, 0.5, 100)
        bounds = [(0, 10), (0, 10)]

        data = Data(x_data=x_data, y_data=y_data, bounds=bounds)
        data.OPTIMAL_PARAM_ = np.array([1.0, 2.0])
        self.assertTrue(np.array_equal(data.optimal_params, data.OPTIMAL_PARAM_))
        self.assertEqual(data.optimal_logp, -np.inf)

        mcmc_results = MCMC_Results(
            SAMPLES_=self.mcmc_samples,
            LogP_=self.mcmc_logp,
            R_hat_=self.mcmc_rhat,
            EFFECTIVE_SIZES_=self.mcmc_effective_sizes,
        )
        data.mcmc_results = mcmc_results

        bic_value = data.get_bic()
        self.assertIsInstance(bic_value, float)


if __name__ == "__main__":
    unittest.main()
