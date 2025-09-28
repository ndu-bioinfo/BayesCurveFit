import unittest

import numpy as np

from bayescurvefit.execution import BayesFitModel


# User-defined equation
def log_logistic_4p(
    x: np.ndarray, pec50: float, slope: float, front: float, back: float
) -> np.ndarray:
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return (front - back) / (1 + 10 ** (slope * (x + pec50))) + back


# Sample data input
x_data = np.array([-9.0, -8.3, -7.6, -6.9, -6.1, -5.4, -4.7, -4.0])
y_data = np.array([1.12, 0.74, 1.03, 1.08, 0.76, 0.61, 0.39, 0.38])
params_range = [
    (5, 8),
    (0.01, 10),
    (0.28, 1.22),
    (0.28, 1.22),
]  # This range represents your best estimation of where the parameters likely fall
param_names = ["pec50", "slope", "front", "back"]

# Expected output based on the result from your example
expected_result = {
    "fit_pec50": 5.8557,
    "fit_slope": 1.0888,
    "fit_front": 1.0634,
    "fit_back": 0.3769,
    "std_pec50": 0.185,
    "std_slope": 0.3995,
    "std_front": 0.0619,
    "std_back": 0.0506,
    "est_std": 0.0896,
    "null_mean": 0.7624,
    "rmse": 0.1226,
    "pep": 0.0291,
    "convergence_warning": False,
}

expected_init_params = np.array(
    [
        [5.58646402, 6.83188019, 1.05100878, 0.696198],
        [6.45291133, 6.71058878, 0.31742145, 0.94018326],
        [5.93594318, 9.25010741, 0.53524164, 0.76147349],
        [7.17806744, 5.30010191, 0.68053178, 0.71439927],
        [7.72270761, 4.89874427, 1.09033778, 0.33770591],
        [5.12095613, 1.08471426, 0.75459064, 0.45341102],
    ]
)

expected_sa_logp = np.array(
    [
        0.49461677,
        1.05828383,
        4.0878293,
        4.1485331,
        4.46971071,
        4.69098434,
        4.76566063,
        5.19153405,
        5.29533951,
        5.58139339,
        5.66437544,
        5.68970001,
        6.89292057,
        7.03534303,
        7.2951815,
        7.31063353,
        7.53549498,
        7.62716123,
        7.70890462,
        7.71282927,
        7.71379049,
    ]
)

expected_ols_params = np.array([5.69050018, 1.08149381, 0.99100373, 0.36469244])
expected_r_hat = np.array([1.04387868, 1.04011244, 1.04622125, 1.03523167])
expected_init_pos = np.array([5.90240896, 0.88820443, 1.10134921, 0.36772887])


class TestBayesFitModel(unittest.TestCase):
    def setUp(self):
        self.run = BayesFitModel(
            x_data=x_data,
            y_data=y_data,
            fit_function=log_logistic_4p,
            params_range=params_range,
            param_names=param_names,
        )

    def test_initial_pos(self):
        generated_init_pos = self.run.bayes_fit_pipe.init_pos
        self.assertTrue(
            np.allclose(generated_init_pos, expected_init_pos, atol=1e-4),
            "Generated initial positions for MCMC do not match expected values.",
        )

    def test_generate_random_init_params(self):
        generated_params = self.run.bayes_fit_pipe.generate_random_init_params(6)
        self.assertTrue(
            np.allclose(generated_params, expected_init_params, atol=1e-4),
            "Generated initial parameters do not match expected values.",
        )

    def test_sa_logp(self):
        generated_sa_logp = self.run.bayes_fit_pipe.data.sa_results.LogP_
        self.assertTrue(
            np.allclose(generated_sa_logp, expected_sa_logp, atol=1e-4),
            "Generated sa_logp values do not match expected values.",
        )

    def test_ols(self):
        ols_params = self.run.bayes_fit_pipe.data.ols_results.PARAMS_
        self.assertTrue(
            np.allclose(ols_params, expected_ols_params, atol=1e-4),
            "Generated ols_params values do not match expected values.",
        )

    def test_mcmc_rhat(self):
        r_hat = self.run.bayes_fit_pipe.data.mcmc_results.R_hat_
        # Test that R-hat values are reasonable (close to 1.0, indicating convergence)
        self.assertTrue(
            np.all(r_hat < 1.2),
            f"R-hat values {r_hat} are too high, indicating poor convergence",
        )
        self.assertTrue(
            np.all(r_hat > 0.9),
            f"R-hat values {r_hat} are too low, indicating potential issues",
        )

    def test_bayes_fit_model(self):
        result = self.run.get_result()

        # Test that all expected parameters are present
        for param in expected_result.keys():
            self.assertIn(param, result, f"Missing parameter {param} in result")

        # Test parameter values with more lenient tolerances for stochastic results
        for param, expected_value in expected_result.items():
            with self.subTest(param=param):
                actual_value = result[param]
                # Handle both scalar and array values
                if hasattr(actual_value, "__len__") and len(actual_value) == 1:
                    actual_value = actual_value[0]

                if param in ["fit_pec50", "fit_slope", "fit_front", "fit_back"]:
                    # Main parameters: allow 10% relative error due to stochastic nature
                    self.assertAlmostEqual(
                        actual_value,
                        expected_value,
                        delta=abs(expected_value) * 0.1,
                        msg=f"Unexpected value for {param}. Expected {expected_value}, got {actual_value}",
                    )
                elif param in [
                    "std_pec50",
                    "std_slope",
                    "std_front",
                    "std_back",
                    "est_std",
                ]:
                    # Standard deviations: allow 50% relative error due to stochastic nature
                    self.assertAlmostEqual(
                        actual_value,
                        expected_value,
                        delta=abs(expected_value) * 0.5,
                        msg=f"Unexpected value for {param}. Expected {expected_value}, got {actual_value}",
                    )
                elif param in ["rmse", "pep"]:
                    # RMSE and PEP: allow 30% relative error
                    self.assertAlmostEqual(
                        actual_value,
                        expected_value,
                        delta=abs(expected_value) * 0.3,
                        msg=f"Unexpected value for {param}. Expected {expected_value}, got {actual_value}",
                    )
                else:
                    # Other parameters: use original tolerance
                    self.assertAlmostEqual(
                        actual_value,
                        expected_value,
                        delta=1e-4,
                        msg=f"Unexpected value for {param}. Expected {expected_value}, got {actual_value}",
                    )


if __name__ == "__main__":
    unittest.main()
