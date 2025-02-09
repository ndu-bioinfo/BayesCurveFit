import numpy as np
import unittest
from bayescurvefit.execution import BayesFitModel

# User-defined equation
def log_logistic_4p(x: np.ndarray, pec50: float, slope: float, front: float, back: float) -> np.ndarray:
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return (front - back) / (1 + 10 ** (slope * (x + pec50))) + back

# Sample data input
x_data = np.array([-9.0, -8.3, -7.6, -6.9, -6.1, -5.4, -4.7, -4.0])
y_data = np.array([1.10, 0.75, 1.05, 1.08, 0.76, 0.61, 0.39, 0.38])
params_range = [(5, 8), (0.01, 10), (0.28, 1.22), (0.28, 1.22)]
param_names = ["pec50", "slope", "front", "back"]

# Expected output based on the result from your example
expected_result = {
    "fit_pec50":5.864,
    "fit_slope":1.0175,
    "fit_front":1.0789,
    "fit_back":0.3742,
    "std_pec50":0.163,
    "std_slope":0.3424,
    "std_front":0.0548,
    "std_back":0.0449,
    "est_std":0.0834,
    "null_mean":0.7651,
    "rmse":0.1218,
    "pep":0.0389,
    "convergence_warning":False,
}


class TestBayesFitModel(unittest.TestCase):

    def test_bayes_fit_model(self):
        run = BayesFitModel(
            x_data=x_data,
            y_data=y_data,
            fit_function=log_logistic_4p,
            params_range=params_range,
            param_names=param_names,
        )

        result = run.get_result()
        
        for param, expected_value in expected_result.items():
            with self.subTest(param=param):
                self.assertAlmostEqual(result[param], expected_value, delta=1e-4, 
                    msg=f"Unexpected value for {param}. Expected {expected_value}, but got {result[param]}")

if __name__ == "__main__":
    unittest.main()