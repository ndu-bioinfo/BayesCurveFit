import numpy as np
import unittest
from bayescurvefit.execution import BayesFitModel

class TestRandomGenerator(unittest.TestCase):
    def test_random_sample(self):
        rng1 = np.random.default_rng(42)
        generated_sample = rng1.random(5)
        expected_sample = np.array([0.77395605, 0.43887844, 0.85859792, 0.69736803, 0.09417735])
        np.testing.assert_allclose(generated_sample, expected_sample, atol=1e-8)

# # User-defined equation
# def log_logistic_4p(x: np.ndarray, pec50: float, slope: float, front: float, back: float) -> np.ndarray:
#     with np.errstate(over="ignore", under="ignore", invalid="ignore"):
#         return (front - back) / (1 + 10 ** (slope * (x + pec50))) + back

# # Sample data input
# x_data = np.array([-9.0, -8.3, -7.6, -6.9, -6.1, -5.4, -4.7, -4.0])
# y_data = np.array([1.10, 0.75, 1.05, 1.08, 0.76, 0.61, 0.39, 0.38])
# params_range = [(5, 8), (0.01, 10), (0.28, 1.22), (0.28, 1.22)]
# param_names = ["pec50", "slope", "front", "back"]

# # Expected output based on the result from your example
# expected_result = {
#     "fit_pec50":5.8263,
#     "fit_slope":0.9744,
#     "fit_front":1.0769,
#     "fit_back":0.3611,
#     "std_pec50":0.1595,
#     "std_slope":0.3459,
#     "std_front":0.0561,
#     "std_back":0.0466,
#     "est_std":0.084,
#     "null_mean":0.765,
#     "rmse":0.121,
#     "pep":0.0377,
#     "convergence_warning":False,
# }


# class TestBayesFitModel(unittest.TestCase):

#     def test_bayes_fit_model(self):
#         run = BayesFitModel(
#             x_data=x_data,
#             y_data=y_data,
#             fit_function=log_logistic_4p,
#             params_range=params_range,
#             param_names=param_names,
#         )

#         result = run.get_result()
        
#         for param, expected_value in expected_result.items():
#             with self.subTest(param=param):
#                 self.assertAlmostEqual(result[param], expected_value, delta=1e-4, 
#                     msg=f"Unexpected value for {param}. Expected {expected_value}, but got {result[param]}")

if __name__ == "__main__":
    unittest.main()