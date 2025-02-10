# import numpy as np
# import unittest
# from bayescurvefit.execution import BayesFitModel

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
#     "fit_pec50":5.864,
#     "fit_slope":1.0175,
#     "fit_front":1.0789,
#     "fit_back":0.3742,
#     "std_pec50":0.163,
#     "std_slope":0.3424,
#     "std_front":0.0548,
#     "std_back":0.0449,
#     "est_std":0.0834,
#     "null_mean":0.7651,
#     "rmse":0.1218,
#     "pep":0.0389,
#     "convergence_warning":False,
# }

# expected_init_params = np.array([
#     [5.58646402, 6.83188019, 1.05100878, 0.696198],
#     [6.45291133, 6.71058878, 0.31742145, 0.94018326],
#     [5.93594318, 9.25010741, 0.53524164, 0.76147349],
#     [7.17806744, 5.30010191, 0.68053178, 0.71439927],
#     [7.72270761, 4.89874427, 1.09033778, 0.33770591],
#     [5.12095613, 1.08471426, 0.75459064, 0.45341102]
# ])

# expected_sa_logp = np.array([1.93398571, 3.88952939, 3.9246344 , 3.99596983, 4.06764128,
#        4.20683231, 4.29672692, 6.26479105, 6.28812786, 6.58644003,
#        6.90552112, 7.16039128, 7.45689347, 7.51631419, 7.60868093,
#        7.92370672, 7.94223019, 8.16091371, 8.1641352 , 8.22836592,
#        8.27574532, 8.34168306, 8.3647144 , 8.36734812, 8.41463087,
#        8.41592676, 8.41854089, 8.42169579, 8.42807503, 8.43141   ,
#        8.43306458, 8.43306468])

# expected_ols_params = np.array([5.69578941, 1.08669861, 0.99338925, 0.36554596])
# expected_r_hat = np.array([1.03578338, 1.09608842, 1.08881342, 1.08360159])
# expected_effective_size = np.array([672.67038378, 377.77348231, 321.57013323, 303.3782508 ])

# class TestBayesFitModel(unittest.TestCase):
    
#     def setUp(self):
#         self.run = BayesFitModel(
#             x_data=x_data,
#             y_data=y_data,
#             fit_function=log_logistic_4p,
#             params_range=params_range,
#             param_names=param_names,
#         )

#     def test_generate_random_init_params(self):
#         generated_params = self.run.bayes_fit_pipe.generate_random_init_params(6)
#         self.assertTrue(np.allclose(generated_params, expected_init_params, atol=1e-4), 
#                         "Generated initial parameters do not match expected values.")
    
#     def test_sa_logp(self):
#         generated_sa_logp = self.run.bayes_fit_pipe.data.sa_results.LogP_
#         self.assertTrue(np.allclose(generated_sa_logp, expected_sa_logp, atol=1e-4),
#                         "Generated sa_logp values do not match expected values.")
        
#     def test_ols(self):
#         ols_params = self.run.bayes_fit_pipe.data.ols_results.PARAMS_
#         self.assertTrue(np.allclose(ols_params, expected_ols_params, atol=1e-4),
#                         "Generated ols_params values do not match expected values.")

#     def test_mcmc_rhat(self):
#         r_hat = self.run.bayes_fit_pipe.data.mcmc_results.R_hat_
#         self.assertTrue(np.allclose(r_hat, expected_r_hat, atol=1e-4),
#             "Generated r_hat values do not match expected values.")

#     def test_mcmc_effectivesize(self):
#         effective_size = self.run.bayes_fit_pipe.data.mcmc_results.EFFECTIVE_SIZES_
#         self.assertTrue(np.allclose(effective_size, expected_effective_size, atol=1e-4),
#             "Generated effective_size values do not match expected values.")
        
#     def test_bayes_fit_model(self):
#         result = self.run.get_result()
        
#         for param, expected_value in expected_result.items():
#             with self.subTest(param=param):
#                 self.assertAlmostEqual(result[param], expected_value, delta=1e-4, 
#                     msg=f"Unexpected value for {param}. Expected {expected_value}, but got {result[param]}")

# if __name__ == "__main__":
#     unittest.main()