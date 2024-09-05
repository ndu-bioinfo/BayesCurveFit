import unittest, os
import numpy as np
import pandas as pd
import pickle
from scipy.stats import norm, laplace
from bayescurvefit.distribution import SimplePDF, GaussianLaplacePDF, GaussianMixturePDF 

def load_output(filename):
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "files/distribution", filename)
    with open(file_path, "rb") as f:
        return pickle.load(f)


class TestPDFs(unittest.TestCase):
    
    def setUp(self):
        base_dir = os.path.dirname(__file__)
        self.data = pd.read_csv(os.path.join(base_dir, "files/simulation/sim_errors.csv"), index_col = 0).values.flatten()
        self.gaussian_laplace_pdf = GaussianLaplacePDF()
        self.gaussian_mixture_pdf = GaussianMixturePDF()
    
    def test_simple_pdf_initialization(self):
        pdf = SimplePDF([5.0, 0.1])
        self.assertEqual(pdf.fixed_pdf_params, [5.0, 0.1])
    
    def test_simple_pdf_mixture(self):
        pdf = SimplePDF([5.0, 0.1])
        data = np.array([0, 1, 2])
        result = pdf.mixture_pdf([1.0], data)
        expected = (1 - 0.1) * norm.pdf(data, loc=0, scale=1.0) + 0.1 * laplace.pdf(data, loc=0, scale=5.0)
        np.testing.assert_allclose(result, expected, rtol=0.1)
    
    def test_gaussian_laplace_pdf_fit(self):
        self.gaussian_laplace_pdf.fit_params(self.data)
        fitted_params = self.gaussian_laplace_pdf.get_pdf_params
        saved_params = load_output("gaussian_laplace_pdf_params.pkl")
        np.testing.assert_allclose(fitted_params, saved_params, rtol=0.1)

    def test_gaussian_laplace_pdf_sample(self):
        self.gaussian_laplace_pdf.fit_params(self.data)
        samples = self.gaussian_laplace_pdf.sample(size=100)
        self.assertEqual(len(samples), 100)


    def test_gaussian_mixture_pdf_fit(self):
        self.gaussian_mixture_pdf.fit_params(self.data)
        fitted_params = self.gaussian_mixture_pdf.get_pdf_params
        saved_params = load_output("gaussian_mixture_pdf_params.pkl")
        np.testing.assert_allclose(fitted_params, saved_params, rtol=0.1)

    def test_gaussian_mixture_pdf_sample(self):
        self.gaussian_mixture_pdf.fit_params(self.data)
        samples = self.gaussian_mixture_pdf.sample(size=100)
        self.assertEqual(len(samples), 100)

    def test_exceptions(self):
        # Test for exceptions if fitting is not done
        with self.assertRaises(ValueError):
            self.gaussian_laplace_pdf.get_pdf_params
        
        with self.assertRaises(ValueError):
            self.gaussian_mixture_pdf.get_pdf_params

if __name__ == "__main__":
    unittest.main()