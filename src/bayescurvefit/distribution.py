from typing import List, Optional, Union

import numpy as np
from scipy.optimize import minimize
from scipy.stats import laplace, norm


class SimplePDF:
    def __init__(self, fixed_pdf_params: List[float] = [5.0, 0.1]) -> None:
        """
        The GaussianLaplacePDF has fixed  mu, b, mix_frac, useful in conditional posterior estimation.
        fixed_pdf_params: Fix parameters for the mixture PDF [mu, b, mix_frac].
                b: Scale parameter of the Laplace distribution.
                mix_frac: Fraction of the Laplace distribution.
        """
        self.fixed_pdf_params = fixed_pdf_params

    def mixture_pdf(
        self, pdf_params: List[float], data: np.ndarray
    ) -> Union[float, np.ndarray]:
        """
        Mixture probability density function consisting of a given distribution and a Laplace distribution.

        Args:
            pdf_params: Parameters for the mixture PDF [scale].
                scale: Scale parameter of the Gaussian distribution.
            data: Data to be used for calculating the PDF, optional for checking PDF at specified conditions.

        Returns:
            Mixture PDF value(s) at the given point(s).
        """
        b, mix_frac = self.fixed_pdf_params
        pdf_norm = norm.pdf(data, loc=0, scale=pdf_params[0])
        pdf_laplace = laplace.pdf(data, loc=0, scale=b)
        return (1 - mix_frac) * pdf_norm + mix_frac * pdf_laplace


class GaussianLaplacePDF:
    def __init__(self, pdf_params: np.array = None):
        """
        Initialize the PDFMixtureFitter class.

        """
        self.x = None
        self.pdf_params = pdf_params

    def mixture_pdf(
        self, pdf_params: List[float] = None, data: Optional[np.ndarray] = None
    ) -> Union[float, np.ndarray]:
        """
        Mixture probability density function consisting of a given distribution and a Laplace distribution.

        Args:
            pdf_params: Parameters for the mixture PDF [mu, b, scale, mix_frac].
                mu: Mean of the mixture distribution.
                b: Scale parameter of the Laplace distribution.
                scale: Scale parameter for the main PDF.
                mix_frac: Fraction of hidden errors represented by the Laplace PDF.
            data: Data to be used for calculating the PDF, optional for checking PDF at specified conditions.

        Returns:
            Mixture PDF value(s) at the given point(s).
        """
        if data is not None:
            self.x = data
        if pdf_params is not None:
            b, scale, mix_frac = pdf_params
        else:
            b, scale, mix_frac = self.pdf_params

        pdf_norm = norm.pdf(self.x, loc=0, scale=scale)
        pdf_laplace = laplace.pdf(self.x, loc=0, scale=b)
        return (1 - mix_frac) * pdf_norm + mix_frac * pdf_laplace

    def _fit_mix_function(self, pdf_params: List[float]) -> float:
        """
        Function to minimize during fitting. Calculates the sum of negative log-likelihood of the mixture PDF.

        Args:
            params: Parameters for the mixture PDF [b, scale, mix_frac].

        Returns:
            Negative sum log-likelihood value.
        """
        pdf_values = self.mixture_pdf(pdf_params=pdf_params)
        return -np.sum(np.log(pdf_values))

    def fit_params(
        self,
        data: np.ndarray,
        init_params_bound: Optional[List[tuple]] = None,
        method: str = "L-BFGS-B",
        options: Optional[dict] = None,
        return_params: bool = False,
    ) -> np.ndarray:
        """
        Fit the parameters of the mixture PDF to the data.

        Args:
            data: Error data to fit.
            init_params_bound: Bounds for the parameters [b, scale, mix_frac].
            method: Optimization method to use.
            options: Additional options for the optimizer.

        Returns:
            Optimized parameters.
        """
        self.x = data
        self.params = None
        if init_params_bound is None:
            init_params_bound = [(0.5, None), (1e-6, None), (0.01, 0.5)]
        if options is None:
            options = {"maxiter": 10000}

        lower = np.percentile(data, 5)
        upper = np.percentile(data, 95)
        init_sigma = np.std(data[(data > lower) & (data < upper)])

        result = minimize(
            self._fit_mix_function,
            x0=[0.5, init_sigma, 0.05],
            bounds=init_params_bound,
            method=method,
            options=options,
        )
        self.pdf_params = result.x
        self.x = None
        if return_params:
            return self.pdf_params

    @property
    def get_pdf_params(self) -> np.array:
        """
        Return the fitted PDF parameters.

        Raises:
            ValueError: If parameters are not fitted yet.

        Returns:
            Fitted PDF parameters.
        """
        if self.pdf_params is not None:
            return self.pdf_params
        else:
            raise ValueError("Not parameters fitted yet, run fit_params first")

    def sample(self, pdf_params: List[float] = None, size: int = 10) -> np.ndarray:
        """
        Sample from the Gaussian mixture distribution.

        Args:
            size: Number of samples to generate.

        Returns:
            Random samples from the Gaussian mixture distribution.
        """
        if pdf_params is None:
            b, scale, mix_frac = self.get_pdf_params
        else:
            b, scale, mix_frac = pdf_params
        n_samples_1 = int(size * (1 - mix_frac))
        n_samples_2 = size - n_samples_1

        samples_1 = np.random.normal(loc=0, scale=scale, size=n_samples_1)
        samples_2 = np.random.laplace(loc=0, scale=b, size=n_samples_2)
        samples = np.concatenate([samples_1, samples_2])
        np.random.shuffle(samples)
        return samples


class GaussianMixturePDF:
    def __init__(self, pdf_params: np.array = None):
        """
        Initialize the GaussianMixturePDF class.
        """
        self.x = None
        self.pdf_params = pdf_params

    def mixture_pdf(
        self, pdf_params: List[float] = None, data: Optional[np.ndarray] = None
    ) -> Union[float, np.ndarray]:
        """
        Mixture probability density function consisting of two Gaussian distributions centered at 0.

        Args:
            pdf_params: Parameters for the mixture PDF [scale1, scale2, mix_frac].
                scale1: Scale parameter for the first Gaussian PDF.
                scale2: Scale parameter for the second Gaussian PDF.
                mix_frac: Fraction of the first Gaussian distribution.
            data: Data to be used for calculating the PDF, optional for checking PDF at specified conditions.

        Returns:
            Mixture PDF value(s) at the given point(s).
        """
        if data is not None:
            self.x = data
        if pdf_params is not None:
            scale1, scale2, mix_frac = pdf_params
        else:
            scale1, scale2, mix_frac = self.pdf_params

        pdf1 = norm.pdf(self.x, loc=0, scale=scale1)
        pdf2 = norm.pdf(self.x, loc=0, scale=scale2)
        return (1 - mix_frac) * pdf1 + mix_frac * pdf2

    def _fit_mix_function(self, pdf_params: List[float]) -> float:
        """
        Function to minimize during fitting. Calculates the sum of negative log-likelihood of the mixture PDF.

        Args:
            params: Parameters for the mixture PDF [scale1, scale2, mix_frac].

        Returns:
            Negative sum log-likelihood value.
        """
        pdf_values = self.mixture_pdf(pdf_params=pdf_params)
        return -np.sum(np.log(pdf_values))

    def fit_params(
        self,
        data: np.ndarray,
        init_params_bound: Optional[List[tuple]] = None,
        method: str = "L-BFGS-B",
        options: Optional[dict] = None,
        return_params: bool = False,
    ) -> np.ndarray:
        """
        Fit the parameters of the mixture PDF to the data.

        Args:
            data: Error data to fit.
            init_params_bound: Bounds for the parameters [scale1, scale2, mix_frac].
            method: Optimization method to use.
            options: Additional options for the optimizer.

        Returns:
            Optimized parameters.
        """
        self.x = data
        self.params = None
        if init_params_bound is None:
            init_params_bound = [(1e-6, None), (1e-6, None), (0.01, 0.99)]
        if options is None:
            options = {"maxiter": 10000}

        lower = np.percentile(data, 25)
        upper = np.percentile(data, 75)
        init_sigma1 = np.std(data[(data > lower) & (data < upper)])
        init_sigma2 = np.std(data) * 2

        result = minimize(
            self._fit_mix_function,
            x0=[init_sigma1, init_sigma2, 0.2],
            bounds=init_params_bound,
            method=method,
            options=options,
        )
        self.pdf_params = result.x
        self.x = None
        if return_params:
            return self.pdf_params

    @property
    def get_pdf_params(self) -> np.array:
        """
        Return the fitted PDF parameters.

        Raises:
            ValueError: If parameters are not fitted yet.

        Returns:
            Fitted PDF parameters.
        """
        if self.pdf_params is not None:
            return self.pdf_params
        else:
            raise ValueError("No parameters fitted yet, run fit_params first")

    def sample(self, pdf_params: List[float] = None, size: int = 10) -> np.ndarray:
        """
        Sample from the Gaussian mixture distribution.

        Args:
            size: Number of samples to generate.

        Returns:
            Random samples from the Gaussian mixture distribution.
        """
        if pdf_params is None:
            scale1, scale2, mix_frac = self.get_pdf_params
        else:
            scale1, scale2, mix_frac = pdf_params
        n_samples1 = int(size * (1 - mix_frac))
        n_samples2 = size - n_samples1

        samples1 = np.random.normal(loc=0, scale=scale1, size=n_samples1)
        samples2 = np.random.normal(loc=0, scale=scale2, size=n_samples2)
        samples = np.concatenate([samples1, samples2])
        np.random.shuffle(samples)
        return samples
