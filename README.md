# BayesCurveFit

BayesCurveFit: A Bayesian Inference Workflow for Enhanced Curve Fitting in Undersampled Drug Discovery Data

## Overview
BayesCurveFit is a Python package designed to apply Bayesian inference for curve fitting, especially tailored for undersampled and outlier-contaminated data. It supports advanced model fitting and uncertainty estimation for biological data, such as dose-response curves in drug discovery.

This example demonstrates how to use the BayesCurveFit package to perform a 4-parameter log-logistic model fit using Bayesian inference.

## Example Usage

```python
import numpy as np
from bayescurvefit.execution import BayesFitModel

def log_logistic_4p(x: np.ndarray, pec50: float, slope: float, front: float, back: float) -> np.ndarray:
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        y = (front - back) / (1 + 10 ** (slope * (x + pec50))) + back
        return y

# Example dose-response data
x_data = np.array([-9.0, -8.3, -7.6, -6.9, -6.1, -5.4, -4.7, -4.0])
y_data = np.array([1.12, 0.74, 1.03, 1.08, 0.76, 0.61, 0.39, 0.38])

# Parameter ranges and names
params_range = [(5, 8), (0.01, 10), (0.28, 1.22), (0.28, 1.22)]
param_names = ["pec50", "slope", "front", "back"]

# Running the Bayesian fit
run = BayesFitModel(
    x_data=x_data,
    y_data=y_data,
    fit_function=log_logistic_4p,
    params_range=params_range,
    param_names=param_names,
)

# Get the fit result
result = run.get_result()
print(result)