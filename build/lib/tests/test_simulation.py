import os
import numpy as np
import pandas as pd
import pytest
from bayescurvefit.simulation import Simulation
from bayescurvefit.distribution import GaussianMixturePDF

def load_csv(filename):
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "files/simulation", filename)
    return pd.read_csv(file_path, index_col=0)

@pytest.fixture
def simulation_instance():
    """Fixture to create a Simulation instance with a set seed."""
    example_func2 = lambda x, a, b: a * x + b
    sim_x = np.linspace(1, 9, 5)
    return Simulation(
        fit_function=example_func2,
        X=sim_x,
        pdf_function=GaussianMixturePDF(pdf_params=[0.3, 1, 0.15]),
        sim_params_bounds=[(1, 1), (1, 2)],
        n_samples=20,
        n_replicates=1,
        positive_obs=False,
        seed=42,  # Ensure reproducibility
    )

def test_df_sim_params(simulation_instance):
    """Test if df_sim_params is correctly generated."""
    df_sim_params_baseline = load_csv('df_sim_params.csv')
    pd.testing.assert_frame_equal(simulation_instance.df_sim_params, df_sim_params_baseline)

def test_df_sim_data(simulation_instance):
    """Test if df_sim_data is correctly generated."""
    df_sim_data_baseline = load_csv('df_sim_data.csv')
    pd.testing.assert_frame_equal(simulation_instance.df_sim_data, df_sim_data_baseline)

def test_df_sim_data_w_error(simulation_instance):
    """Test if df_sim_data_w_error is correctly generated."""
    df_sim_data_w_error_baseline = load_csv('df_sim_data_w_error.csv')
    pd.testing.assert_frame_equal(simulation_instance.df_sim_data_w_error, df_sim_data_w_error_baseline)

def test_sim_errors(simulation_instance):
    """Test if sim_errors is correctly generated."""
    sim_errors_baseline = load_csv('sim_errors.csv')
    pd.testing.assert_frame_equal(simulation_instance.sim_errors, sim_errors_baseline)