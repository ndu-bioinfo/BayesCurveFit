import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from bayescurvefit.execution import BayesFitModel


def log_logistic_4p(x, pec50, slope, front, back):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        y = (front - back) / (1 + 10 ** (slope * (x + pec50))) + back
        return y


def process_sample(sample, x_data, y_data):
    mask = ~np.isnan(y_data)
    bayesfit_run = BayesFitModel(
        x_data=x_data[mask],
        y_data=y_data[mask],
        fit_function=log_logistic_4p,
        params_range=[
            (4.53, 8.53),  # measurement bounds
            (0.01, 10),  # same as CurveCurator
            (max(min(y_data[mask]) - 0.1, 0.0001), max(y_data[mask]) + 0.1),
            (max(min(y_data[mask]) - 0.1, 0.0001), max(y_data[mask]) + 0.1),
        ],
        param_names=["pec50", "slope", "front", "back"],
        run_mcmc=True,
    )
    return sample, bayesfit_run.get_result()


def run_bayescurvefit(input_csv, x_data, drug, output_csv):
    df_input = pd.read_csv(input_csv).set_index("Gene names")
    data_cols = df_input.filter(like="Ratio").columns
    df_input.index = [x.split("_$$")[0] for x in df_input.index]

    samples = df_input.index.tolist()
    y_data_all = df_input[data_cols].values

    results = Parallel(n_jobs=-1)(
        delayed(process_sample)(sample, x_data, y_data_all[i])
        for i, sample in enumerate(tqdm(samples))
    )

    dict_result = {sample: res for sample, res in results}
    df_output = pd.DataFrame(dict_result).T
    df_output["drug"] = drug
    df_output.to_csv(output_csv)


if __name__ == "__main__":
    import sys

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    drug = sys.argv[3]
    doses = [3e-3, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 30000.0]
    x_data = np.log10(np.array(doses) * 1e-9)

    run_bayescurvefit(input_csv, x_data, drug, output_csv)
