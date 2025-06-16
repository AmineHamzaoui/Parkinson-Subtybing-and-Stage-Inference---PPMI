#!/usr/bin/env python
# coding: utf-8
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pySuStaIn.OrdinalSustain import OrdinalSustain
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print(f"Operation took {self.interval:.2f} seconds")


def calculate_probabilities(data_healthy, data_pd, score_vals):
    """Improved probability calculation with numerical stability"""
    n_samples, n_biomarkers = data_pd.shape
    N_scores = score_vals.shape[1]

    prob_nl = np.zeros((n_samples, n_biomarkers))
    prob_score = np.zeros((n_samples, n_biomarkers, N_scores))

    print("\nCalculating probabilities...")
    print(f"Healthy data shape: {data_healthy.shape}")
    print(f"PD data shape: {data_pd.shape}")
    print(f"Score values shape: {score_vals.shape}")

    for j in range(n_biomarkers):
        healthy_vals = data_healthy[:, j]
        mu, sigma = np.mean(healthy_vals), np.std(healthy_vals)
        sigma = max(sigma, 0.1)  # Prevent division by zero

        if j < 3:
            print(f"\nBiomarker {j}:")
            print(f"Healthy mean: {mu:.2f}, std: {sigma:.2f}")
            print(
                f"PD values range: {np.min(data_pd[:, j])} to {np.max(data_pd[:, j])}"
            )

        # Calculate prob_nl using CDF for better stability
        for i in range(n_samples):
            obs_val = data_pd[i, j]
            prob_nl[i, j] = (
                norm.cdf(obs_val + 0.5, loc=mu, scale=sigma)
                - norm.cdf(obs_val - 0.5, loc=mu, scale=sigma)
            )

        # Calculate prob_score with boundary checks
        for true_score in range(N_scores):
            for obs_score in range(N_scores + 1):
                mask = data_pd[:, j] == obs_score
                if obs_score == 0:
                    prob_score[mask, j, true_score] = norm.cdf(
                        0.5, loc=true_score + 1, scale=1
                    )
                elif obs_score == N_scores:
                    prob_score[mask, j, true_score] = 1 - norm.cdf(
                        N_scores - 0.5, loc=true_score + 1, scale=1
                    )
                else:
                    prob_score[mask, j, true_score] = (
                        norm.cdf(obs_score + 0.5, loc=true_score + 1, scale=1)
                        - norm.cdf(obs_score - 0.5,
                                   loc=true_score + 1, scale=1)
                    )

    # Normalize and clip probabilities
    prob_nl = np.clip(prob_nl, 1e-10, 1)
    prob_score = np.clip(prob_score, 1e-10, 1)

    print("\nProbability validation:")
    print(f"prob_nl min: {np.min(prob_nl)}, max: {np.max(prob_nl)}")
    print(f"prob_score min: {np.min(prob_score)}, max: {np.max(prob_score)}")
    print(f"NaN in prob_nl: {np.isnan(prob_nl).any()}")
    print(f"NaN in prob_score: {np.isnan(prob_score).any()}")

    return prob_nl, prob_score


if __name__ == "__main__":
    print("Loading data...")
    # Load dataset
    df = pd.read_csv(
        r"C:\Users\nss_1\Desktop\SustalIn\pySuStaIn\notebooks\result_4_long_format.csv",
        sep=";",
        low_memory=False,
    )

    # Drop duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    # Split cohorts early
    df_healthy = df[
        (df["COHORT"] == "Healthy Control") & (df["EVENT_ID"] == "V08")
    ].copy()
    df = df[df["EVENT_ID"] == "V08"].copy()

    # Define ordinal variables
    ordinal_vars = [
        'COGCAT', 'COGDXCL', 'COGSTATE', 'FEATCOGFLC', 'FEATCRTSNS', 'FEATDCRARM',
        'FEATDELHAL', 'FEATDEPRES', 'FEATDIMOLF', 'FEATDYSART', 'FEATDYSKIN',
        'FEATDYSPHG', 'FEATDYSTNA', 'FEATMYCLNS', 'FEATNEURSS', 'FEATNOLEVO',
        'FEATPOSHYP', 'FEATPST3YR', 'FEATPYRTCT', 'FEATSBDERM', 'FEATSEXDYS',
        'FEATURNDYS', 'FEATWDGAIT', 'MCAABSTR', 'MCASER7', 'MCASNTNC', 'MRIRSLT',
        'NP1ANXS', 'NP1APAT', 'NP1COG', 'NP1DPRS', 'NP1HALL', 'NP2DRES', 'NP2EAT',
        'NP2FREZ', 'NP2HOBB', 'NP2HWRT', 'NP2HYGN', 'NP2RISE', 'NP2SALV',
        'NP2SPCH', 'NP2SWAL', 'NP2TRMR', 'NP2TURN', 'NP2WALK', 'NP3PTRML',
        'NP3RTALL', 'NP3RTARL', 'NP3SPCH', 'PTCGBOTH', 'STAIAD1', 'STAIAD10',
        'STAIAD11', 'STAIAD12', 'STAIAD13', 'STAIAD14', 'STAIAD15', 'STAIAD16',
        'STAIAD17', 'STAIAD18', 'STAIAD19', 'STAIAD2', 'STAIAD20', 'STAIAD21',
        'STAIAD22', 'STAIAD23', 'STAIAD24', 'STAIAD25', 'STAIAD26', 'STAIAD27',
        'STAIAD28', 'STAIAD29', 'STAIAD3', 'STAIAD30', 'STAIAD31', 'STAIAD32',
        'STAIAD33', 'STAIAD34', 'STAIAD35', 'STAIAD36', 'STAIAD37', 'STAIAD38',
        'STAIAD39', 'STAIAD4', 'STAIAD40', 'STAIAD5', 'STAIAD6', 'STAIAD7',
        'STAIAD8', 'STAIAD9'
    ]

    selected_columns = ["PATNO", "AGE_AT_VISIT", "FINAL_SEX_ENCODED", "COHORT"] + ordinal_vars + [
        "EVENT_ID"
    ]
    df = df[selected_columns].copy()

    # Extract PD/Prodromal
    df_pd = df[df["COHORT"].isin(["PD", "Prodromal"])].copy()
    print(
        f"Healthy controls: {df_healthy.shape[0]} | PD/Prodromal: {df_pd.shape[0]}")

    # Drop biomarkers with >20% missing
    null_fraction = df_pd[ordinal_vars].isna().mean()
    cols_to_drop = list(null_fraction[null_fraction > 0.20].index)
    ordinal_vars = [v for v in ordinal_vars if v not in cols_to_drop]
    df_pd.drop(columns=cols_to_drop, inplace=True)
    df_healthy.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    # Drop biomarkers with only one unique value
    low_variance_vars = [v for v in ordinal_vars if df_pd[v].nunique() <= 1]
    if low_variance_vars:
        print(f"Dropping low-variance biomarkers: {low_variance_vars}")
        ordinal_vars = [v for v in ordinal_vars if v not in low_variance_vars]
        df_pd.drop(columns=low_variance_vars, inplace=True)
        df_healthy.drop(columns=low_variance_vars,
                        inplace=True, errors="ignore")

    print(f"Remaining biomarkers after filtering: {len(ordinal_vars)}")

    # Drop rows with missing values
    df_pd.dropna(subset=ordinal_vars, inplace=True)
    df_healthy.dropna(subset=ordinal_vars, inplace=True)

    # Create data matrices
    print("Creating data matrices...")
    data_mat_healthy = df_healthy[ordinal_vars].astype(float).values
    data_mat_pd = df_pd[ordinal_vars].astype(float).values

    # Calculate score values
    print("Calculating score values...")

    def get_score_vals(data_healthy, data_pd):
        combined = np.vstack([data_healthy, data_pd])
        max_scores = [int(combined[:, b].max()) +
                      1 for b in range(combined.shape[1])]
        score_vals = np.zeros((len(max_scores), max(max_scores)))
        for b in range(len(max_scores)):
            score_vals[b, :max_scores[b]] = np.arange(max_scores[b])
        return score_vals

    score_vals = get_score_vals(data_mat_healthy, data_mat_pd)

    # Calculate probabilities with stability fixes
    print("Calculating probabilities with stability fixes...")
    with Timer():
        prob_nl, prob_score = calculate_probabilities(
            data_mat_healthy, data_mat_pd, score_vals)

    # Run SuStaIn with stability checks
    print("\nInitializing SuStaIn model...")
    model = OrdinalSustain(
        prob_nl=prob_nl,
        prob_score=prob_score,
        score_vals=score_vals,
        biomarker_labels=ordinal_vars,
        N_startpoints=5,  # Reduced for testing
        N_S_max=1,       # Start with 1 subtype
        N_iterations_MCMC=1000,  # Reduced for testing
        output_folder="./ordinal_output",
        dataset_name="PD_ordinal_V08",
        use_parallel_startpoints=False,
        seed=42
    )

    print("\nModel configuration:")
    print(
        f"Number of biomarkers: {model._OrdinalSustain__sustainData.getNumBiomarkers()}")
    print(
        f"Number of stages: {model._OrdinalSustain__sustainData.getNumStages()}")
    print(
        f"Number of samples: {model._OrdinalSustain__sustainData.getNumSamples()}")

    # Run algorithm
    print("\nRunning SuStaIn algorithm...")
    with Timer():
        model.run_sustain_algorithm()

    # Plot results if successful
    if hasattr(model, "samples_sequence"):
        print("\nGenerating plots...")
        figs, axs = model.plot_sustain_model(
            samples_sequence=model.samples_sequence,
            samples_f=model.samples_f,
            n_samples=data_mat_pd.shape[0],
            score_vals=score_vals,
            biomarker_labels=ordinal_vars,
            ml_f_EM=model.ml_f,
            separate_subtypes=False
        )
        for i, fig in enumerate(figs):
            fig.savefig(f"./ordinal_output/sustain_plot_{i}.png")
            plt.close(fig)
        print("Analysis complete!")
