#!/usr/bin/env python
# coding: utf-8
import sys
from pySuStaIn.OrdinalSustain import OrdinalSustain
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# 1) Load dataset
df = pd.read_csv(
    r'C:\Users\nss_1\Desktop\SustalIn\pySuStaIn\notebooks\result_4_long_format.csv',
    sep=';',
    low_memory=False
)

# 2) Drop duplicate columns
df = df.loc[:, ~df.columns.duplicated()]

# 3) Split cohorts early
df_healthy = df[(df["COHORT"] == "Healthy Control")
                & (df["EVENT_ID"] == "V08")].copy()
df = df[df["EVENT_ID"] == "V08"].copy()

# 4) Define ordinal variables
ordinal_vars = [
    'NP1APAT', 'NP1COG', 'NP1DPRS'
]

selected_columns = ["PATNO", "AGE_AT_VISIT",
                    "FINAL_SEX_ENCODED", "COHORT"] + ordinal_vars + ["EVENT_ID"]
df = df[selected_columns].copy()

# 5) Extract PD/Prodromal
df_pd = df[df["COHORT"].isin(["PD", "Prodromal"])].copy()
print(
    f"Healthy controls: {df_healthy.shape[0]} | PD/Prodromal: {df_pd.shape[0]}")

# 6) Drop biomarkers with >20% missing
null_fraction = df_pd[ordinal_vars].isna().mean()
cols_to_drop = list(null_fraction[null_fraction > 0.20].index)
ordinal_vars = [v for v in ordinal_vars if v not in cols_to_drop]
df_pd.drop(columns=cols_to_drop, inplace=True)
df_healthy.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# 7) Drop biomarkers with only one unique value
low_variance_vars = [v for v in ordinal_vars if df_pd[v].nunique() <= 1]
if low_variance_vars:
    print(f"Dropping low-variance biomarkers: {low_variance_vars}")
    ordinal_vars = [v for v in ordinal_vars if v not in low_variance_vars]
    df_pd.drop(columns=low_variance_vars, inplace=True)
    df_healthy.drop(columns=low_variance_vars, inplace=True, errors='ignore')

print(f"Remaining biomarkers after filtering: {len(ordinal_vars)}")

# 8) Drop rows with missing values
df_pd.dropna(subset=ordinal_vars, inplace=True)
df_healthy.dropna(subset=ordinal_vars, inplace=True)

# 9) Create data matrices
data_mat_healthy = df_healthy[ordinal_vars].astype(float).values
data_mat_pd = df_pd[ordinal_vars].astype(float).values
n_samples = data_mat_pd.shape[0]
n_biomarkers = data_mat_pd.shape[1]

# 10) Build score matrix


def get_score_vals_padded(data_mat_healthy, data_mat_pd):
    combined_data = np.vstack([data_mat_healthy, data_mat_pd])
    n_biomarkers = combined_data.shape[1]
    max_scores_per_biomarker = [
        int(combined_data[:, b].max()) + 1 for b in range(n_biomarkers)]
    max_num_scores = max(max_scores_per_biomarker)
    score_vals = np.zeros((n_biomarkers, max_num_scores), dtype=int)
    for b in range(n_biomarkers):
        valid_scores = np.arange(0, max_scores_per_biomarker[b])
        score_vals[b, :len(valid_scores)] = valid_scores
    return score_vals


score_vals = get_score_vals_padded(data_mat_healthy, data_mat_pd)

# 11) Compute prob_nl
prob_nl = np.zeros((n_samples, n_biomarkers))
for j in range(n_biomarkers):
    vals = data_mat_healthy[:, j]
    mu, sigma = np.mean(vals), np.std(vals)
    sigma = sigma if sigma > 0 else 0.1
    prob_nl[:, j] = norm.pdf(data_mat_pd[:, j], loc=mu, scale=sigma)

# 12) Build p_score_dist matrix
score_vals_flat = sorted(set(np.unique(score_vals)))
N_scores = len(score_vals_flat)
# rows = true scores (1 to N), cols = observed (0 to N)
p_score_dist = np.zeros((N_scores, N_scores + 1))
for z in range(1, N_scores + 1):  # true score from 1 to N_scores
    for s in range(N_scores + 1):  # observed score from 0 to N_scores
        p_score_dist[z - 1, s] = norm.pdf(s, loc=z, scale=1.0)
    p_score_dist[z - 1] /= np.sum(p_score_dist[z - 1])  # normalize

# 13) Compute prob_score


def compute_prob_score(data_mat_pd, p_score_dist):
    n_samples, n_biomarkers = data_mat_pd.shape
    N_scores = p_score_dist.shape[0]
    prob_score = np.zeros((n_samples, n_biomarkers, N_scores))
    for z in range(N_scores):
        for s in range(N_scores + 1):
            match_indices = (data_mat_pd == s)
            prob_score[match_indices, z] = p_score_dist[z, s]
    return prob_score


prob_score = compute_prob_score(data_mat_pd, p_score_dist)

# 14) Check for NaNs
print("NaNs in prob_nl:", np.isnan(prob_nl).sum())
print("NaNs in prob_score:", np.isnan(prob_score).sum())
if np.isnan(prob_nl).any() or np.isnan(prob_score).any():
    raise ValueError(
        "There are NaNs in the probability matrices. Please check data preprocessing.")

# 15) Output directory
output_folder = "./ordinal_output"
os.makedirs(output_folder, exist_ok=True)

# 16) Run SuStaIn
model = OrdinalSustain(
    prob_nl=prob_nl,
    prob_score=prob_score,
    score_vals=score_vals,
    biomarker_labels=ordinal_vars,
    N_startpoints=25,
    N_S_max=2,
    N_iterations_MCMC=int(1e4),
    output_folder=output_folder,
    dataset_name="PD_ordinal_V08",
    use_parallel_startpoints=False,
    seed=42
)
model.run_sustain_algorithm()

# 17) Plot results
figs, axs = model.plot_sustain_model(
    samples_sequence=model.samples_sequence,
    samples_f=model.samples_f,
    n_samples=n_samples,
    score_vals=score_vals,
    biomarker_labels=ordinal_vars,
    ml_f_EM=model.ml_f,
    separate_subtypes=False
)

for i, fig in enumerate(figs):
    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, f"sustain_plot_{i}.png"))
    plt.show()
