#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 16:00:13 2025

@author: admin
"""
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import chi2
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Load and prepare NAO index data
nao_index = pd.read_csv('./data_persistence/norm.daily.nao.cdas.z500.19500101_current.csv')
nao_index['date'] = pd.to_datetime(nao_index[['year', 'month', 'day']])
nao_index.set_index('date', inplace=True)
nao_index.drop(columns=['year', 'month', 'day'], inplace=True)        
nao_index = nao_index.replace(np.nan, 0)
nao_index.columns = ['nao_index_cdas']
nao_index_winter = nao_index.loc[nao_index.index.month.isin([12,1,2])]
nao_index_winter_old = nao_index_winter.loc[nao_index_winter.index.year <= 1979]
nao_index_winter_new = nao_index_winter.loc[(nao_index_winter.index.year >= 1990) & (nao_index_winter.index.year < 2020)]

# Test for equality of the NAO index in winter
ks_2samp(nao_index_winter_new.nao_index_cdas, nao_index_winter_old.nao_index_cdas)

# Compute transition matrices for old period
q_labels_old = pd.qcut(nao_index_winter_old.nao_index_cdas, q=5, labels=False) + 1  # labels 1 to 5
counts_old = np.zeros((5, 5))
for t in range(len(q_labels_old) - 1):
    i = q_labels_old[t] - 1
    j = q_labels_old[t+1] - 1
    counts_old[i, j] += 1

# Transition probabilities (row-normalized)
row_sums_old = counts_old.sum(axis=1, keepdims=True)
P_hat_old = counts_old / row_sums_old

# Compute transition matrices for new period
q_labels_new = pd.qcut(nao_index_winter_new.nao_index_cdas, q=5, labels=False) + 1  # labels 1 to 5
counts_new = np.zeros((5, 5))
for t in range(len(q_labels_new) - 1):
    i = q_labels_new[t] - 1
    j = q_labels_new[t+1] - 1
    counts_new[i, j] += 1

# Transition probabilities (row-normalized)
row_sums_new = counts_new.sum(axis=1, keepdims=True)
P_hat_new = counts_new / row_sums_new

# Confidence intervals
alpha = 0.05
z = norm.ppf(1 - alpha / 2)

# Extract diagonals
diag_old = np.diag(P_hat_old)
diag_new = np.diag(P_hat_new)

# Standard errors using normal approximation
SE_old = np.sqrt(diag_old * (1 - diag_old) / row_sums_old.flatten())
SE_new = np.sqrt(diag_new * (1 - diag_new) / row_sums_new.flatten())

# Confidence intervals
lower_old = diag_old - z * SE_old
upper_old = diag_old + z * SE_old

lower_new = diag_new - z * SE_new
upper_new = diag_new + z * SE_new

# Clip to [0, 1]
lower_old = np.clip(lower_old, 0, 1)
upper_old = np.clip(upper_old, 0, 1)
lower_new = np.clip(lower_new, 0, 1)
upper_new = np.clip(upper_new, 0, 1)

# Difference and standard error of difference
diff = diag_new - diag_old
SE_diff = np.sqrt(SE_old**2 + SE_new**2)

# Z-scores and p-values
z_scores = diff / SE_diff

# Display
print(f"{'Quintile':<10} {'Old Pii':>10} {'New Pii':>10} {'95% CI Old':>18} {'95% CI New':>18}")
print("-" * 70)
for i in range(5):
    print(f"{i+1:<10} {diag_old[i]:>10.3f} {diag_new[i]:>10.3f} "
          f"[{lower_old[i]:.3f}, {upper_old[i]:.3f}] [{lower_new[i]:.3f}, {upper_new[i]:.3f}] "
         )


def plot_transition_probabilities(greyscale=False):
    """
    Plots transition probabilities with 95% confidence intervals.
    
    Parameters:
    - greyscale (bool): If True, use greyscale colors (default: False)
    """
    quintiles = np.arange(1, 6)
    x = np.arange(5)
    offset = 0.1  # horizontal shift for side-by-side points

    # Define colors based on greyscale parameter
    if greyscale:
        color_old = '#808080'  # Medium grey for old data
        color_new = '#404040'  # Dark grey for new data
    else:
        color_old = 'orange'
        color_new = 'red'

    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white', dpi=100)

    # Old period — shifted left
    ax.errorbar(
        x - offset, diag_old, yerr=z * SE_old,
        fmt='o', color=color_old, capsize=4, label='1950–1980'
    )

    # New period — shifted right
    ax.errorbar(
        x + offset, diag_new, yerr=z * SE_new,
        fmt='s' if greyscale else 'o', color=color_new, capsize=4, label='1990–2020'
    )

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels([f'P_{i}{i}' for i in quintiles])
    ax.set_ylabel('Transition Probability')
    ax.set_title('Transition Probabilities with 95% Confidence Intervals')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    # Adjust y-limits to fit the data nicely with a margin
    all_vals = np.concatenate([diag_old, diag_new])
    all_errs = np.concatenate([z * SE_old, z * SE_new])
    y_min = (all_vals - all_errs).min()
    y_max = (all_vals + all_errs).max()
    margin = 0.05 * (y_max - y_min)
    ax.set_ylim(y_min - margin, y_max + margin)

    plt.tight_layout()
    plt.show()


# Call the function with greyscale
plot_transition_probabilities(greyscale=True)


# Chi-squared test
n_states = counts_old.shape[0]
chi_squared = 0

for i in range(n_states):
    n1 = counts_old[i].sum()
    n2 = counts_new[i].sum()
    n_total = n1 + n2

    if n1 == 0 or n2 == 0:
        continue  # skip rows with no transitions

    # Empirical probabilities
    p1 = counts_old[i] / n1
    p2 = counts_new[i] / n2
    p_pool = (counts_old[i] + counts_new[i]) / n_total

    # Chi-squared component for this row
    for j in range(n_states):
        if p_pool[j] == 0:
            continue  # skip divisions by 0
        diff = p1[j] - p2[j]
        weight = 1 / (1/n1 + 1/n2)
        chi_squared += (diff**2 / p_pool[j]) * weight

# Degrees of freedom: (k - 1) * k = 20 for 5 states
df = (n_states - 1) * n_states
p_value = 1 - chi2.cdf(chi_squared, df)

# Print result
print(f"Chi-squared test statistic: {chi_squared:.3f}")
print(f"Degrees of freedom: {df}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("→ Reject H₀: The transition matrices are significantly different.")
else:
    print("→ Fail to reject H₀: No significant difference in transition matrices.")


def plot_nao_kde_over_time(greyscale=False):
    """
    Plots Kernel Density Estimates of Winter NAO Index over time.
    
    Parameters:
    - greyscale (bool): If True, use greyscale colors (default: False)
    """
    # Prepare data
    nao_winter_plot = nao_index_winter.copy()
    nao_winter_plot.index = pd.to_datetime(nao_winter_plot.index)
    nao_winter_plot = nao_winter_plot.reset_index()
    nao_winter_plot = nao_winter_plot[nao_winter_plot['date'].dt.year < 2020]
    nao_winter_plot['period_5yr'] = (nao_winter_plot['date'].dt.year // 5) * 5

    # Get sorted list of 5-year periods
    periods = sorted(nao_winter_plot['period_5yr'].unique())

    # Define colormap based on greyscale parameter
    if greyscale:
        # Greyscale colormap from light grey to dark grey
        grey_cmap = LinearSegmentedColormap.from_list("grey_scale", ["#b0b0b0", "#202020"])
        color_range = np.linspace(0, 1, len(periods))
        colors = [grey_cmap(c) for c in color_range]
    else:
        # Orange to red colormap
        orange_red_cmap = LinearSegmentedColormap.from_list("orange_red", ["orange", "red"])
        color_range = np.linspace(0, 1, len(periods))
        colors = [orange_red_cmap(c) for c in color_range]

    # Set up the plot
    plt.figure(figsize=(12, 8), dpi=100, facecolor='white')

    # Plot KDEs by 5-year bins
    for i, period in enumerate(periods):
        subset = nao_winter_plot[nao_winter_plot['period_5yr'] == period]
        if len(subset) > 1:
            sns.kdeplot(
                data=subset,
                x='nao_index_cdas',
                label=f'{period}-{period+4}',
                color=colors[i],
                alpha=0.7,
                linewidth=1 + (i / len(periods))  # thicker as time progresses
            )

    # Format plot
    plt.title('Kernel Density Estimates of Winter NAO Index over time')
    plt.xlabel('NAO Index')
    plt.ylabel('Density')
    plt.legend(title='Period', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Call the function with greyscale
plot_nao_kde_over_time(greyscale=True)