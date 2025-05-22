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
import numpy as np
import numpy as np
from scipy.stats import chi2

nao_index = pd.read_csv('../data_persistence/norm.daily.nao.cdas.z500.19500101_current.csv')
nao_index['date'] = pd.to_datetime(nao_index[['year', 'month', 'day']])
nao_index.set_index('date', inplace=True)
nao_index.drop(columns=['year', 'month', 'day'], inplace=True)        
nao_index = nao_index.replace(np.nan, 0)
nao_index.columns = ['nao_index_cdas']
nao_index_winter = nao_index.loc[nao_index.index.month.isin([12,1,2])]
nao_index_winter_old = nao_index_winter.loc[nao_index_winter.index.year <= 1979]

nao_index_winter_new = nao_index_winter.loc[(nao_index_winter.index.year >= 1990) & (nao_index_winter.index.year < 2020)]

#test for equality of the nao index in winter
ks_2samp(nao_index_winter_new.nao_index_cdas, nao_index_winter_old.nao_index_cdas)

q_labels_old = pd.qcut(nao_index_winter_old.nao_index_cdas, q=5, labels=False) + 1  # labels 1 to 5
counts_old = np.zeros((5, 5))
for t in range(len(q_labels_old) - 1):
    i = q_labels_old[t] - 1
    j = q_labels_old[t+1] - 1
    counts_old[i, j] += 1

# Transition probabilities (row-normalized)
row_sums_old = counts_old.sum(axis=1, keepdims=True)
P_hat_old = counts_old / row_sums_old


q_labels_new = pd.qcut(nao_index_winter_new.nao_index_cdas, q=5, labels=False) + 1  # labels 1 to 5
counts_new = np.zeros((5, 5))
for t in range(len(q_labels_new) - 1):
    i = q_labels_new[t] - 1
    j = q_labels_new[t+1] - 1
    counts_new[i, j] += 1

# Transition probabilities (row-normalized)
row_sums_new = counts_new.sum(axis=1, keepdims=True)
P_hat_new = counts_new / row_sums_new



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
    

quintiles = np.arange(1, 6)
x = np.arange(5)

offset = 0.1  # horizontal shift for side-by-side points

fig, ax = plt.subplots(figsize=(8, 5), facecolor='white', dpi=100)

# Old period (orange) — shifted left
ax.errorbar(
    x - offset, diag_old, yerr=z * SE_old,
    fmt='o', color='orange', capsize=4, label='1950–1980'
)

# New period (red) — shifted right
ax.errorbar(
    x + offset, diag_new, yerr=z * SE_new,
    fmt='o', color='red', capsize=4, label='1990–2020'
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


#Test of the hypothesis that several samples are from the same Markov chain of a given order.
#https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-28/issue-1/Statistical-Inference-about-Markov-Chains/10.1214/aoms/1177707039.full

# Assume counts_old and counts_new are 5x5 matrices
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
    
    
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Ensure datetime index
nao_index_winter = nao_index_winter.copy()
nao_index_winter.index = pd.to_datetime(nao_index_winter.index)

# Reset index
nao_index_winter = nao_index_winter.reset_index()

# Filter data to only include years before 2020
nao_index_winter = nao_index_winter[nao_index_winter['date'].dt.year < 2020]

# Create 5-year period column
nao_index_winter['period_5yr'] = (nao_index_winter['date'].dt.year // 5) * 5

# Get sorted list of 5-year periods
periods = sorted(nao_index_winter['period_5yr'].unique())


# Define a custom colormap from orange to red
orange_red_cmap = LinearSegmentedColormap.from_list("orange_red", ["orange", "red"])

# Generate evenly spaced colors from the new colormap
color_range = np.linspace(0, 1, len(periods))
colors = [orange_red_cmap(c) for c in color_range]


# Set up the plot
plt.figure(figsize=(12, 8), dpi=100, facecolor='white')

# Plot KDEs by 5-year bins
for i, period in enumerate(periods):
    subset = nao_index_winter[nao_index_winter['period_5yr'] == period]
    if len(subset) > 1:
        sns.kdeplot(
            data=subset,
            x='nao_index_cdas',
            label=f'{period}-{period+4}',
            color=colors[i],
            alpha=.7,
            linewidth = 1 + (i / len(periods))  # thicker as time progresses

        )

# Format plot
plt.title('Kernel Density Estimates of Winter NAO Index over time')
plt.xlabel('NAO Index')
plt.ylabel('Density')
plt.legend(title='Period', fontsize=9)
plt.grid(True)
plt.tight_layout()
plt.show()

