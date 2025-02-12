#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:46:31 2024

@author: admin
"""

import numpy as np
import statsmodels.api as sm
from QAR import QAR_temperature

# Set parameters
sCity = 'PRAHA-KLEMENTINUM'
quantile = 0.95  # Quantile level
oldstart = '1906-'
oldend = '1936-'

# Define ranges for num_terms_level and num_terms_pers
num_terms_pers_values = [0, 1, 2, 3]
num_terms_level_values = [0, 1, 2, 3]

# Data preparation
test = QAR_temperature(sCity=sCity, num_terms_pers=num_terms_pers_values[0], fTau=quantile, num_terms_level=num_terms_level_values[0])
test.prepare_data()
n = len(test.new) - 1

# Function to calculate BIC for given num_terms_level and num_terms_pers
def calculate_bic(test, quantile, n):
    residuals = test.new.Temp.values[1:] - test.results()[0].predict(test.mX_new)
    quantile_loss = np.sum(np.where(residuals >= 0, quantile * residuals, (1 - quantile) * np.abs(residuals)))
    b = quantile_loss / n
    log_likelihood = - n * np.log(b)
    k = len(test.results()[0].params)  # Number of parameters
    return -2 * log_likelihood + k * np.log(n)

# Test different configurations
min_bic = float('inf')
best_config = None
results_table = np.zeros((4, 4))  # 4x4 square table for BIC values

for i, num_terms_pers in enumerate(num_terms_pers_values):
    for j, num_terms_level in enumerate(num_terms_level_values):
        test = QAR_temperature(sCity='MAASTRICHT', num_terms_pers=num_terms_pers, fTau=quantile, num_terms_level=num_terms_level)
        test.prepare_data()
        bic_value = calculate_bic(test, quantile, n)
        results_table[i, j] = bic_value

# Find the best configuration
min_bic = np.min(results_table)
best_indices = np.unravel_index(np.argmin(results_table), results_table.shape)
best_config = (num_terms_pers_values[best_indices[0]], num_terms_level_values[best_indices[1]])

# Output the best configuration
print(f"Best configuration for BIC minimization:")
print(f"Number of terms in period (num_terms_pers): {best_config[0]}")
print(f"Number of terms in level (num_terms_level): {best_config[1]}")
print(f"Minimal BIC: {min_bic:.3f}")

# Display the results table
print("\nResults Table:")
print("{:<15} {:<15} {:<10}".format("num_terms_pers", "num_terms_level", "BIC"))
print("="*40)
for i, num_terms_pers in enumerate(num_terms_pers_values):
    for j, num_terms_level in enumerate(num_terms_level_values):
        print("{:<15} {:<15} {:<10.3f}".format(num_terms_pers, num_terms_level, results_table[i, j]))
