#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 17:24:05 2025

@author: admin
"""
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

from QAR_persistence_precip import QAR_precipitation

# Load the NetCDF file
file_path = "../data_persistence/Weather-regimes_attribution_winter_NAtl_Eur.nc"
ds = xr.open_dataset(file_path)

nao_index = pd.read_csv('../data_persistence/norm.daily.nao.cdas.z500.19500101_current.csv')
nao_index['date'] = pd.to_datetime(nao_index[['year', 'month', 'day']])
nao_index.set_index('date', inplace=True)
nao_index.drop(columns=['year', 'month', 'day'], inplace=True)        
nao_index = nao_index.replace(np.nan, 0)
nao_index.columns = ['nao_index_cdas']
nao_index.columns=['regime']
nao_index_winter = nao_index.loc[nao_index.index.month.isin([12,1,2])]

datafalkena = pd.DataFrame([ds['time'].values, ds['Regimes'].values[1, :]]).T
datafalkena.columns = ['date', 'regime']
datafalkena.set_index(pd.DatetimeIndex(datafalkena.date),inplace=True)
datafalkena.drop('date',axis=1,inplace=True)
datafalkena = datafalkena.loc[~((datafalkena.index.month == 2) & (datafalkena.index.day == 29))]
datafalkenawinter = datafalkena.loc[datafalkena.index.month.isin([12,1,2])]


import pandas as pd

# Step 1: Create quantile-based bins for the NAO index (5% intervals)
merged = nao_index_winter.merge(
    datafalkenawinter,
    left_index=True,
    right_index=True,
    suffixes=('_nao', '_falkena')
)
merged = merged.loc[merged.index.year<=2022]
# Create 20 bins (5% steps) using qcut
merged['nao_bin'] = pd.qcut(merged['regime_nao'], q=20, labels=False)

# Optional: label bins more clearly (e.g., 0–5%, 5–10%, ...)
quantile_labels = [f'{i*5}--{(i+1)*5}\\%' for i in range(20)]
merged['nao_bin_label'] = pd.Categorical(
    merged['nao_bin'].map(dict(enumerate(quantile_labels))),
    categories=quantile_labels,
    ordered=True
)

cross_table = pd.crosstab(
    merged['nao_bin_label'],
    merged['regime_falkena']
)


# Step 3: Normalize rows to get percentage distribution per NAO quantile bin
cross_table_percent = cross_table.div(cross_table.sum(axis=1), axis=0) * 100

# Display the result
print("Classic regime distribution (%) within each 5%-quantile bin of NAO index:")
# Round and convert to string, then replace 0.0 with empty string
clean_table = cross_table_percent.round(1).astype(str)

# Export to LaTeX
print(clean_table.to_latex())

