#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 15:53:04 2025

@author: admin
"""

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

## test for equality daily NAO index

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
ks_daily_nao = ks_2samp(nao_index_winter_new.nao_index_cdas, nao_index_winter_old.nao_index_cdas)
print(ks_daily_nao)

## test for equality monthly NAO index
nao_index = pd.read_csv('../data_persistence/norm.monthly.nao.cdas.z500.19500101_current.csv')
nao_index['date'] = pd.to_datetime(nao_index[['year', 'month', 'day']])
nao_index.set_index('date', inplace=True)
nao_index.drop(columns=['year', 'month', 'day'], inplace=True)        
nao_index = nao_index.replace(np.nan, 0)
nao_index.columns = ['nao_index_cdas']
nao_index_winter = nao_index.loc[nao_index.index.month.isin([12,1,2])]
nao_index_winter_old = nao_index_winter.loc[nao_index_winter.index.year <= 1979].drop_duplicates()

nao_index_winter_new = nao_index_winter.loc[(nao_index_winter.index.year >= 1990) & (nao_index_winter.index.year < 2020)].drop_duplicates()

#test for equality of the nao index in winter
ks_monthly_nao = ks_2samp(nao_index_winter_new.nao_index_cdas, nao_index_winter_old.nao_index_cdas)
print(ks_monthly_nao)


## test for equality monthly SCAND index
nao_index = pd.read_csv('../data_persistence/scand_index.csv')
nao_index['date'] = pd.to_datetime(nao_index[['year', 'month', 'day']])
nao_index.set_index('date', inplace=True)
nao_index.drop(columns=['year', 'month', 'day'], inplace=True)        
nao_index = nao_index.replace(np.nan, 0)
nao_index.columns = ['nao_index_cdas']
nao_index_winter = nao_index.loc[nao_index.index.month.isin([12,1,2])]
nao_index_winter_old = nao_index_winter.loc[nao_index_winter.index.year <= 1979].drop_duplicates()

nao_index_winter_new = nao_index_winter.loc[(nao_index_winter.index.year >= 1990) & (nao_index_winter.index.year < 2020)].drop_duplicates()

#test for equality of the nao index in winter
ks_monthly_scand = ks_2samp(nao_index_winter_new.nao_index_cdas, nao_index_winter_old.nao_index_cdas)
print(ks_monthly_scand)


## test for equality monthly AMO index
nao_index = pd.read_csv('../data_persistence/amo_daily.csv')
nao_index['date'] = pd.to_datetime(nao_index[['year', 'month', 'day']])
nao_index.set_index('date', inplace=True)
nao_index.drop(columns=['year', 'month', 'day'], inplace=True)        
nao_index = nao_index.replace(np.nan, 0)
nao_index.columns = ['nao_index_cdas']
nao_index_winter = nao_index.loc[nao_index.index.month.isin([12,1,2])]
nao_index_winter_old = nao_index_winter.loc[nao_index_winter.index.year <= 1979].drop_duplicates()

nao_index_winter_new = nao_index_winter.loc[(nao_index_winter.index.year >= 1990) & (nao_index_winter.index.year < 2020)].drop_duplicates()

#test for equality of the nao index in winter
ks_monthly_AMO = ks_2samp(nao_index_winter_new.nao_index_cdas, nao_index_winter_old.nao_index_cdas)
print(ks_monthly_AMO)

