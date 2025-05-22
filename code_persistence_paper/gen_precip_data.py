#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 21:53:56 2025

@author: admin
"""
import numpy as np
import pandas as pd
import os
from QAR_persistence_precip import QAR_precipitation
from statsmodels.tools.sm_exceptions import PerfectSeparationError
import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
from scipy.stats import norm

def read_climate_data(file_path):
    # Initialize variables
    station_name = None
    starting_date = None
    
    # Open the file for reading
    with open(file_path, 'r', encoding='ISO-8859-1') as file:
        # Read the file line by line
        for line in file:
            line = line.strip()  # Remove leading and trailing whitespace
            if line.startswith("This is the blended series of station"):
                station_info = line.split("(")
                if len(station_info) > 1:
                    station_name = station_info[1].split(")")[0].split(",")[-1].strip()[7:]
                break  # Exit the loop after finding station info

        # Open the file again for reading
        # Skip the header lines
        for line in file:
            if line.startswith("STAID"):
                break
        
        # Read the first observation to get the starting date
        for line in file:
            line = line.strip()
            if line:
                data = line.split(",")
                starting_date = data[2][:4]  # Extract the year part of the date
                break  # Exit the loop after finding starting date

    return station_name, starting_date

def map_station_with_city(station_name, file_name):
    # Open the file for reading
    with open(file_name, 'r', encoding='ISO-8859-1') as file:
        # Skip the header lines
        next(file)
        next(file)

        # Read the file line by line
        for line in file:
            line = line.strip()  # Remove leading and trailing whitespace
            if line:
                # Extract station information
                station_data = line.split(",")
                if len(station_data) >= 5:  # Ensure there are enough elements in the list
                    current_station_name = station_data[0].strip()
                    if current_station_name == station_name:
                        city_name = station_data[1].strip()
                        latitude = station_data[3].strip()
                        longitude = station_data[4].strip()
                        return city_name, latitude, longitude

    return None, None, None  # Return None if station not found


# Define function to assign seasons
def get_season(month):
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    elif month in [9, 10, 11]:
        return 'autumn'

###### DATA FOR FIGURE 1


#load only those stations that are used in Fig 5 as well
start_date = 1950
start_year_old = start_date
end_year_old = start_date + 30
start_year_new = 1990
end_year_new = start_year_new + 30
drop_na_larger_than = 0.05


folder_path = '../data_persistence/ECA_blend_rr/'
lendata = len(np.sort(os.listdir(folder_path))[:-4])
lat_long = pd.DataFrame(np.zeros((lendata, 5)))
lat_long[:] = np.nan
for (i, file_name) in enumerate(np.sort(os.listdir(folder_path))[1:-4]):
    station_name, starting_date = read_climate_data(folder_path + file_name)
    city_name, latitude, longitude = map_station_with_city(station_name, folder_path + 'stations.txt')
    if type(starting_date) != type(None):
        if int(starting_date) <= start_date:
                lat_long.iloc[i,:] = [file_name, station_name, latitude, longitude, city_name]

# now for the selected stations calculate the statistics for dec 23-feb 24 
start_date = 1950
start_year_old = start_date
end_year_old = start_date + 30
start_year_new = 1950
end_year_new = start_year_new + 70
drop_na_larger_than = 0.05

df = lat_long.dropna()  
df.columns =  ['file_name', 'STAID', 'latitude', 'longitude', 'city_name']
df_results = pd.DataFrame(np.zeros((len(df), 12)), columns=['STANAME', 'STAID', 'latitude', 'longitude', 
                                                            'maxStreak', 'total_rain_acc', 'percentage_rainy_days_upperquintile', 'percentage_rainy_days_NAOplus', \
                                                                'nao_upper_quintile_acc', 'nao_pos_acc', 'total_precip_acc_yearly', 'percentage_rainy_days_total'])
df_results[:] = np.nan
for (i, file_name) in enumerate(df.file_name):
    print(f'\rCurrently calculating station {i+1} out of {len(df.file_name)}', end='')
    
    try:
        test = QAR_precipitation(sFile=file_name, dropna=drop_na_larger_than,
                       oldend = str(end_year_old) + '-', oldstart=str(start_year_old) + '-', 
                       newend = str(end_year_new) + '-', newstart=str(start_year_new) + '-', include_nao=True
                      )

        test.prepare_data()  
        # Generate example binary time series data for test.old

        # Generate example binary time series data for test.new
        y_prec_new = (test.new.Temp >= 5) * 1
        data_new = pd.DataFrame(y_prec_new, columns=['Temp'])
        data_new.columns=['rainy_day']
        data_new['nao_index_cdas'] = test.new.nao_index_cdas
        data_new['rain_mm'] = test.new.Temp
        data_new = data_new[data_new != -9999]
        
        # Assign season to each row
        data_new['season'] = data_new.index.month.isin([12])#data_new.index.month.map(get_season).values
        data_winter_new = data_new.loc[data_new.season == True] #& (data_new.index >=pd.Timestamp('2023-03-01'))]
        total_precip_acc = data_winter_new['rain_mm'].sum()
        total_precip_acc /= data_new.index.year[-1] - data_new.index.year[0] + 1
        iT = len(data_winter_new)
        # Calculate the number of rainy days (rainy_day == 1) in the filtered data
        rainy_days_count = data_winter_new['rainy_day'].sum()
        # Calculate the percentage of rainy days
        if iT > 0:
            percentage_rainy_days_total = (rainy_days_count / iT) * 100
        else:
            percentage_rainy_days_total = 0
        
        # Filter the rows where nao_index_cdas >= 0.9492
        filtered_data = data_winter_new[data_winter_new['nao_index_cdas'] >= 0.9492]
        
        # Calculate the total number of filtered rows
        total_filtered = len(filtered_data)
        
        # Calculate the number of rainy days (rainy_day == 1) in the filtered data
        rainy_days_count = filtered_data['rainy_day'].sum()
        nao_upper_quintile_acc = filtered_data['rain_mm'].sum()
        nao_upper_quintile_acc /= data_new.index.year[-1] - data_new.index.year[0] + 1

        # Calculate the percentage of rainy days
        if total_filtered > 0:
            percentage_rainy_days_upperquintile = (rainy_days_count / total_filtered) * 100
        else:
            percentage_rainy_days_upperquintile = 0
        

            
        # Filter the rows where nao_index_cdas >= 0.9492
        filtered_data = data_winter_new[data_winter_new['nao_index_cdas'] >= 0.]
        
        # Calculate the total number of filtered rows
        total_filtered = len(filtered_data)
        
        # Calculate the number of rainy days (rainy_day == 1) in the filtered data
        rainy_days_count = filtered_data['rainy_day'].sum()
        
        # Calculate the percentage of rainy days
        if total_filtered > 0:
            percentage_rainy_days_NAOplus = (rainy_days_count / total_filtered) * 100
        else:
            percentage_rainy_days_NAOplus = 0
        nao_pos_acc = filtered_data['rain_mm'].sum()
        nao_pos_acc /= data_new.index.year[-1] - data_new.index.year[0] + 1

        # Identify streaks of consecutive ones in the 'Temp' column
        data_winter_new.loc[:, 'streak'] = (data_winter_new['rainy_day'] != data_winter_new['rainy_day'].shift()).cumsum()
        streaks = data_winter_new[data_winter_new['rainy_day'] == 1].groupby('streak').size()
        
        # Get the maximum streak of ones
        max_streak = streaks.max()
        acc_rainfall = np.sum(data_winter_new.rain_mm)
        
        df_results.iloc[i, :] = [df.city_name.iloc[i], df.STAID.iloc[i], df.latitude.iloc[i], df.longitude.iloc[i], \
                                 max_streak, acc_rainfall, percentage_rainy_days_upperquintile, percentage_rainy_days_NAOplus, \
                                     nao_upper_quintile_acc, nao_pos_acc, total_precip_acc, percentage_rainy_days_total]
    except (ValueError, np.linalg.LinAlgError, PerfectSeparationError) as e: 
        pass 

df_results_fig1 = df_results.dropna().set_index('STANAME')
#df_results_fig1 = df_results_fig1.drop(['IZANA','ELAT', 'ELAT-1', 'STA. CRUZ DE TENERIFE', 'TENERIFE/LOS RODEOS'],axis=0)
#df_results_fig1.to_csv('/Users/admin/Documents/PhD/persistence/data_persistence/results_precipitation_' + str(start_date) + 'Fig1_dec_1950_2020.csv')


##### DATA FOR FIG 4A
start_date = 1950
start_year_old = start_date
end_year_old = start_date + 30
start_year_new = 1990
end_year_new = start_year_new + 30
drop_na_larger_than = 0.05
alpha = 0.05
z_score = norm.ppf(1 - alpha / 2)
pattern = 'AMO'

folder_path = '../data_persistence/ECA_blend_rr/'
lendata = len(np.sort(os.listdir(folder_path))[:-4])
lat_long = pd.DataFrame(np.zeros((lendata, 5)))
lat_long[:] = np.nan
for (i, file_name) in enumerate(np.sort(os.listdir(folder_path))[1:-4]):
    station_name, starting_date = read_climate_data(folder_path + file_name)
    city_name, latitude, longitude = map_station_with_city(station_name, folder_path + 'stations.txt')
    if type(starting_date) != type(None):
        if int(starting_date) <= start_date:
                lat_long.iloc[i,:] = [file_name, station_name, latitude, longitude, city_name]

df = lat_long.dropna()  
df.columns =  ['file_name', 'STAID', 'latitude', 'longitude', 'city_name']
df_results = pd.DataFrame(np.zeros((len(df), 6)), columns=['STANAME', 'STAID', 'latitude', 'longitude', 
                                                            'mean_diff_winter', 'hit'])
df_results[:] = np.nan
for (i, file_name) in enumerate(df.file_name):
    print(f'\rCurrently calculating station {i+1} out of {len(df.file_name)}', end='')
    
    try:
        test = QAR_precipitation(sFile=file_name, dropna=drop_na_larger_than,
                       oldend = str(end_year_old) + '-', oldstart=str(start_year_old) + '-', 
                       newend = str(end_year_new) + '-', newstart=str(start_year_new) + '-', include_nao=True, pattern = pattern)
        test.prepare_data()  
        # Generate example binary time series data for test.old
        y_prec_old = (test.old.Temp >= 5) * 1
        data_old = pd.DataFrame(y_prec_old, columns=['Temp'])
        data_old['nao_index_cdas'] = test.old.nao_index_cdas
        
        # Generate example binary time series data for test.new
        y_prec_new = (test.new.Temp >= 5) * 1
        data_new = pd.DataFrame(y_prec_new, columns=['Temp'])
        data_new['nao_index_cdas'] = test.new.nao_index_cdas
        
        # Assign season to each row
        data_old['season'] = data_old.index.month.map(get_season).values
        data_new['season'] = data_new.index.month.map(get_season).values
        
        
        data_winter_new, data_winter_old = data_new.loc[data_new.season == 'winter'],  data_old.loc[data_old.season == 'winter']
        if test.pattern == 'NAO':
            mask_winter_new = data_winter_new.nao_index_cdas.shift(1) > np.quantile(data_winter_new.nao_index_cdas, .8)
            mask_winter_old = data_winter_old.nao_index_cdas.shift(1) > np.quantile(data_winter_old.nao_index_cdas, .8)
        elif test.pattern == 'SCAND':
            mask_winter_new = data_winter_new.nao_index_cdas.shift(1) < np.quantile(data_winter_new.nao_index_cdas, .2)
            mask_winter_old = data_winter_old.nao_index_cdas.shift(1) < np.quantile(data_winter_old.nao_index_cdas, .2)
        elif test.pattern == 'AMO':
            mask_winter_new = data_winter_new.nao_index_cdas.shift(1) > np.quantile(data_winter_new.nao_index_cdas, .8)
            mask_winter_old = data_winter_old.nao_index_cdas.shift(1) > np.quantile(data_winter_old.nao_index_cdas, .8)
        subset_old = data_winter_old.loc[mask_winter_old]
        subset_new = data_winter_new.loc[mask_winter_new]

        p_rain_cond_nao_new_winter = subset_new.Temp.mean()
        p_rain_cond_nao_old_winter = subset_old.Temp.mean()
        diff_winter_unc = p_rain_cond_nao_new_winter - p_rain_cond_nao_old_winter
        
        std_temp_old, n_old = subset_old.Temp.std(), subset_old.shape[0]
        se_old = std_temp_old / np.sqrt(n_old) if n_old > 0 else 0
        ci_low_old = p_rain_cond_nao_old_winter - z_score * se_old
        ci_high_old = p_rain_cond_nao_old_winter + z_score * se_old

        std_temp_new, n_new = subset_new.Temp.std(), subset_new.shape[0]
        se_new = std_temp_new / np.sqrt(n_new) if n_new > 0 else 0
        ci_low_new = p_rain_cond_nao_new_winter - z_score * se_new
        ci_high_new = p_rain_cond_nao_new_winter + z_score * se_new
        no_overlap = (ci_high_old < ci_low_new) or (ci_high_new < ci_low_old)

        df_results.iloc[i, :] = [df.city_name.iloc[i], df.STAID.iloc[i], df.latitude.iloc[i], df.longitude.iloc[i], \
                                 diff_winter_unc, no_overlap]
    except (ValueError, np.linalg.LinAlgError, PerfectSeparationError) as e: 
        pass 

df_results_4a = df_results.dropna().set_index('STANAME')
#df_results_4a = df_results_4a.drop(['IZANA','ELAT', 'ELAT-1', 'STA. CRUZ DE TENERIFE', 'TENERIFE/LOS RODEOS'],axis=0)
#df_results_4a.to_csv('../data_persistence/results_precipitation_' + str(start_date) + 'WithUncProbabilities_hits_AMO_high.csv')



###### DATA FOR FIG 4B

start_year_old = 1950
end_year_old = 1980
start_year_new = 1950
end_year_new = 2020 
drop_na_larger_than = 0.05


df = lat_long.dropna()  
df.columns =  ['file_name', 'STAID', 'latitude', 'longitude', 'city_name']
df_results = pd.DataFrame(np.zeros((len(df), 12)), columns=['STANAME', 'STAID', 'latitude', 'longitude', 
                                                            'maxStreak', 'rain_acc', 'percentage_rainy_days_upperquintile', 'percentage_rainy_days_NAOplus', \
                                                                'nao_upper_quintile_acc', 'nao_pos_acc', 'total_precip_acc', 'percentage_rainy_days_total'])
df_results[:] = np.nan
for (i, file_name) in enumerate(df.file_name):
    print(f'\rCurrently calculating station {i+1} out of {len(df.file_name)}', end='')
    
    try:
        test = QAR_precipitation(sFile=file_name, dropna=drop_na_larger_than,
                       oldend = str(end_year_old) + '-', oldstart=str(start_year_old) + '-', 
                       newend = str(end_year_new) + '-', newstart=str(start_year_new) + '-', include_nao=True
                      )

        test.prepare_data()  
        # Generate example binary time series data for test.old

        # Generate example binary time series data for test.new
        y_prec_new = (test.new.Temp >= 5) * 1
        data_new = pd.DataFrame(y_prec_new, columns=['Temp'])
        data_new.columns=['rainy_day']
        data_new['nao_index_cdas'] = test.new.nao_index_cdas
        data_new['rain_mm'] = test.new.Temp
        data_new = data_new[data_new != -9999]
        
        # Assign season to each row
        data_new['season'] = data_new.index.month.map(get_season).values
        data_winter_new = data_new.loc[(data_new.season == 'winter')]
        total_precip_acc = data_winter_new['rain_mm'].sum()
        
        iT = len(data_winter_new)
        # Calculate the number of rainy days (rainy_day == 1) in the filtered data
        rainy_days_count = data_winter_new['rainy_day'].sum()
        # Calculate the percentage of rainy days
        if iT > 0:
            percentage_rainy_days_total = (rainy_days_count / iT) * 100
        else:
            percentage_rainy_days_total = 0
        
        # Filter the rows where nao_index_cdas >= 0.9492
        filtered_data = data_winter_new[data_winter_new['nao_index_cdas'] >= 0.9492]
        
        # Calculate the total number of filtered rows
        total_filtered = len(filtered_data)
        
        # Calculate the number of rainy days (rainy_day == 1) in the filtered data
        rainy_days_count = filtered_data['rainy_day'].sum()
        nao_upper_quintile_acc = filtered_data['rain_mm'].sum()
        # Calculate the percentage of rainy days
        if total_filtered > 0:
            percentage_rainy_days_upperquintile = (rainy_days_count / total_filtered) * 100
        else:
            percentage_rainy_days_upperquintile = 0
        

            
        # Filter the rows where nao_index_cdas >= 0.9492
        filtered_data = data_winter_new[data_winter_new['nao_index_cdas'] >= 0.]
        
        # Calculate the total number of filtered rows
        total_filtered = len(filtered_data)
        
        # Calculate the number of rainy days (rainy_day == 1) in the filtered data
        rainy_days_count = filtered_data['rainy_day'].sum()
        
        # Calculate the percentage of rainy days
        if total_filtered > 0:
            percentage_rainy_days_NAOplus = (rainy_days_count / total_filtered) * 100
        else:
            percentage_rainy_days_NAOplus = 0
        nao_pos_acc = filtered_data['rain_mm'].sum()

        # Identify streaks of consecutive ones in the 'Temp' column
        data_winter_new.loc[:, 'streak'] = (data_winter_new['rainy_day'] != data_winter_new['rainy_day'].shift()).cumsum()
        streaks = data_winter_new[data_winter_new['rainy_day'] == 1].groupby('streak').size()
        
        # Get the maximum streak of ones
        max_streak = streaks.max()
        acc_rainfall = np.sum(data_winter_new.rain_mm)
        
        df_results.iloc[i, :] = [df.city_name.iloc[i], df.STAID.iloc[i], df.latitude.iloc[i], df.longitude.iloc[i], \
                                 max_streak, acc_rainfall, percentage_rainy_days_upperquintile, percentage_rainy_days_NAOplus, \
                                     nao_upper_quintile_acc, nao_pos_acc, total_precip_acc, percentage_rainy_days_total]
    except (ValueError, np.linalg.LinAlgError, PerfectSeparationError) as e: 
        pass 

df_results_4b = df_results.dropna().set_index('STANAME')
#df_results_4b = df_results_4b.drop(['IZANA','ELAT', 'ELAT-1', 'STA. CRUZ DE TENERIFE', 'TENERIFE/LOS RODEOS'],axis=0)
#df_results_4b.to_csv('../data_persistence/results_precipitation_1950WithUncProbabilities.csv')

