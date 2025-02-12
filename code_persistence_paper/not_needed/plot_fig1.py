#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 13:45:12 2024

@author: admin
"""
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import griddata
from matplotlib.colors import TwoSlopeNorm
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from QAR_persistence_precip import QAR_precipitation
from statsmodels.tools.sm_exceptions import PerfectSeparationError
import copy
import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

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

# Create lag features for each season, including shifting nao_index_cdas and creating dummies
def create_lagged_features(data, n_lags):
    lagged_data = data.copy()
    
    # Create lagged features for 'Temp'
    for lag in range(1, n_lags + 1):
        lagged_data[f'lag_{lag}'] = lagged_data['Temp'].shift(lag)
    
    # Create categorical indicators for 'nao_index_cdas'
    lagged_data['nao_index_cdas'] = lagged_data['nao_index_cdas'].shift(1)  # Shift nao_index_cdas
    quantiles = [0, 0.5, 0.8, 1.0]  # Define quantiles for categorical conversion
    lagged_data['nao_index_cdas_cat'] = pd.qcut(lagged_data['nao_index_cdas'], quantiles, labels=False)
    
    # Convert categorical indicator into dummy variables
    nao_index_dummies = pd.get_dummies(lagged_data['nao_index_cdas_cat'], prefix='nao_index_cat', drop_first=True)
    lagged_data = pd.concat([lagged_data, nao_index_dummies], axis=1)
    
    # Drop original categorical indicator column and rows with NaN values due to lagging and shifting
    lagged_data = lagged_data.drop(['nao_index_cdas', 'nao_index_cdas_cat'], axis=1).dropna()
    
    return lagged_data

# Function to fit AR logistic regression for all winter months
def fit_ar_logistic_regression(data, n_lags, months=[12, 1, 2]):
    winter_data = data[data.index.month.isin(months)].copy()
    seasonal_data = create_lagged_features(winter_data, n_lags)

    # Prepare predictors (X) and response variable (y)
    X = seasonal_data[[f'lag_{lag}' for lag in range(1, n_lags + 1)] + list(seasonal_data.filter(like='nao_index_cat').columns)].values
    y = seasonal_data['Temp'].values

    # Add a constant term for the intercept
    X = sm.add_constant(X)
    
    # Standardize the predictors (optional but recommended)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[:, 1:])  # Exclude the constant column
    X_scaled = np.column_stack((np.ones(X_scaled.shape[0]), X_scaled))  # Add back the constant column
    
    # Fit the autoregressive logistic regression model
    model = sm.Logit(y, X)
    result = model.fit(disp=0)

    return result


def plot_combined_heatmap(df_results, sTypes, titles):
    """
    Generate a combined heatmap with multiple subplots based on specified sTypes.

    Args:
        df_results: DataFrame containing data.
        sTypes: List of `sType` values to plot (e.g., ['total_precip_acc', 'percentage_rainy_days_total']).
        titles: List of titles for each subplot (same order as sTypes).
    """
    try:
        df_results.set_index('STANAME', inplace=True)
        df_results = df_results.drop(['IZANA', 'ELAT', 'ELAT-1', 'STA. CRUZ DE TENERIFE', 'TENERIFE/LOS RODEOS'], axis=0)
        df_results.reset_index(inplace=True)
    except KeyError:
        pass

    # Initialize figure and subplots
    n_panels = len(sTypes)
    fig, axes = plt.subplots(1, n_panels, figsize=(7.5 * n_panels, 7), dpi=200)

    # Ensure axes is iterable
    if n_panels == 1:
        axes = [axes]

    # Basemap setup
    def create_map(ax):
        m = Basemap(projection='merc', llcrnrlat=30, urcrnrlat=72, llcrnrlon=-20, urcrnrlon=45, resolution='l', ax=ax)
        m.drawcoastlines()
        m.drawcountries()
        return m

    # Convert latitudes and longitudes to x, y coordinates
    latitude = df_results['latitude'].apply(lambda x: float(x.split(':')[0]) + float(x.split(':')[1]) / 60 + float(x.split(':')[2]) / 3600)
    longitude = df_results['longitude'].apply(lambda x: float(x.split(':')[0]) + float(x.split(':')[1]) / 60 + float(x.split(':')[2]) / 3600)

    # Special locations
    special_locations = [
        {'name': 'Nottingham', 'lat': 52.950001, 'lon': -1.150000},
        {'name': 'Abbeville', 'lat': 50.1054, 'lon': 1.8332},
        {'name': 'De Bilt', 'lat': 52.1089, 'lon': 5.1805},
        {'name': 'Hettstedt', 'lat': 51.6519, 'lon': 11.5077},
        {'name': 'Nykobing Falster', 'lat': 54.7691, 'lon': 11.8743}
    ]

    # Plotting helper function
    def plot_data(ax, sType, title):
        m = create_map(ax)
        x, y = m(list(longitude), list(latitude))

        # Determine colormap normalization based on sType
        if sType in ['percentage_rainy_days_total', 'percentage_rainy_days_upperquintile', 'percentage_rainy_days_NAOplus']:
            vmin, vcenter, vmax = 0, 50, 100
            data_col = sType
            data = df_results[data_col]
        elif sType == 'total_precip_acc':
            vmin, vcenter, vmax = 0, 100, 200
            data_col = sType
            data = df_results[data_col] / 10
        elif sType == 'nao_upper_quintile_acc':
            vmin, vcenter, vmax = 0, 50, 100
            data_col = 'nao_upper_quintile_acc'
            data = df_results[data_col] / 10  # Rescale
        elif sType == 'nao_pos_acc':
            vmin, vcenter, vmax = 0, 50, 100
            data_col = 'nao_pos_acc'
            data = df_results[data_col] / 10  # Rescale
        else:
            raise ValueError(f"Unknown sType: {sType}")

        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        cmap = plt.cm.YlOrRd
        sc = m.scatter(x, y, c=data, cmap=cmap, norm=norm, s=60, marker='o', alpha=1)

        # Plot numbered, fully blue filled circles with white text for special locations
        for i, location in enumerate(special_locations, start=1):
            loc_x, loc_y = m(location['lon'], location['lat'])
            # Fully blue filled circle marker
            m.scatter(loc_x, loc_y, facecolors='blue', edgecolors='blue', s=300, marker='o', alpha=1)
            # White text in the center
            ax.text(loc_x, loc_y, str(i), fontsize=18, ha='center', va='center', color='white', fontweight='bold')
        
        # Draw parallels and meridians
        parallels = np.arange(30, 81, 10)
        m.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=8, linewidth=0)
        meridians = np.arange(-20, 60, 20)
        m.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=8, linewidth=0)


        ax.set_title(title)
        return sc

    # Plot each panel
    for ax, sType, title in zip(axes, sTypes, titles):
        sc = plot_data(ax, sType, title)
        cbar = fig.colorbar(sc, ax=ax, orientation='vertical', shrink=0.85)
        if sType in ['percentage_rainy_days_total', 'percentage_rainy_days_upperquintile', 'percentage_rainy_days_NAOplus']:
            cbar.set_label('Fraction of days with more than 0.5mm precipitation (in %)')
        else:
            cbar.set_label('Accumulated precipitation in mm')

    plt.tight_layout()
    plt.show()

start, end = '2023-11-30', '2024-01-04'
# Initialize test objects for each city
test_objects = QAR_precipitation(sCity='DE BILT', fTau=.95, use_statsmodels=True, include_nao=True, oldstart='1990-', oldend='2020-',newend='2025-') 

# Prepare data for each city
for test in test_objects:
    test.prepare_data()

# Setup your figure
fig, ax = plt.subplots(dpi=200, figsize=(10, 6))

# Example NAO data for one city (can be used as a reference for x-axis)
test = test_objects[0]
test.new.nao_index_cdas.loc[(test.new.index > start) & (test.new.index < end)].plot(ax=ax, label='NAO index')

# Add lines for y=0 and the 80th percentile
plt.axhline(y=0, color='grey', linestyle='-', linewidth=0.5)
plt.axhline(y=0.95, color='red', linestyle='--', linewidth=1, label='80th percentile NAO')

# Add vertical line for a specific event (e.g., Storm Henk)
plt.axvline(x='2024-01-02', color='black', linestyle='--', linewidth=1, label='Storm Henk')

# Define vertical spacing for each city's rainfall
y_min, y_max = ax.get_ylim()
spacing = 0.1  # Small vertical space between cities



# Add y-axis label
ax.set_ylabel('NAO index value')

# Adjust layout to prevent cutting off labels
plt.tight_layout()

# Show the plot
plt.show()

test = QAR_temperature(sCity='DE BILT', include_nao=True)
test.prepare_data()
newwinter = test.new.nao_index_cdas.loc[test.new.index.month.isin([12,1,2])]
oldwinter = test.old.nao_index_cdas.loc[test.old.index.month.isin([12,1,2])]
# KDE plot
plt.figure(figsize=(10, 6), dpi=120)

sns.kdeplot(oldwinter, color='orange', fill=False, alpha=0.5, linewidth=2, label='1950-1979')
sns.kdeplot(newwinter, color='red', fill=False, alpha=0.5, linewidth=2, label='1990-2020')

# Customizing the plot
plt.xlabel('NAO Index', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.axvline(x=0.693091150856237, color='black', linestyle='--', linewidth=1, label='Mean of NAO index December 2023')
plt.legend(fontsize=10)
plt.grid(alpha=0.3)

# Display the plot
plt.tight_layout()
plt.show()

sTypes = ['percentage_rainy_days_total', 'total_precip_acc']
titles = ['', '']
plot_combined_heatmap(df_results, sTypes, titles)


start_date = 2020
start_year_old = start_date
end_year_old = start_date + 30
start_year_new = 2023
end_year_new = start_year_new + 35
tau = .5
drop_na_larger_than = 0.05

folder_path = '/Users/admin/Downloads/ECA_blend_rr/'
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
        data_new['season'] = data_new.index.month.isin([12])#data_new.index.month.map(get_season).values
        data_winter_new = data_new.loc[(data_new.season == True) & (data_new.index >=pd.Timestamp('2023-03-01'))]
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

df_results = df_results.dropna().set_index('STANAME')
#df_results = df_results.drop(['IZANA','ELAT', 'ELAT-1', 'STA. CRUZ DE TENERIFE', 'TENERIFE/LOS RODEOS'],axis=0)
df_results.to_csv('/Users/admin/Documents/PhD/persistence/data_persistence/results_precipitation_' + str(start_date) + 'Fig1_dec.csv')

plot_heatmap_precip_fig1(df_results, 'winter', sType='percentage_rainy_days_total')
plot_heatmap_precip_fig1(df_results, 'winter', sType='total_precip_acc')
