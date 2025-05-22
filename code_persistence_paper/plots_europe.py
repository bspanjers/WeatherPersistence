#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 13:45:12 2024

@author: admin
"""
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import Normalize
import numpy as np
from QAR import *
from QAR_persistence_precip import QAR_precipitation
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.gridspec as gridspec
import seaborn as sns
import scipy.stats as stats

def analyze_nao_precipitation():
    """
    Function to analyze NAO index, precipitation data, and temperature-based rainfall events.

    Parameters:
    - data_path (str): Path to precipitation dataset CSV file.
    - cities (list): List of city names to analyze.
    - start (str): Start date for NAO time series (YYYY-MM-DD).
    - end (str): End date for NAO time series (YYYY-MM-DD).
    - storm_event (str): Date of a specific storm event to highlight (YYYY-MM-DD).

    Returns:
    - None (Displays plots)
    """
    # Define the path to your dataset
    data_path_2023 = '../data_persistence/results_precipitation_2023Fig1_dec.csv' 
    data_path_1950_2020 = '../data_persistence/results_precipitation_1950Fig1_dec_1950_2020.csv'
    start = '2023-11-30'
    end = '2024-01-04'
    storm_event = pd.Timestamp('2024-01-02')
    
    # Load precipitation data
    df_results_2023 = pd.read_csv(data_path_2023)
    df_results_2023 = df_results_2023.dropna().set_index('STANAME')
    df_results_1950_2020 = pd.read_csv(data_path_1950_2020)
    df_results_1950_2020 = df_results_1950_2020.dropna().set_index('STANAME')
    # Ensure common stations only and consistent row order
    common_index = df_results_2023.index.intersection(df_results_1950_2020.index)
    
    df_results_2023 = df_results_2023.loc[common_index].sort_index()
    df_results_1950_2020 = df_results_1950_2020.loc[common_index].sort_index()
    df_results_2023 = df_results_2023[~df_results_2023.index.duplicated()]
    df_results_1950_2020 = df_results_1950_2020[~df_results_1950_2020.index.duplicated()]
    # Compute the difference
    df_results_diff = df_results_2023.copy()
    df_results_diff.iloc[:, 4:] = df_results_2023.iloc[:, 4:].values - df_results_1950_2020.iloc[:, 4:].values

    # Convert latitude and longitude from DMS to decimal format
    latitude = df_results_diff['latitude'].apply(lambda x: float(x.split(':')[0]) + float(x.split(':')[1]) / 60 + float(x.split(':')[2]) / 3600)
    longitude = df_results_diff['longitude'].apply(lambda x: float(x.split(':')[0]) + float(x.split(':')[1]) / 60 + float(x.split(':')[2]) / 3600)

    # Initialize figure with a gridspec layout
    fig = plt.figure(figsize=(14, 11), dpi=200, facecolor="white")
    gs = gridspec.GridSpec(2, 2, height_ratios=[1,2])  # Top row gets 1/3, bottom row 2/3

    test = QAR_precipitation(sCity='DE BILT', include_nao=True)
    test.prepare_data()
    newwinter = test.new.nao_index_cdas.loc[test.new.index.month.isin([12, 1, 2])]
    oldwinter = test.old.nao_index_cdas.loc[test.old.index.month.isin([12, 1, 2])]

    # Panel 1: KDE plot of NAO index distributions
    ax1 = fig.add_subplot(gs[0, 0])
    oldwinter = oldwinter.dropna().astype(float)
    newwinter = newwinter.dropna().astype(float)
    sns.kdeplot(oldwinter.squeeze(), color='orange',  linewidth=2, label='Winter 1950-1980', ax=ax1)
    sns.kdeplot(newwinter.squeeze(), color='red', linewidth=2, label='Winter 1990-2020', ax=ax1)
    ax1.axvline(x=0.693091150856237, color='black', linestyle='--', linewidth=1, label='Mean of NAO index Dec 2023')
    ax1.set_xlabel('NAO Index', fontsize=10)
    ax1.set_ylabel('Density', fontsize=10)
    ax1.set_title('(a)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    test = QAR_precipitation(sCity='DE BILT', fTau=.95, use_statsmodels=True, include_nao=True, oldstart='1990-', oldend='2020-', newend='2025-')
    test.prepare_data()

    # Panel 2: NAO index time series
    ax2 = fig.add_subplot(gs[0, 1])
    test.new.nao_index_cdas.loc[(test.new.index > start) & (test.new.index < end)].plot(ax=ax2, label='NAO index')
    ax2.axhline(y=0, color='grey', linestyle='-', linewidth=0.5)
    ax2.axvline(x=storm_event, color='black', linestyle='--', linewidth=1, label='Storm Henk')
    ax2.set_ylabel('NAO index value')
    ax2.set_title('(b)', fontsize=12)
    ax2.legend(fontsize=10)

    # Panel 3: Heatmap for percentage_rainy_days_total
    ax3 = fig.add_subplot(gs[1, 0])
    sc3 = plot_heatmap(df_results_diff, latitude, longitude, ax3, 'percentage_rainy_days_total', '(c)')
    cbar3 = fig.colorbar(sc3, ax=ax3, orientation='vertical', shrink=0.85)
    cbar3.set_label('Absolute Difference in Percentage Rainy Days')

    # Panel 4: Heatmap for total_precip_acc
    ax4 = fig.add_subplot(gs[1, 1])
    sc4 = plot_heatmap(df_results_diff, latitude, longitude, ax4, 'total_precip_acc', '(d)')
    cbar4 = fig.colorbar(sc4, ax=ax4, orientation='vertical', shrink=0.85)
    cbar4.set_label('Absolute Difference in Accumulated Precipitation (mm)')

    plt.tight_layout()
    plt.show()
    
def read_climate_data(file_path):
    # Initialize variables
    station_name = None
    starting_date = None
    
    # Open the file for reading
    with open(file_path, 'r') as file:
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
    
def map_station_with_city(station_name, file_name):
    # Open the file for reading
    with open(file_name, 'r') as file:
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


def plot_heatmap(df_results, latitude, longitude, ax, sType, title):
    # Basemap setup
    m = Basemap(projection='merc', 
                llcrnrlat=30, urcrnrlat=72, 
                llcrnrlon=-20, urcrnrlon=40, 
                resolution='l', ax=ax)
    m.drawcoastlines()
    m.drawcountries()

    # Generate longitude and latitude points for plotting
    x, y = m(list(longitude), list(latitude))

    # Define color scale and data for different sTypes
    if sType == 'percentage_rainy_days_total':
        vmin, vcenter, vmax = -20, 5, 30
        data = df_results[sType]
    elif sType == 'total_precip_acc':
        vmin, vcenter, vmax = -500, 400, 1000
        data = df_results[sType] 
    else:
        raise ValueError(f"Unknown sType: {sType}")

    # Normalize the data for plotting
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    cmap = plt.cm.YlOrRd

    # Plot the heatmap
    sc = m.scatter(x, y, c=data, cmap=cmap, norm=norm, s=60, marker='o', alpha=1)
    special_locations = [
        {'name': 'Nottingham', 'lat': 52.950001, 'lon': -1.150000},
        {'name': 'Abbeville', 'lat': 50.1054, 'lon': 1.8332},
        {'name': 'De Bilt', 'lat': 52.1089, 'lon': 5.1805},
        {'name': 'Hettstedt', 'lat': 51.6519, 'lon': 11.5077},
        {'name': 'Nykobing Falster', 'lat': 54.7691, 'lon': 11.8743}
    ]

    # Mark special locations
    for i, location in enumerate(special_locations, start=1):
        loc_x, loc_y = m(location['lon'], location['lat'])
        m.scatter(loc_x, loc_y, facecolors='blue', edgecolors='blue', s=300, marker='o', alpha=1)
        ax.text(loc_x, loc_y, str(i), fontsize=12, ha='center', va='center', color='white', fontweight='bold')

    # Add latitude (parallels) and longitude (meridians) ticks with labels
    parallels = np.arange(30, 81, 10)  # Adjust range and interval as needed
    meridians = np.arange(-20, 41, 10)  # Adjust range and interval as needed

    # Draw parallels and meridians
    m.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=8, linewidth=0.0001)  # Labels on the left
    m.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=8, linewidth=0.0001)  # Labels on the bottom

    # Set title
    ax.set_title(title, fontsize=12)
    return sc

def rolling_window_precipitation(sCity='DE BILT', mm_threshold=5, quantile=0.8, window_size=30, confidence=0.95):
    """
    Performs a rolling window analysis (30-year periods) of the probability of precipitation 
    in the upper quintile of the NAO index.

    Parameters:
    - sCity (str): Name of the city (default: 'DE BILT')
    - mm_threshold (int): Temperature threshold for binary rain data (default: 0.5mm)
    - quantile (float): Upper quintile of NAO index (default: 0.8)
    - window_size (int): Rolling window size in years (default: 30)
    - confidence (float): Confidence level for confidence intervals (default: 0.95)

    Returns:
    - DataFrame: Contains probability of rain, confidence intervals, and upper NAO quintile per window.
    - Generates a plot showing the probability of rain and the NAO upper quintile over time.
    """

    # Constants
    months = [12, 1, 2]  # Winter months (Dec, Jan, Feb)
    z_score = stats.norm.ppf((1 + confidence) / 2)  # Z-score for confidence interval

    # Initialize lists to store results
    prob_rain_upper_nao_30_year = []
    upper_quintile_values = []

    # Define the range of years for the 30-year rolling windows
    start_years = range(1950, 1991)  # Last start year is 1990 (to get 1990-2019)

    # Loop through each 30-year window
    for start_year in start_years:
        end_year = start_year + window_size

        # Define dynamic start and end years for QAR object
        oldstart_str = f'{start_year}-'
        oldend_str = f'{end_year}-'

        # Initialize QAR_precipitation object
        test = QAR_precipitation(sCity=sCity, fTau=.95, use_statsmodels=True, include_nao=True, oldstart=oldstart_str, oldend=oldend_str)
        test.prepare_data()

        # Generate binary time series for precipitation
        y_prec_old = (test.old.Temp >= mm_threshold) * 1
        data_old = pd.DataFrame(y_prec_old, columns=['Temp'])
        data_old['nao_index_cdas'] = test.old.nao_index_cdas

        # Filter for winter months
        data_winter_old = data_old.loc[data_old.index.month.isin(months)]

        # Calculate the upper quintile of the NAO index in the current 30-year window
        upper_quintile_nao = np.quantile(data_winter_old['nao_index_cdas'], quantile)
        upper_quintile_values.append(upper_quintile_nao)

        # Filter the data for the upper NAO quintile
        subset = data_winter_old[data_winter_old['nao_index_cdas'].shift(1) >= upper_quintile_nao]

        # Calculate the probability of rain in the upper NAO quintile
        prob_rain = subset['Temp'].mean()

        # Calculate standard error and confidence interval
        n = len(subset['Temp'])  # Number of observations
        std_temp = subset['Temp'].std()  # Standard deviation
        se_temp = std_temp / np.sqrt(n) if n > 0 else 0  # Standard error
        ci_low = prob_rain - z_score * se_temp if n > 0 else np.nan  # Lower bound
        ci_high = prob_rain + z_score * se_temp if n > 0 else np.nan  # Upper bound

        # Store results
        prob_rain_upper_nao_30_year.append({
            'start_year': start_year,
            'end_year': end_year - 1,  # Inclusive
            'prob_rain': prob_rain,
            'ci_low': ci_low,
            'ci_high': ci_high
        })

    # Convert results into DataFrame
    prob_rain_df = pd.DataFrame(prob_rain_upper_nao_30_year)

    # Plot results
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=200, facecolor='white')

    # Plot probability of rain
    line1, = ax1.plot(prob_rain_df['end_year'].values, prob_rain_df['prob_rain'].values, marker='o', linestyle='-', color='blue', label='Probability of rain (upper NAO quintile)')
    ax1.fill_between(prob_rain_df['end_year'], prob_rain_df['ci_low'], prob_rain_df['ci_high'], color='blue', alpha=0.3, label='95% Confidence Interval')
    ax1.set_xlabel('End Year of 30-Year Window')
    ax1.set_ylabel('Probability of Rain', color='black')
    ax1.grid(True)
    ax1.set_title(f'Probability of Rain in Upper NAO Quintile in {sCity} in Winter (30-Year Windows)')
    ax1.tick_params(axis='y', colors='blue')

    # Add second axis for NAO index upper quintile
    ax2 = ax1.twinx()
    line2, = ax2.plot(prob_rain_df['end_year'].values, upper_quintile_values, marker='s', linestyle='--', color='red', label='80$^{th}$ Percentile of NAO Index')
    ax2.set_ylabel('NAO Index Value', color='black')
    ax2.tick_params(axis='y', colors='red')

    # Merge legends from both axes
    lines = [line1, line2]
    labels = [line1.get_label(), line2.get_label()]
    ax1.legend(lines, labels, loc='upper left')

    # Show plot
    plt.show()



def plot_binomial_probabilities(total_days=21, p1=0.6518518518518519, p2=0.5555555555555556):
    """
    Plots the binomial probability mass function (PMF) and cumulative density function (CDF)
    for two given probabilities over a defined number of trials (days).

    Parameters:
    - total_days (int): Number of trials (e.g., total days in observation period)
    - p1 (float): First probability of success
    - p2 (float): Second probability of success

    Returns:
    - None (Displays the plot)
    """

    x_days = np.arange(1, total_days + 1)  # Number of consecutive hits (successes)

    # Cumulative probabilities (CDF) for both p1 and p2
    cumulative_prob_x_days_1 = stats.binom.cdf(x_days, total_days, p1)
    cumulative_prob_x_days_2 = stats.binom.cdf(x_days, total_days, p2)

    # Binomial probabilities of exactly x hits (PMF)
    prob_x_days_1 = stats.binom.pmf(x_days, total_days, p1)
    prob_x_days_2 = stats.binom.pmf(x_days, total_days, p2)

    # Create the combined figure
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), dpi=200)

    # First subplot: Binomial probability (PMF)
    axs[0].plot(x_days, prob_x_days_1, label=f'p = {np.round(p1,4)}', marker='o', color='red')
    axs[0].plot(x_days, prob_x_days_2, label=f'p = {np.round(p2,4)}', marker='o', color='orange')
    axs[0].set_xlabel('Days with precipitation exceeding 0.5mm')
    axs[0].set_ylabel('Density')
    axs[0].set_xticks(x_days)
    axs[0].grid(True)
    axs[0].legend()
    axs[0].set_title('(a) Probability Density')

    # Second subplot: Cumulative probability (CDF)
    axs[1].plot(x_days, cumulative_prob_x_days_1, label=f'Cumulative p = {np.round(p1,4)}', marker='o', color='red')
    axs[1].plot(x_days, cumulative_prob_x_days_2, label=f'Cumulative p = {np.round(p2,4)}', marker='o', color='orange')
    axs[1].set_xlabel('Days with precipitation exceeding 0.5mm')
    axs[1].set_ylabel('Cumulative Density')
    axs[1].set_xticks(x_days)
    axs[1].grid(True)
    axs[1].legend()
    axs[1].set_title('(b) Cumulative Density')
    axs[1].axhline(y=1-0.1054, color='orange', linestyle='--', linewidth=.75, label='$y=1-0.1054$')
    axs[1].axhline(y=1-0.3634, color='red', linestyle='--', linewidth=.75, label='$y=1-0.3634$')
    
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_combined(df_results_list, sSeason, l_sType=['pers_', 'pers_', 'pers_'], tau_list=[0.05, .5, .95], bSignificance=False, pattern='NAO'):
    fig, axs = plt.subplots(3, 3, figsize=(15, 17), dpi=100, sharey=True, sharex=False, facecolor='white')
    fig.subplots_adjust(hspace=0.1, wspace=0.01)
    
    scatter_plots = []

    # First loop for the first row (NAO-)
    for i, (df_results, sType, tau) in enumerate(zip(df_results_list[:3], l_sType, tau_list)):
        ax = axs[0, i]
        if bSignificance:
            df_plot = df_results[df_results['hit']]
        else:
            df_plot = df_results
        # Create a Basemap of Europe
        m = Basemap(projection='merc', llcrnrlat=30, urcrnrlat=75, llcrnrlon=-20, urcrnrlon=59, resolution='c', ax=ax)

        # Draw coastlines and countries
        m.drawcoastlines()
        m.drawcountries()

        # Convert latitudes and longitudes to x, y coordinates
        latitude = df_plot['latitude'].apply(lambda x: float(x.split(':')[0]) + float(x.split(':')[1])/60 + float(x.split(':')[2])/3600)
        longitude = df_plot['longitude'].apply(lambda x: float(x.split(':')[0]) + float(x.split(':')[1])/60 + float(x.split(':')[2])/3600)
        x, y = m(list(longitude), list(latitude))

        # Define colormap and normalization
        cmap = plt.cm.RdYlGn_r if sType == 'pers_' else plt.cm.RdBu_r
        vmin, vmax = (-0.151, 0.151) if sType == 'pers_' else (-15, 15)
        norm = Normalize(vmin=vmin, vmax=vmax)

        # Plot scattered dots
        if sType == 'pers_':
            sc = m.scatter(x, y, c=df_plot[f'mean_diff_pers_{sSeason}'], cmap=cmap, norm=norm, s=100, marker='o', alpha=.7)
        else: 
            sc = m.scatter(x, y, c=df_plot[f'mean_diff_{sSeason}'], cmap=cmap, norm=norm, s=100, marker='o', alpha=.7)

        scatter_plots.append(sc)

        # Add title with subplot numbering
        ax.set_title(f'({chr(97 + i)})' + ' $\\overline{\\Delta}_{\\phi}(\\tau)$ with' + f' $\\tau$ = {tau}', fontsize=12)

        # Add lat and long labels without gridlines
        meridians = np.arange(-20, 61, 20)
        #m.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=8, linewidth=0)  # labels on the bottom
        if i == 0:
            parallels = np.arange(30, 81, 10)
            m.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=8, linewidth=0.01)  # labels on the left

    # Second loop for the second and third rows (NAO+)
    for i, (df_results, sType, tau) in enumerate(zip(df_results_list[3:], l_sType, tau_list)):
        for j, pm in enumerate(['0', '1']):
            ax = axs[j+1, i]
            if bSignificance:
                hit_col = f"hit{pm}"
                df_plot = df_results[df_results[hit_col].astype(bool)].copy()
            else:
                df_plot = df_results.copy()
            # Create a Basemap of Europe
            m = Basemap(projection='merc', llcrnrlat=30, urcrnrlat=75, llcrnrlon=-20, urcrnrlon=59, resolution='c', ax=ax)

            # Draw coastlines and countries
            m.drawcoastlines()
            m.drawcountries()

            # Convert latitudes and longitudes to x, y coordinates
            latitude = df_plot['latitude'].apply(lambda x: float(x.split(':')[0]) + float(x.split(':')[1])/60 + float(x.split(':')[2])/3600)
            longitude = df_plot['longitude'].apply(lambda x: float(x.split(':')[0]) + float(x.split(':')[1])/60 + float(x.split(':')[2])/3600)
            x, y = m(list(longitude), list(latitude))

            # Define colormap and normalization
            cmap = plt.cm.RdYlGn_r if sType == 'pers_' else plt.cm.RdBu_r
            vmin, vmax = (-0.151, 0.151) if sType == 'pers_' else (-15, 15)
            norm = Normalize(vmin=vmin, vmax=vmax)

            # Plot scattered dots
            if sType == 'pers_':
                sc = m.scatter(x, y, c=df_plot[f'mean_diff_pers_{sSeason}_{pm}'], cmap=cmap, norm=norm, s=100, marker='o', alpha=.7)
            else: 
                sc = m.scatter(x, y, c=df_plot[f'mean_diff_{sSeason}_{pm}'], cmap=cmap, norm=norm, s=100, marker='o', alpha=.7)

            scatter_plots.append(sc)

            # Add title with subplot numbering
            sign = '+' if pm==str(1) else '-'
            
            ax.set_title(f'({chr(97 + 3 + 3*j + i)})' + ' $\\overline{\\Delta}_{\\psi_s}(\\tau)$ with $s=$' + pattern + sign + f' and $\\tau$ = {tau}', fontsize=12)

            # Add lat and long labels without gridlines
            if j == 1:
                meridians = np.arange(-20, 61, 20)
                m.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=8, linewidth=0.01)  # labels on the bottom
            if i == 0:
                parallels = np.arange(30, 81, 10)
                m.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=8, linewidth=0.01)  # labels on the left

    # Create a single colorbar horizontally at the bottom of the plots
    cbar_ax = fig.add_axes([0.2, 0.075, 0.6, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(scatter_plots[0], cax=cbar_ax, orientation='horizontal')
    plt.show()
    
def plot_combined_heatmaps(df1, df2, sType1, title1, sSeason, sType2, title2, sign=False):
    """
    Generate a combined heatmap with two horizontally aligned subplots, with the positions swapped.

    Args:
        df1: First DataFrame containing data for the second heatmap (swapped position).
        df2: Second DataFrame containing data for the first heatmap (swapped position).
        sType1: The `sType` value for the second heatmap.
        title1: Title for the second heatmap.
        sSeason: String representing the season for the first heatmap.
        sType2: The `sType` value for the first heatmap.
        title2: Title for the first heatmap.
    """
    def preprocess_data(df, drop_stations):
        """Helper function to preprocess DataFrame."""
        try:
            df.set_index('STANAME', inplace=True)
            df = df.drop(drop_stations, axis=0)
            df.reset_index(inplace=True)
        except KeyError:
            pass
        return df
    
    # Preprocess DataFrames
    drop_stations = ['IZANA', 'ELAT', 'ELAT-1', 'STA. CRUZ DE TENERIFE', 'TENERIFE/LOS RODEOS']
    df1 = preprocess_data(df1, drop_stations)  # Data for second heatmap
    df2 = preprocess_data(df2, drop_stations)  # Data for first heatmap
    if sign:
        df2 = df2.loc[df2.hit==True]
    # Initialize figure with two side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=200, facecolor='white')

    # Helper function to create a Basemap
    def create_basemap(ax):
        m = Basemap(projection='merc', llcrnrlat=30, urcrnrlat=75, llcrnrlon=-20, urcrnrlon=60, resolution='l', ax=ax)
        m.drawcoastlines()
        m.drawcountries()
        return m

    # First subplot (swapped: now showing Fig. 2)
    ax1 = axes[0]
    m1 = create_basemap(ax1)
    latitude1 = df2['latitude'].apply(lambda x: float(x.split(':')[0]) + float(x.split(':')[1]) / 60 + float(x.split(':')[2]) / 3600)
    longitude1 = df2['longitude'].apply(lambda x: float(x.split(':')[0]) + float(x.split(':')[1]) / 60 + float(x.split(':')[2]) / 3600)
    x1, y1 = m1(list(longitude1), list(latitude1))

    # Plot first heatmap
    cmap1 = plt.cm.RdYlGn_r
    if sType2 != '_unc':
        vmin1, vcenter1, vmax1 = -0.151, 0, 0.151
        data1 = df2['mean_diff_' + sSeason]
    else:
        vmin1, vcenter1, vmax1 = -0.151, 0, 0.151
        data1 = df2['mean_diff_' + sSeason + '_unc']
    norm1 = TwoSlopeNorm(vmin=vmin1, vcenter=vcenter1, vmax=vmax1)
    sc1 = m1.scatter(x1, y1, c=data1, cmap=cmap1, norm=norm1, s=30, marker='o', alpha=1)

    # Add parallels and meridians to first subplot
    parallels = np.arange(30, 81, 10)
    m1.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=8, linewidth=0.0001)
    meridians = np.arange(-20, 60, 20)
    m1.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=8, linewidth=0.0001)

    # Add colorbar and title for first subplot
    cbar1 = fig.colorbar(sc1, ax=ax1, orientation='vertical', shrink=0.85, pad=0.02)
    cbar1.set_label('$\Delta_{\\gamma_5}$', fontsize=10)
    ax1.set_title('(a) ' + title2, fontsize=14, pad=20)

    # Second subplot (swapped: now showing Fig. 1)
    ax2 = axes[1]
    m2 = create_basemap(ax2)
    latitude2 = df1['latitude'].apply(lambda x: float(x.split(':')[0]) + float(x.split(':')[1]) / 60 + float(x.split(':')[2]) / 3600)
    longitude2 = df1['longitude'].apply(lambda x: float(x.split(':')[0]) + float(x.split(':')[1]) / 60 + float(x.split(':')[2]) / 3600)
    x2, y2 = m2(list(longitude2), list(latitude2))

    # Plot second heatmap
    norm2 = TwoSlopeNorm(vmin=0, vcenter=50, vmax=100)
    cmap2 = plt.cm.YlOrRd
    data2 = df1[sType1]
    sc2 = m2.scatter(x2, y2, c=data2, cmap=cmap2, norm=norm2, s=30, marker='o', alpha=1)

    # Add parallels and meridians to second subplot
    m2.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=8, linewidth=0.0001)
    m2.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=8, linewidth=0.0001)

    # Add colorbar and title for second subplot
    cbar2 = fig.colorbar(sc2, ax=ax2, orientation='vertical', shrink=0.85, pad=0.02)
    cbar2.set_label('Fraction of days with more than 0.5mm precipitation (in %)', fontsize=10)
    ax2.set_title('(b) ' + title1, fontsize=14, pad=20)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()
    


def plot_single_heatmap(df1, sType, title, sSeason, sign=True):
    """
    Plot a single heatmap showing precipitation or persistence data on a Basemap.

    Args:
        df1: DataFrame with station metadata and values to plot.
        sType: Column name to use for coloring (e.g., 'precip_percent' or 'mean_diff_winter').
        title: Title of the plot.
        sSeason: Season string used to access column names if needed.
    """
    # Drop specific stations if present
    drop_stations = ['IZANA', 'ELAT', 'ELAT-1', 'STA. CRUZ DE TENERIFE', 'TENERIFE/LOS RODEOS']
    df1 = df1[~df1['STANAME'].isin(drop_stations)]
    if sign:
        df1 = df1.loc[df1.hit==True]
    # Convert lat/lon from D:M:S to decimal
    latitude = df1['latitude'].apply(lambda x: float(x.split(':')[0]) + float(x.split(':')[1]) / 60 + float(x.split(':')[2]) / 3600)
    longitude = df1['longitude'].apply(lambda x: float(x.split(':')[0]) + float(x.split(':')[1]) / 60 + float(x.split(':')[2]) / 3600)

    # Setup figure and Basemap
    fig, ax = plt.subplots(figsize=(7, 6), dpi=100, facecolor='white')
    m = Basemap(projection='merc', llcrnrlat=30, urcrnrlat=75, llcrnrlon=-20, urcrnrlon=60, resolution='l', ax=ax)
    m.drawcoastlines()
    m.drawcountries()

    # Convert to map coordinates
    x, y = m(list(longitude), list(latitude))

    # Color map and normalization
    if sType != '_unc':
        cmap = plt.cm.RdYlGn_r
        vmin, vcenter, vmax = -0.151, 0, 0.151
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        data = df1['mean_diff_' + sSeason] if 'mean_diff_' + sSeason in df1.columns else df1[sType]
        colorbar_label = '$\Delta_{Q_5}$'
    else:
        cmap = plt.cm.YlOrRd
        norm = TwoSlopeNorm(vmin=0, vcenter=50, vmax=100)
        data = df1[sType]
        colorbar_label = 'Fraction of days > 0.5mm precipitation (%)'

    # Scatter plot
    sc = m.scatter(x, y, c=data, cmap=cmap, norm=norm, s=30, marker='o', alpha=1)

    # Add gridlines
    parallels = np.arange(30, 81, 10)
    meridians = np.arange(-20, 60, 20)
    m.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=8, linewidth=0.0001)
    m.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=8, linewidth=0.0001)

    # Add colorbar and title
    cbar = fig.colorbar(sc, ax=ax, orientation='vertical', shrink=0.85, pad=0.02)
    cbar.set_label(colorbar_label, fontsize=10)
    ax.set_title(title, fontsize=14, pad=20)

    plt.tight_layout()
    plt.show()

    
def plot_coefficient_evolution(sCity='DE BILT'):
    """
    Computes and plots the evolution of coefficients and NAO index over 30-year windows.

    Parameters:
    - sCity (str): City to analyze (default: 'DE BILT')

    Returns:
    - None (Displays plots)
    """

    # Initialize lists to store values for minus (0) and plus (1) versions
    l_coefs = {fTau: {'minus': [], 'plus': []} for fTau in [0.05, 0.5, 0.95]}
    l_conf_low = {fTau: {'minus': [], 'plus': []} for fTau in [0.05, 0.5, 0.95]}
    l_conf_up = {fTau: {'minus': [], 'plus': []} for fTau in [0.05, 0.5, 0.95]}
    upper_quintile_values = {fTau: {'minus': [], 'plus': []} for fTau in [0.05, 0.5, 0.95]}

    x = np.arange(0, 41, 1)  # Range for year windows

    # Iterate over year windows
    for i in x:
        yearstart, yearend = 1950 + i, 1980 + i
        print(f'\rCurrently calculating coefficients for {yearstart}-{yearend}', end='')

        for fTau in [0.05, 0.5, 0.95]:
            # Perform analysis for minus version
            test = QAR_temperature(sCity=sCity, fTau=fTau, use_statsmodels=True, include_nao=True, split_nao=True, oldstart=str(yearstart)+'-', oldend=str(yearend)+'-')
            test.plot_paths_with_nao(2019, plot=False, alpha=0.05)
            l_coefs[fTau]['minus'].append(test.mCurves_old.iloc[0])
            l_conf_low[fTau]['minus'].append(test.mCurves_old_conf_low.iloc[0])
            l_conf_up[fTau]['minus'].append(test.mCurves_old_conf_up.iloc[0])

            # Perform analysis for plus version
            l_coefs[fTau]['plus'].append(test.mCurves_old.iloc[0])
            l_conf_low[fTau]['plus'].append(test.mCurves_old_conf_low.iloc[0])
            l_conf_up[fTau]['plus'].append(test.mCurves_old_conf_up.iloc[0])

            # NAO index calculation
            test = QAR_temperature(sCity=sCity, fTau=fTau, use_statsmodels=True, include_nao=True, oldstart=str(yearstart)+'-', oldend=str(yearend)+'-')
            test.prepare_data()
            data_old = test.old.nao_index_cdas
            data_winter_old = data_old.loc[data_old.index.month.isin([12, 1, 2])]

            # Compute upper quintile of NAO index
            upper_quintile_nao = np.quantile(data_winter_old, 0.8)
            upper_quintile_values[fTau]['minus'].append(upper_quintile_nao)
            upper_quintile_values[fTau]['plus'].append(upper_quintile_nao)

    # Plot results
    fig, axs = plt.subplots(2, 3, figsize=(18, 12), sharex=True, sharey=False, dpi=300, facecolor='white')

    # Subplot labels and titles
    subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    titles = [
        "$\\phi_t(0.05), s=$NAO$-$", "$\\phi_t(0.5), s=$NAO$-$", "$\\phi_t(0.95), s=$NAO$-$",
        "$\\phi_t(0.05), s=$NAO$+$", "$\\phi_t(0.5), s=$NAO$+$", "$\\phi_t(0.95), s=$NAO$+$"
    ]

    x = np.arange(1980, 2021)

    for row, fTau in enumerate([0.05, 0.5, 0.95]):
        for col, sign in enumerate(['minus', 'plus']):
            ax = axs[col, row]
            
            # Plot coefficient values
            line1, = ax.plot(x, l_coefs[fTau][sign], marker='o', linestyle='-', color='blue')
            ax.fill_between(x, [l_conf_low[fTau][sign][i][0] for i in range(len(l_conf_low[fTau][sign]))],
                            [l_conf_up[fTau][sign][i][0] for i in range(len(l_conf_up[fTau][sign]))],
                            color='blue', alpha=0.3, label='95% Confidence Interval')
            ax.set_xlabel('End year of 30-year window')
            ax.set_ylabel('Coefficient Value', color='black', rotation=90)
            ax.grid(True)
            ax.set_title(f'{subplot_labels[row + (col * 3)]} {titles[row + (col * 3)]}')
            ax.tick_params(axis='y', colors='blue')

            # Twin axis for NAO index values
            ax2 = ax.twinx()
            line2, = ax2.plot(x, upper_quintile_values[fTau][sign], marker='s', linestyle='--', color='red')
            ax2.set_ylabel('NAO Index Value', color='black', rotation=90)
            ax2.tick_params(axis='y', colors='red')

    plt.tight_layout()
    plt.show()
    
    
    
def plot_nao_quintiles_vs_rain_prob(sCity='DE BILT', fTau=0.95, mm_threshold=5, confidence=0.95, pattern='NAO'):
    """
    Computes and plots the probability of rain conditioned on NAO index quintiles
    for both past and recent periods.

    Parameters:
    - sCity (str): City name (default: 'DE BILT')
    - fTau (float): Quantile regression parameter (default: 0.95)
    - mm_threshold (float): Threshold for considering a day as rainy (default: 5mm)
    - confidence (float): Confidence level for error bars (default: 0.95)

    Returns:
    - None (Displays the plot)
    """

    # Initialize QAR object
    test = QAR_precipitation(sCity=sCity, fTau=fTau, use_statsmodels=True, include_nao=True, pattern=pattern)
    test.prepare_data()

    # Winter months
    months = [12, 1, 2]
    
    # Generate binary time series for precipitation
    y_prec_old = (test.old.Temp >= mm_threshold) * 1
    data_old = pd.DataFrame(y_prec_old, columns=['Temp'])
    data_old['nao_index_cdas'] = test.old.nao_index_cdas

    y_prec_new = (test.new.Temp >= mm_threshold) * 1
    data_new = pd.DataFrame(y_prec_new, columns=['Temp'])
    data_new['nao_index_cdas'] = test.new.nao_index_cdas

    # Filter winter months
    data_winter_new = data_new.loc[data_new.index.month.isin(months)]
    data_winter_old = data_old.loc[data_old.index.month.isin(months)]

    # Confidence interval calculation
    z_score = stats.norm.ppf((1 + confidence) / 2)

    # Define quintiles
    quantiles = np.linspace(0, 1, 6)  
    p_rain_new, p_rain_old, ci_low_new, ci_high_new, ci_low_old, ci_high_old = [], [], [], [], [], []

    # Compute probabilities and confidence intervals for each NAO quintile
    for i in range(len(quantiles) - 1):
        lower_quantile = quantiles[i]
        upper_quantile = quantiles[i + 1]

        # New dataset
        mask_new = (data_winter_new.nao_index_cdas.shift(1) < np.quantile(data_winter_new.nao_index_cdas, upper_quantile)) & \
                   (data_winter_new.nao_index_cdas.shift(1) > np.quantile(data_winter_new.nao_index_cdas, lower_quantile))
        subset_new = data_winter_new.loc[mask_new]
        mean_temp_new = subset_new.mean().Temp
        std_temp_new = subset_new.Temp.std()
        n_new = subset_new.shape[0]
        se_new = std_temp_new / np.sqrt(n_new) if n_new > 0 else 0
        ci_low_new.append(mean_temp_new - z_score * se_new)
        ci_high_new.append(mean_temp_new + z_score * se_new)
        p_rain_new.append(mean_temp_new)

        # Old dataset
        mask_old = (data_winter_old.nao_index_cdas.shift(1) < np.quantile(data_winter_old.nao_index_cdas, upper_quantile)) & \
                   (data_winter_old.nao_index_cdas.shift(1) > np.quantile(data_winter_old.nao_index_cdas, lower_quantile))
        subset_old = data_winter_old.loc[mask_old]
        mean_temp_old = subset_old.mean().Temp
        std_temp_old = subset_old.Temp.std()
        n_old = subset_old.shape[0]
        se_old = std_temp_old / np.sqrt(n_old) if n_old > 0 else 0
        ci_low_old.append(mean_temp_old - z_score * se_old)
        ci_high_old.append(mean_temp_old + z_score * se_old)
        p_rain_old.append(mean_temp_old)

    # Create a plot
    plt.figure(figsize=(10, 5), dpi=100, facecolor="white")

    # Define positions for the bars
    x = np.arange(len(p_rain_new))  # The x locations for the groups
    width = 0.2  # The width of the bars

    # Plot old data
    plt.errorbar(x - width/2, p_rain_old, yerr=[np.array(p_rain_old) - np.array(ci_low_old), np.array(ci_high_old) - np.array(p_rain_old)], fmt='o', color='orange', label='Old')

    # Plot new data
    plt.errorbar(x + width/2, p_rain_new, yerr=[np.array(p_rain_new) - np.array(ci_low_new), np.array(ci_high_new) - np.array(p_rain_new)], fmt='o', color='red', label='New')

    # Labeling
    plt.xticks(x, [f'Q{i+1}' for i in range(len(p_rain_new))])
    plt.xlabel('NAO Index Quintiles')
    plt.ylabel('Probability of Rain')
    plt.title('(b)')
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()


def analyze_temperature_vs_nao(sCity='DE BILT', quant=0.95, temp=False, fix_old=False):
    """
    Analyzes the relationship between NAO index and temperature for winter months.

    Parameters:
    - sCity (str): City name (default: 'DE BILT')
    - quant (float): Quantile for filtering (default: 0.95)
    - temp (bool): Whether to filter based on temperature (default: False)
    - fix_old (bool): Whether to fix the old dataset to a narrow range of NAO values (default: False)

    Returns:
    - None (Displays KDE and Scatter plots)
    """

    # Initialize QAR_temperature object
    test = QAR_temperature(sCity=sCity, fTau=0.95, use_statsmodels=True, include_nao=True, split_nao=True, iLeafs=2)
    test.prepare_data()

    # Filter data for winter months (December, January, February)
    old_winter = test.old.loc[test.old.index.month.isin([12, 1, 2])]
    new_winter = test.new.loc[test.new.index.month.isin([12, 1, 2])]

    # Apply filtering based on NAO index or temperature
    if temp:
        if quant > 0.5:
            old_filtered = old_winter.loc[(old_winter.Temp.shift(1) >= np.quantile(old_winter.Temp, quant))]
            new_filtered = new_winter.loc[(new_winter.Temp.shift(1) >= np.quantile(new_winter.Temp, quant))]
        else:
            old_filtered = old_winter.loc[(old_winter.Temp.shift(1) <= np.quantile(old_winter.Temp, quant))]
            new_filtered = new_winter.loc[(new_winter.Temp.shift(1) <= np.quantile(new_winter.Temp, quant))]
    else:
        if quant > 0.5:
            old_filtered = old_winter.loc[(old_winter.nao_index_cdas >= np.quantile(old_winter.nao_index_cdas, quant))]
            new_filtered = new_winter.loc[(new_winter.nao_index_cdas >= np.quantile(new_winter.nao_index_cdas, quant))]
        else:
            old_filtered = old_winter.loc[(old_winter.nao_index_cdas <= np.quantile(old_winter.nao_index_cdas, quant))]
            new_filtered = new_winter.loc[(new_winter.nao_index_cdas <= np.quantile(new_winter.nao_index_cdas, quant))]

    # Fix old dataset to a narrow range of NAO values
    if fix_old:
        if temp:
            if quant > 0.5:
                old_filtered = old_winter.loc[(old_winter.Temp.shift(1) >= np.quantile(old_winter.Temp, quant)) & (old_winter.Temp.shift(1) <= np.quantile(old_winter.Temp, .99))]
                new_filtered = new_winter.loc[(new_winter.Temp.shift(1) >= np.quantile(old_winter.Temp, quant)) & (new_winter.Temp.shift(1) <= np.quantile(old_winter.Temp, .99))]
            else:
                old_filtered = old_winter.loc[(old_winter.Temp.shift(1) <= np.quantile(old_winter.Temp, .075)) & (old_winter.Temp.shift(1) >= np.quantile(old_winter.Temp, 0.025))]
                new_filtered = new_winter.loc[(new_winter.Temp.shift(1) <= np.quantile(old_winter.Temp, .075)) & (new_winter.Temp.shift(1) >= np.quantile(old_winter.Temp, 0.025))]
        else:
            if quant > 0.5:
                old_filtered = old_winter.loc[(old_winter.nao_index_cdas >= np.quantile(old_winter.nao_index_cdas, .925)) & (old_winter.nao_index_cdas <= np.quantile(old_winter.nao_index_cdas, .975))]
                new_filtered = new_winter.loc[(new_winter.nao_index_cdas >= np.quantile(old_winter.nao_index_cdas, .925)) & (new_winter.nao_index_cdas <= np.quantile(old_winter.nao_index_cdas, .975))]
            else:
                old_filtered = old_winter.loc[(old_winter.nao_index_cdas <= np.quantile(old_winter.nao_index_cdas, quant))]
                new_filtered = new_winter.loc[(new_winter.nao_index_cdas <= np.quantile(old_winter.nao_index_cdas, quant))]

    # Extract temperature values for KDE plot
    old_data_q5 = old_filtered['Temp']
    new_data_q5 = new_filtered['Temp']

    # Calculate interquantile ranges
    quantiles = [.05, .1, .25, .75, .9, .95]
    def calculate_interquantile_ranges(data, quantiles):
        return {q: (np.quantile(data, q), np.quantile(data, 1 - q)) for q in quantiles}

    old_interquantile_ranges = calculate_interquantile_ranges(old_data_q5, quantiles)
    new_interquantile_ranges = calculate_interquantile_ranges(new_data_q5, quantiles)

    # KDE Plot
    plt.figure(figsize=(16, 8), dpi=100, facecolor="white")
    sns.kdeplot(old_data_q5, color='orange', label='Old Data', fill=True, alpha=0.5)
    sns.kdeplot(new_data_q5, color='red', label='New Data', fill=True, alpha=0.5)


    plt.xlabel('Temperature')
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    plt.show()

    # Scatter Plot for Temperatures vs Lagged NAO Index
    old = test.old.loc[test.old.index.month.isin([12, 1, 2])]
    new = test.new.loc[test.new.index.month.isin([12, 1, 2])]

    shifted_temp_old = old.nao_index_cdas.shift(1)
    q_low, q_high = shifted_temp_old.quantile(quant-0.025), shifted_temp_old.quantile(quant+0.025)


    old_is_fat = (shifted_temp_old >= q_low) & (shifted_temp_old <= q_high)
    old_sizes = np.where(old_is_fat, 3, 1)
    old_alphas = np.where(old_is_fat, 1, 0.1)

    shifted_temp_new = new.nao_index_cdas.shift(1)
    new_is_fat = (shifted_temp_new >= q_low) & (shifted_temp_new <= q_high)
    new_sizes = np.where(new_is_fat, 3, 1)
    new_alphas = np.where(new_is_fat, 1.0, 0.1)

    plt.figure(dpi=100, facecolor="white")
    plt.scatter(shifted_temp_old, old.Temp, c='orange', s=3, alpha=old_alphas)
    plt.scatter(shifted_temp_new, new.Temp, c='red', s=3, alpha=new_alphas)
    if quant == 0.05:
        plt.axvline(x=q_low, color='orange', linestyle='dotted', label='2.5th percentile (old)')
        plt.axvline(x=q_high, color='orange', linestyle='dotted', label='7.5th percentile (old)')
    elif quant == 0.95:
        plt.axvline(x=q_low, color='orange', linestyle='dotted', label='92.5th percentile (old)')
        plt.axvline(x=q_high, color='orange', linestyle='dotted', label='97.5th percentile (old)')
    plt.xlabel('Lagged Temperature')
    plt.ylabel('Temperature')


def analyze_temperature_vs_nao_combined(test, quant=0.95, temp=False):
    def get_filtered(data, quant, temp, fix_old, is_new, ref_data=None):
        shifted = data['Temp'].shift(1) if temp else data['nao_index_cdas']
        if fix_old:
            base = ref_data['nao_index_cdas'] if (not temp and is_new and ref_data is not None) else data['Temp'] if temp else data['nao_index_cdas']
            if quant > 0.5:
                return data[(shifted >= np.quantile(base, 0.925)) & (shifted <= np.quantile(base, 0.975))] if not temp else \
                       data[(shifted >= np.quantile(base, quant)) & (shifted <= np.quantile(base, 0.99))]
            else:
                return data[(shifted <= np.quantile(base, quant))] if not temp else \
                       data[(shifted <= np.quantile(base, 0.075)) & (shifted >= np.quantile(base, 0.025))]
        else:
            return data[(shifted >= np.quantile(shifted, quant))] if quant > 0.5 else \
                   data[(shifted <= np.quantile(shifted, quant))]

    test.prepare_data()
    old_winter = test.old[test.old.index.month.isin([12, 1, 2])].copy()
    new_winter = test.new[test.new.index.month.isin([12, 1, 2])].copy()

    fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=100, facecolor='white')

    for i, fix_old in enumerate([True, False]):
        # Scatter FIRST (top row left, bottom row left)
        row = 0 if i == 0 else 1
        col_scatter = 0
        col_kde = 1

        old = old_winter.copy()
        new = new_winter.copy()
        old['nao_lag1'] = old['nao_index_cdas'].shift(1)
        new['nao_lag1'] = new['nao_index_cdas'].shift(1)

        if fix_old:
            # Shared percentile based on old NAO
            q_low = old['nao_lag1'].quantile(quant - 0.025)
            q_high = old['nao_lag1'].quantile(quant + 0.025)

            old['alpha'] = np.where((old['nao_lag1'] >= q_low) & (old['nao_lag1'] <= q_high), 1.0, 0.1)
            new['alpha'] = np.where((new['nao_lag1'] >= q_low) & (new['nao_lag1'] <= q_high), 1.0, 0.1)

        else:
            # Independent thresholds for old and new
            q_old = old['nao_lag1'].quantile(quant)
            q_new = new['nao_lag1'].quantile(quant)

            old['alpha'] = np.where(old['nao_lag1'] >= q_old, 1.0, 0.1)
            new['alpha'] = np.where(new['nao_lag1'] >= q_new, 1.0, 0.1)

        ax_scatter = axs[row, col_scatter]
        ax_scatter.scatter(old['nao_lag1'], old['Temp'], c='orange', s=3, alpha=old['alpha'], label='Old')
        ax_scatter.scatter(new['nao_lag1'], new['Temp'], c='red', s=3, alpha=new['alpha'], label='New')
        
        if fix_old:
            ax_scatter.axvline(x=q_low, color='orange', linestyle='dotted', label='2.5th–97.5th (old)')
            ax_scatter.axvline(x=q_high, color='orange', linestyle='dotted')
        else:
            ax_scatter.axvline(x=q_old, color='orange', linestyle='dotted', label='95th (old)')
            ax_scatter.axvline(x=q_new, color='red', linestyle='dotted', label='95th (new)')
        
        ax_scatter.set_title(f"({chr(97 + i*2)}) Scatter – fix_old={fix_old}")
        ax_scatter.set_xlabel("Lagged NAO Index")
        ax_scatter.set_ylabel("Temperature")
        ax_scatter.legend()

        ax_scatter.set_title(f"({chr(97 + i*2)})")
        ax_scatter.set_xlabel("Lagged NAO Index")
        ax_scatter.set_ylabel("Temperature")
        ax_scatter.legend()

        # KDE — corresponding subplot (right of each row)
        old_filtered = get_filtered(old_winter, quant, temp, fix_old, is_new=False)
        new_filtered = get_filtered(new_winter, quant, temp, fix_old, is_new=True, ref_data=old_winter)

        ax_kde = axs[row, col_kde]
        sns.kdeplot(data=old_filtered, x='Temp', color='orange', fill=False, ax=ax_kde, label='Old')
        sns.kdeplot(data=new_filtered, x='Temp', color='red', fill=False, ax=ax_kde, label='New')
        ax_kde.set_title(f"({chr(98 + i*2)})")
        ax_kde.set_xlabel("Temperature")
        ax_kde.set_ylabel("Density")
        ax_kde.legend()

    plt.tight_layout()
    plt.show()
    




