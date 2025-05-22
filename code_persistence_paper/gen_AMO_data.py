#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 18:01:03 2025

@author: admin
"""

import pandas as pd

import requests

url = "https://www.psl.noaa.gov/data/correlation/amon.us.data"
response = requests.get(url)

# Save to file
with open("amon.us.data", "w") as f:
    f.write(response.text)

print("AMO data downloaded successfully.")

# Load the file using whitespace delimiter
df = pd.read_csv(
    "amon.us.data", 
    skiprows=1,
    delim_whitespace=True, 
    header=None, 
    names=["Year", "Jan", "Feb", "Mar", "Apr", "May", "Jun", 
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
)

# Melt into long format
df_long = df.melt(id_vars="Year", var_name="Month", value_name="AMO")

# Drop missing values (some rows might contain -99.900 placeholders)
df_long["AMO"] = pd.to_numeric(df_long["AMO"], errors="coerce")
df_long = df_long.dropna()

# Convert to datetime
df_long["Date"] = pd.to_datetime(df_long["Year"].astype(str) + df_long["Month"], format="%Y%b")

# Sort chronologically
df_long = df_long.sort_values("Date").reset_index(drop=True)


df_long.set_index('Date', inplace=True, drop=True)
df_long.drop(['Year', 'Month'],axis=1, inplace=True)
df_long = df_long.loc[(df_long.index.year<=2019) & (df_long.index.year >= 1950)]

# Step 1: Ensure AMO index is datetime
monthly_amo = df_long.AMO.copy()
monthly_amo.index = pd.to_datetime(monthly_amo.index)

# Step 2: Create daily date range
daily_index = pd.date_range(start=monthly_amo.index.min(), end=monthly_amo.index.max() + pd.offsets.MonthEnd(1), freq='D')

# Step 3: Convert daily index to month starts (to join with monthly AMO)
month_start_index = daily_index.to_period('M').to_timestamp()

# Step 4: Map each day to its month's AMO value
daily_amo = monthly_amo.reindex(month_start_index.values).values

# Step 5: Build final DataFrame
df_amo_daily = pd.DataFrame({
    'year': daily_index.year,
    'month': daily_index.month,
    'day': daily_index.day,
    'amo_index': daily_amo
})
