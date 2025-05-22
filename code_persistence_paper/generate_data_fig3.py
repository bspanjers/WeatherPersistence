
import pandas as pd
import numpy as np
import os
from plots_europe import read_climate_data, map_station_with_city
from QAR import QAR_temperature


start_date = 1950
start_year_old = start_date
end_year_old = start_date + 30
start_year_new = 1990
end_year_new = start_year_new + 30
tau = .5
drop_na_larger_than = 0.05
iLeafs = 2
if iLeafs > 1:
    split_nao = True
    include_nao = True
else: 
    split_nao = False
    include_nao = False
    
folder_path = '../data_persistence/ECA_blend_tg/'
lendata = len(np.sort(os.listdir(folder_path))[:-4])
lat_long = pd.DataFrame(np.zeros((lendata, 5)))
lat_long[:] = np.nan
for (i, file_name) in enumerate(np.sort(os.listdir(folder_path))[1:-4]):
    station_name, starting_date = read_climate_data(folder_path + file_name)
    city_name, latitude, longitude = map_station_with_city(station_name, folder_path + 'stations.txt')
    if type(starting_date) != type(None):
        if int(starting_date)<=start_date:
                lat_long.iloc[i,:] = [file_name, station_name, latitude, longitude, city_name]

df = lat_long.dropna()  
df.columns =  ['file_name', 'STAID', 'latitude', 'longitude', 'city_name']
df_results = pd.DataFrame(np.zeros((len(df), 8)), columns=['STANAME', 'STAID', 'latitude', 'longitude', 
                                                            'mean_diff_winter', 'mean_diff_spring', 'mean_diff_summer', 'mean_diff_autumn'])
df_results[:] = np.nan
datetime_index_2019 = pd.date_range(start='2019-01-01', end='2019-12-31', freq='D')

for (i, file_name) in enumerate(df.file_name):
    print(f'\rCurrently calculating station {i+1} out of {len(df.file_name)}', end='')
    test = QAR_temperature(sFile=file_name, dropna=drop_na_larger_than, fTau=tau, 
                       oldend = str(end_year_old) + '-', oldstart=str(start_year_old) + '-', 
                       newend = str(end_year_new) + '-', newstart= str(start_year_new) + '-',
                       include_nao=include_nao, split_nao=split_nao, iLeafs=iLeafs)
    if test.iLeafs >= 2:
        season_list_pers = ['mean_diff_pers_winter_', 'mean_diff_pers_spring_', 'mean_diff_pers_summer_', 'mean_diff_pers_autumn_', 'hit']
    else: 
        season_list_pers = ['mean_diff_pers_winter', 'mean_diff_pers_spring', 'mean_diff_pers_summer', 'mean_diff_pers_autumn', 'hit']
    season_list_mean = ['mean_diff_winter_', 'mean_diff_spring_', 'mean_diff_summer_', 'mean_diff_autumn_']
    try: 
        if test.iLeafs >= 2:    
            test.results()   
        else: 
            test.results()
        for leaf in range(test.iLeafs):
            #differences in persistence for NAO+
            diff_pers = test.mCurves_new - test.mCurves_old
            diff_pers.index = datetime_index_2019
            mean_diff_pers_winter = diff_pers.loc[diff_pers.index.month.isin([12, 1, 2])].mean()
            mean_diff_pers_spring = diff_pers.loc[diff_pers.index.month.isin([3, 4, 5])].mean()
            mean_diff_pers_summer = diff_pers.loc[diff_pers.index.month.isin([6, 7, 8])].mean()
            mean_diff_pers_autumn = diff_pers.loc[diff_pers.index.month.isin([9, 10, 11])].mean()
            if test.iLeafs>1:
                uniform_lower_bound, uniform_upper_bound = test.lower_combined.iloc[:,leaf], test.upper_combined.iloc[:,leaf]
            else: 
                uniform_lower_bound, uniform_upper_bound = test.lower_combined, test.upper_combined

            uniform_lower_bound_winter = uniform_lower_bound.loc[uniform_lower_bound.index.month.isin([12,1,2])]
            uniform_upper_bound_winter = uniform_upper_bound.loc[uniform_upper_bound.index.month.isin([12,1,2])]
            outside_zero = (uniform_lower_bound_winter > 0) | (uniform_upper_bound_winter < 0)

            # Check if any such violations exist
            hit = outside_zero.any()
            if test.iLeafs >= 2:
                df_results.loc[i, [season_list_pers[i] + str(leaf) for i in range(len(season_list_pers))]] = mean_diff_pers_winter.values[leaf], mean_diff_pers_spring.values[leaf], mean_diff_pers_summer.values[leaf], mean_diff_pers_autumn.values[leaf], hit
            else: 
                df_results.loc[i, [season_list_pers[i] for i in range(len(season_list_pers))]] = mean_diff_pers_winter, mean_diff_pers_spring, mean_diff_pers_summer, mean_diff_pers_autumn, hit

        #mean differences in temperature per season
        mean_diff_winter = test.new.loc[test.new.index.month.isin([12, 1, 2])].mean() - test.old.loc[test.old.index.month.isin([12,1,2])].mean()
        mean_diff_spring = test.new.loc[test.new.index.month.isin([3, 4, 5])].mean() - test.old.loc[test.old.index.month.isin([3, 4, 5])].mean()
        mean_diff_summer = test.new.loc[test.new.index.month.isin([6, 7, 8])].mean() - test.old.loc[test.old.index.month.isin([6, 7, 8])].mean()
        mean_diff_autumn = test.new.loc[test.new.index.month.isin([9, 10, 11])].mean() - test.old.loc[test.old.index.month.isin([9, 10, 11])].mean()
        df_results.iloc[i, :8] = df.city_name.iloc[i], df.STAID.iloc[i], df.latitude.iloc[i], df.longitude.iloc[i], mean_diff_winter.values[0], mean_diff_spring.values[0], mean_diff_summer.values[0], mean_diff_autumn.values[0]
    except ValueError: 
        pass 

df_results05 = df_results
#df_results = df_results.dropna().set_index('STANAME')
#df_results = df_results.drop(['IZANA','ELAT', 'ELAT-1', 'STA. CRUZ DE TENERIFE', 'TENERIFE/LOS RODEOS'],axis=0)
#df_results.to_csv('/Users/admin/Documents/PhD/persistence/data_persistence/results_' + str(tau)[-2:] + '_' + str(start_date) + 'AMOsign_hits.csv')



