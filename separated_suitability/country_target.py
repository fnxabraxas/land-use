# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from auxiliar import tif2array, array2tif, csv2dict, fill_average


files=csv2dict('url_list.csv', 'Magnitude', 'url') # dictionary pointing where data files are

countries_code_map_file=files['countries_code'] 
countries_names_code_file=files['countries_names_code']
frac_crops_present_file=files['LU_crops_present']
frac_grass_present_file=files['LU_grass_present']
frac_crops_future_file=files['LU_crops_future']
frac_grass_future_file=files['LU_grass_future']
yield_ratio_crops_file=files['yield_ratio_crops']
yield_ratio_grass_file=files['yield_ratio_grass']
suit_crops_file=files['suitability_crops']
suit_grass_file=files['suitability_grass']


# reading
df_countries_names_code=pd.read_csv(countries_names_code_file, usecols=['CODE', 'NAME'])
countries_code_map=tif2array(countries_code_map_file)
frac_crops_present=tif2array(frac_crops_present_file, model_map=countries_code_map_file, nodata_value=0)
frac_grass_present=tif2array(frac_grass_present_file, model_map=countries_code_map_file, nodata_value=0)
frac_crops_future=tif2array(frac_crops_future_file, model_map=countries_code_map_file, nodata_value=0)
frac_grass_future=tif2array(frac_grass_future_file, model_map=countries_code_map_file, nodata_value=0)
yield_ratio_grass=tif2array(yield_ratio_grass_file, model_map=countries_code_map_file, nodata_value=0)
yield_ratio_crops=tif2array(yield_ratio_crops_file, model_map=countries_code_map_file, nodata_value=0)
suit_crops=tif2array(suit_crops_file, model_map=countries_code_map_file)
suit_grass=tif2array(suit_grass_file, model_map=countries_code_map_file)
suit_crops[np.isnan(suit_crops)] = 0. # pixels that are not defined in suitability (just in case)
suit_grass[np.isnan(suit_grass)] = 0. # pixels that are not defined in suitability (just in casa)

# total suitability maps
SUIT_1P_crops_map = frac_crops_present*suit_crops 
SUIT_1F_crops_map = frac_crops_future*suit_crops 
SUIT_present_crops_map = frac_crops_present*suit_crops*yield_ratio_crops 
SUIT_future_crops_map = frac_crops_future*suit_crops*yield_ratio_crops 
SUIT_1P_grass_map = frac_grass_present*suit_grass 
SUIT_1F_grass_map = frac_grass_future*suit_grass 
SUIT_present_grass_map = frac_grass_present*suit_grass*yield_ratio_grass 
SUIT_future_grass_map = frac_grass_future*suit_grass*yield_ratio_grass  

# sigma maps
sigma_crops = suit_crops * yield_ratio_crops
sigma_grass = suit_grass * yield_ratio_grass
sigma_1P_crops = suit_crops
sigma_1P_grass = suit_grass
delta_sigma_1P_crops = sigma_1P_crops - sigma_crops
delta_sigma_1P_grass = sigma_1P_grass - sigma_grass
array2tif(files['sigma_crops'], sigma_crops, countries_code_map_file)
array2tif(files['sigma_grass'], sigma_grass, countries_code_map_file)
array2tif(files['delta_sigma_1P_crops'], delta_sigma_1P_crops, countries_code_map_file)
array2tif(files['delta_sigma_1P_grass'], delta_sigma_1P_grass, countries_code_map_file)

# results for each country
df_out = df_countries_names_code.copy()
df_out = df_out.reindex(columns = df_out.columns.tolist() + ['DeltaS_crops','DeltaS_1P_crops', 'DeltaS_1F_crops', 'fC_eq_crops', 'DeltaS_grass','DeltaS_1P_grass', 'DeltaS_1F_grass', 'fC_eq_grass', 'DeltaS_aggregated','DeltaS_1P_aggregated', 'DeltaS_1F_aggregated', 'fC_eq_aggregated' ])
for k, country_code in enumerate(df_out['CODE']):
    index_bool = countries_code_map==country_code
    if np.sum(index_bool)>0: # ignoring countries without associated pixels 
        print('Calculating targets for country ' + str(k))
        S_1P_crops = np.sum(SUIT_1P_crops_map[index_bool])
        S_1F_crops = np.sum(SUIT_1F_crops_map[index_bool])
        S_pres_crops = np.sum(SUIT_present_crops_map[index_bool])
        S_fut_crops = np.sum(SUIT_future_crops_map[index_bool])       
        df_out.at[k,'DeltaS_crops'] = S_pres_crops - S_fut_crops
        df_out.at[k,'DeltaS_1P_crops'] = S_1P_crops - S_pres_crops
        df_out.at[k,'DeltaS_1F_crops'] = S_1F_crops - S_fut_crops   
        S_1P_grass = np.sum(SUIT_1P_grass_map[index_bool])
        S_1F_grass = np.sum(SUIT_1F_grass_map[index_bool])
        S_pres_grass = np.sum(SUIT_present_grass_map[index_bool])
        S_fut_grass = np.sum(SUIT_future_grass_map[index_bool])       
        df_out.at[k,'DeltaS_grass'] = S_pres_grass - S_fut_grass
        df_out.at[k,'DeltaS_1P_grass'] = S_1P_grass - S_pres_grass
        df_out.at[k,'DeltaS_1F_grass'] = S_1F_grass - S_fut_grass 
df_out['fC_eq_crops']= -df_out['DeltaS_crops'] / df_out['DeltaS_1P_crops']
df_out['fC_eq_grass']= -df_out['DeltaS_grass'] / df_out['DeltaS_1P_grass']
df_out['DeltaS_aggregated'] = df_out['DeltaS_crops'] + df_out['DeltaS_grass']
df_out['DeltaS_1P_aggregated'] = df_out['DeltaS_1P_crops'] + df_out['DeltaS_1P_grass']
df_out['DeltaS_1F_aggregated'] = df_out['DeltaS_1F_crops'] + df_out['DeltaS_1F_grass'] 
df_out['fC_eq_aggregated']= -df_out['DeltaS_aggregated'] / df_out['DeltaS_1P_aggregated']
df_out.to_csv(files['country_targets'], index=False, na_rep='nan')

