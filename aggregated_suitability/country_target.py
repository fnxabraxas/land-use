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
opportunity_crops_file=files['opportunity_cost_crops']
opportunity_grass_file=files['opportunity_cost_grass']

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
opportunity_crops=tif2array(opportunity_crops_file, model_map=countries_code_map_file)
opportunity_grass=tif2array(opportunity_grass_file, model_map=countries_code_map_file)

# total suitability maps summing separetely crops and grass and then aggregating
SUIT_1P_map = frac_crops_present*suit_crops + frac_grass_present*suit_grass
SUIT_1F_map = frac_crops_future*suit_crops + frac_grass_future*suit_grass
SUIT_present_map = frac_crops_present*suit_crops*yield_ratio_crops + frac_grass_present*suit_grass*yield_ratio_grass
SUIT_future_map = frac_crops_future*suit_crops*yield_ratio_crops + frac_grass_future*suit_grass*yield_ratio_grass

# contribution of crops and pasture for sigma maps
contribution_sigma_crops = frac_crops_present.copy()
contribution_sigma_grass = frac_grass_present.copy()
index_opportunity = (frac_crops_present==0) & (frac_grass_present==0) & (~np.isnan(countries_code_map)) # replacing pixels within countries that have both frac equal to 0 with opportunity costs
contribution_sigma_crops[index_opportunity] = opportunity_crops[index_opportunity] 
contribution_sigma_grass[index_opportunity] = opportunity_grass[index_opportunity] 
total_contribution_sigma = contribution_sigma_crops + contribution_sigma_grass
index_opportunity0 = (~np.isnan(countries_code_map)) & (total_contribution_sigma>0)  # avoiding dividing by zero
contribution_sigma_crops[index_opportunity0] /= total_contribution_sigma[index_opportunity0]
contribution_sigma_grass[index_opportunity0] /= total_contribution_sigma[index_opportunity0]
array2tif(files['contribution_sigma_crops'], contribution_sigma_crops, countries_code_map_file)
array2tif(files['contribution_sigma_grass'], contribution_sigma_grass, countries_code_map_file)
# sigma maps
sigma = (contribution_sigma_crops * suit_crops * yield_ratio_crops + contribution_sigma_grass * suit_grass * yield_ratio_grass)
sigma_1P = (contribution_sigma_crops * suit_crops + contribution_sigma_grass * suit_grass)
delta_sigma_1P = sigma_1P - sigma
array2tif(files['sigma'], sigma, countries_code_map_file)
array2tif(files['delta_sigma_1P'], delta_sigma_1P, countries_code_map_file)

# results for each country
df_out = df_countries_names_code.copy()
df_out = df_out.reindex(columns = df_out.columns.tolist() + ['DeltaS','DeltaS_1P', 'DeltaS_1F', 'fC_eq' ])
for k, country_code in enumerate(df_out['CODE']):
    index_bool = countries_code_map==country_code
    if np.sum(index_bool)>0: # ignoring countries without associated pixels 
        print('Calculating targets for country ' + str(k))
        S_1P = np.sum(SUIT_1P_map[index_bool])
        S_1F = np.sum(SUIT_1F_map[index_bool])
        S_pres = np.sum(SUIT_present_map[index_bool])
        S_fut = np.sum(SUIT_future_map[index_bool])       
        df_out.at[k,'DeltaS'] = S_pres - S_fut
        df_out.at[k,'DeltaS_1P'] = S_1P - S_pres
        df_out.at[k,'DeltaS_1F'] = S_1F - S_fut       
df_out['fC_eq']= -df_out['DeltaS'] / df_out['DeltaS_1P']
df_out.to_csv(files['country_targets'], index=False, na_rep='nan')

