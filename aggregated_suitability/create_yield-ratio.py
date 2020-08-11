# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from auxiliar import tif2array, array2tif, csv2dict, fill_average


files=csv2dict('url_list.csv', 'Magnitude', 'url') # dictionary pointing where data files are


countries_code_map_file=files['countries_code'] # it will be used to make nan of pixels that are not within a country
countries_names_code_file=files['countries_names_code'] # csv with codes of countries

countries_code_map=tif2array(countries_code_map_file)
df_countries_names_code=pd.read_csv(countries_names_code_file, usecols=['CODE', 'NAME'])
list_countries_code=df_countries_names_code['CODE'].tolist()

# crops
frac_crops_file=files['LU_crops_present']
frac_crops=tif2array(frac_crops_file, model_map=countries_code_map_file, nodata_value=0) # set unkonwn pixels within countries to 0, and oceans to nan
yield_ratio_crops_file=files['yield_ratio_crops_input']
yield_ratio_crops=tif2array(yield_ratio_crops_file, model_map=countries_code_map_file)
yield_ratio_crops[yield_ratio_crops==0] = np.nan  # pixels without defined yield_ratio (these missing values wil be filled with the average of the region)
yield_ratio_crops[yield_ratio_crops==1] = (0 + 0.1) / 2 
yield_ratio_crops[yield_ratio_crops==2] = (0.1 + 0.25) / 2
yield_ratio_crops[yield_ratio_crops==3] = (0.25 + 0.40) / 2
yield_ratio_crops[yield_ratio_crops==4] = (0.4 + 0.55) / 2
yield_ratio_crops[yield_ratio_crops==5] = (0.55 + 0.70) / 2
yield_ratio_crops[yield_ratio_crops==6] = (0.70 + 0.85) / 2
yield_ratio_crops[yield_ratio_crops==7] = (0.85 + 1) / 2
# Filling missing values with the average of the region (weights are taken from frac_crops)
yield_ratio_crops_filled, average_yield_ratio_crops_country, sum_frac_crops = fill_average(yield_ratio_crops, countries_code_map, frac_crops, list_model=list_countries_code, verbose=True)
array2tif(files['yield_ratio_crops'], yield_ratio_crops_filled, countries_code_map_file)

# grassland
frac_grass_file=files['LU_grass_present']
frac_grass=tif2array(frac_grass_file, model_map=countries_code_map_file, nodata_value=0) # set unkonwn pixels within countries to 0, and oceans to nan
gap_grass_file=files['gap_grass']
gap_grass=tif2array(gap_grass_file, model_map=countries_code_map_file)
yield_ratio_grass = 1. - gap_grass
yield_ratio_grass[(gap_grass==0)] = np.nan  # gap==0 means no info (these missing values will be filled with the average of the region)
# Filling missing values with the average of the region (weights are taken from frac_grass)
yield_ratio_grass_filled, average_yield_ratio_grass_country, sum_frac_grass = fill_average(yield_ratio_grass, countries_code_map, frac_grass, list_model=list_countries_code, verbose=True)
array2tif(files['yield_ratio_grass'], yield_ratio_grass_filled, countries_code_map_file)

# create dataframe with yield ratio averages (averages were calculated before replacing missing values)
df_ave = df_countries_names_code.merge(pd.DataFrame(average_yield_ratio_crops_country,columns=['CODE', 'average_yield_rate_crops']), on='CODE', how='left')
df_ave = df_ave.merge(pd.DataFrame(average_yield_ratio_grass_country,columns=['CODE', 'average_yield_rate_grass']), on='CODE', how='left')
df_ave.to_csv(files['average_yield_ratio'], index=False)


