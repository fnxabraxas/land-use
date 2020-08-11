# -*- coding: utf-8 -*-

import numpy as np
from auxiliar import tif2array, array2tif, csv2dict

files=csv2dict('url_list.csv', 'Magnitude', 'url') # dictionary pointing where data files are
with open('crops_list.txt') as f: commodities_crops = [i.strip() for i in f.readlines()] # list with crop names

countries_code_map_file=files['countries_code'] # it will be used to make nan of pixels that are not within a country
countries_code_map=tif2array(countries_code_map_file)

# crops files
presufix_pot_prod_crops=files['potential_productivity_crops'].split('*')
pot_prod_crops_file=[presufix_pot_prod_crops[0] + s + presufix_pot_prod_crops[1] for s in commodities_crops]
presufix_area_crops=files['area_crops'].split('*')
area_crops_file=[presufix_area_crops[0] + s + presufix_area_crops[1] for s in commodities_crops]
# grassland files
pot_prod_grass_file=files['potential_productivity_grass']

# suitability map for crops
numerator=[]
area_tot=[]
for i,pot_prod_file in enumerate(pot_prod_crops_file): # weighted average using area
    pot_prod=tif2array(pot_prod_file, model_map=countries_code_map_file)
    max_pot_prod=np.nanmax(pot_prod)
    area=tif2array(area_crops_file[i], model_map=countries_code_map_file)
    area_tot = area if i==0 else area_tot + area
    numerator_element = area*pot_prod/max_pot_prod
    numerator = numerator_element if i==0 else numerator + numerator_element
suit_crops=numerator
condition=(~np.isnan(suit_crops))&(area_tot>0) # to avoid dividing by zero
suit_crops[condition]/=area_tot[condition]

# suitability map for grassland
pot_prod_grass=tif2array(pot_prod_grass_file, model_map=countries_code_map_file)
max_pot_prod_grass=np.nanmax(pot_prod_grass)
suit_grass=pot_prod_grass/max_pot_prod_grass

# Removing unimportant areas (most of them outside of the margins) within countries (according to countries_map) that do not have information about suitability and fraction.
suit_crops[np.isnan(suit_crops)&(~np.isnan(countries_code_map))]=0.
suit_grass[np.isnan(suit_grass)&(~np.isnan(countries_code_map))]=0.

array2tif(files['suitability_crops'], suit_crops, countries_code_map_file)
array2tif(files['suitability_grass'], suit_grass, countries_code_map_file)



