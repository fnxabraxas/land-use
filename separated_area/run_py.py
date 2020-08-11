
import sys
import os
from datetime import datetime
import numpy as np
import pandas as pd
import rasterio
# import rasterio._shim       # to avoid problems with pyinstaller
# import rasterio.control     # to avoid problems with pyinstaller
# import rasterio.crs         # to avoid problems with pyinstaller
# import rasterio.sample      # to avoid problems with pyinstaller
# import rasterio.vrt         # to avoid problems with pyinstaller
# import rasterio._features   # to avoid problems with pyinstaller


#============ Auxiliary functions ==============================================

def tif2array(input_file, masked=True, nodata_value=np.nan, band=1, model_map=None, band_model_map=1, verbose=True):
    if verbose==True: print('Reading ' + input_file)
    with rasterio.open(input_file,'r') as ds:
        array=ds.read(band, masked=masked) # take the first layer 'band' (1 is the first)
    if masked==True: array.data[array.mask]=nodata_value # if masked, replace nodata values by nodata_value        
    if model_map is not None:
        with rasterio.open(model_map,'r') as mm:
            model=mm.read(band_model_map, masked=True) # take the first layer 'band_model_map' (1 is the first)
        array.data[model.mask]=nodata_value # replace nodata_value model_map pixel value of input_file by nodata_value (typically they are oceans)
    return array.data

def array2tif(output, array, model_map, nodata_value=None, cut_with_model_map=True, band_model_map=1, compress='lzw', verbose=True):
    arr=array.copy()
    if verbose==True: print('Writing ' + output)
    mm = rasterio.open(model_map,'r')
    arr = arr.astype(mm.meta['dtype']) 
    if cut_with_model_map==True:  # filling output map with nan pixels based on nan pixels of model_map 
        model=mm.read(band_model_map, masked=True) # take the first layer 'band_model_map' (1 is the first)
        arr[model.mask]=np.nan
    if nodata_value is not None: mm.meta['nodata']=nodata_value
    arr[np.isnan(arr)]=mm.meta['nodata'] # fill nan with nodata_value
    with rasterio.open(output,'w', compress=compress, **mm.meta) as ds:
        ds.write(arr, 1) # write array in the first band
        
def csv2dict(input_file, col1, col2, delimiter=','):
# create a dictonary using values of two columns of a csv file
    df=pd.read_csv(input_file, delimiter=delimiter)
    dictionary=dict(zip(list(df[col1]),list(df[col2])))
    return dictionary

def hmean(x,y):
# hamonic mean of two arrays
    x=np.array(x)
    y=np.array(y)
    suma = x+y
    suma[suma==0] = 1 # avoiding to divide by 0
    if((np.sign(x)*np.sign(y)==-1).any()): print ('WARNING: Different signs when calculating harmonic mean')
    return 2*x*y/suma

def mean(x,y):
# choosen mean for ranking land to be transfered
    return (x+y)/2.

#==========================================================================================================


def main_py(input_csv, output_folder, verbose_log=True):
# if verbose_log is a string, a log file is created with its name

    if isinstance(verbose_log, str):
        old_stdout = sys.stdout
        log_file = open(output_folder +verbose_log,'w') 
        sys.stdout = log_file

    print('')
    print(datetime.now())
    print('')
    
    files=csv2dict(input_csv, 'Variable', 'Value') # dictionary with parameters and pointing where data files are
    # inputs
    threshold_dominance =  float(files['threshold_for_dominance'])
    threshold_matrix_crops = float(files['threshold_matrix_crops'])
    threshold_matrix_grass = float(files['threshold_matrix_grass'])
    ls = int(files['land_sparing'])
    area_pixel = float(files['area_pixel'])
    countries_code_map_file=files['countries_code'] 
    ###countries_names_code_file=files['countries_names_code']
    frac_crops_file=files['LU_crops_present']
    frac_grass_file=files['LU_grass_present']
    suit_crops_file=files['suitability_crops']
    suit_grass_file=files['suitability_grass']
    country_targets_file = output_folder + files['country_targets']
    # outputs
    country_targets_transformed_file = output_folder + files['country_targets_transformed']
    frac_crops_transf_file = output_folder + files['LU_crops_present_transformed']
    frac_grass_transf_file = output_folder + files['LU_grass_present_transformed']
    propconvert_crops_file = output_folder + files['propconvert_crops']
    propconvert_grass_file = output_folder + files['propconvert_grass']
    spvar_crops_file = output_folder + files['spvar_crops']
    spvar_grass_file = output_folder + files['spvar_grass']
    area_matrix_crops_file = output_folder + files['area_matrix_crops']
    area_matrix_grass_file = output_folder + files['area_matrix_grass']
    
    # reading
    ###df_countries_names_code=pd.read_csv(countries_names_code_file, usecols=['CODE', 'NAME'])
    countries_code_map=tif2array(countries_code_map_file)
    frac_crops=tif2array(frac_crops_file, model_map=countries_code_map_file, nodata_value=0)
    frac_grass=tif2array(frac_grass_file, model_map=countries_code_map_file, nodata_value=0)
    frac_antro = frac_crops + frac_grass
    suit_grass=tif2array(suit_grass_file, model_map=countries_code_map_file, nodata_value=0)
    suit_crops=tif2array(suit_crops_file, model_map=countries_code_map_file, nodata_value=0)
    df_targets_ini = pd.read_csv(country_targets_file)
    
    
    area_crops = frac_crops * area_pixel
    area_grass = frac_grass * area_pixel
    
    # Initialise output maps
    frac_crops_transf = frac_crops.copy()
    frac_grass_transf = frac_grass.copy()
    propconvert_crops = np.zeros(countries_code_map.shape) 
    propconvert_grass = np.zeros(countries_code_map.shape)
    if ls==1:
        spvar_ls_crops_file = output_folder + files['spvar_landsparing_agr']
        spvar_ls_grass_file = output_folder + files['spvar_landsparing_past']
        spvar_crops=tif2array(spvar_ls_crops_file, model_map=countries_code_map_file, nodata_value=0)
        spvar_grass=tif2array(spvar_ls_grass_file, model_map=countries_code_map_file, nodata_value=0)
    elif ls==0:
        spvar_crops=np.ones(countries_code_map.shape)
        spvar_grass=np.ones(countries_code_map.shape)
    else:
        print('Error in land_sparing: '+str(ls))

    # creating new country target dataframe correcting names of columns
    colname_crops = [col for col in df_targets_ini if col.startswith('T_crops')] 
    colname_grass = [col for col in df_targets_ini if col.startswith('T_grass')]
    if (len(colname_crops)!=1) | (len(colname_grass)!=1): print('WARNING: Too many T_crops or T_grass columns')        
    df_targets = df_targets_ini.copy()[['CODE', 'NAME', colname_crops[0], colname_grass[0]]] # to be modified if columns have different names
    df_targets.rename(columns={colname_crops[0]:'T_crops_ini', colname_grass[0]:'T_grass_ini'}, inplace=True)
    df_targets['T_grass2crops'] = df_targets['T_crops2grass'] = 0.
    df_targets['T_crops_end'] = df_targets['T_crops_ini']
    df_targets['T_grass_end'] = df_targets['T_grass_ini']
    
    
    for k, country_code in enumerate(df_targets['CODE']):  
        print('')
        print('Working with region ' + str(country_code))
        index_bool_country = countries_code_map==country_code
        
        ######  TRANSFERING (country targets and LU transformed) ######################################################################################
        
        Tcrops = df_targets.loc[k,'T_crops_ini']
        Tgrass = df_targets.loc[k,'T_grass_ini']
        T = np.array([Tcrops, Tgrass])
        Tsum = Tcrops+Tgrass
        # checking for inconsistencies in targets
        area_crops_tot = np.sum(area_crops[index_bool_country])
        area_grass_tot = np.sum(area_grass[index_bool_country])
        area_noantro_tot = np.sum(index_bool_country)*area_pixel - area_crops_tot - area_grass_tot
        if (Tcrops>area_crops_tot) | (Tgrass>area_grass_tot):
            print('WARNING: target/s for restoration higher than anthropic area available')
            print('Targets: ' + str(Tcrops) + ', ' + str(Tgrass) + '. Anthropic area: ' + str(area_crops_tot) + ', ' + str(area_grass_tot))
        if (Tsum<-area_noantro_tot):
            print('WARNING: not enough non-anthropic area available to deforest')
            print('Targets: ' + str(Tcrops) + ', ' + str(Tgrass) + '. Non-anthropic area: ' + str(area_noantro_tot))     
        
        if (~np.isnan(T).any()) & (np.all(T)!=0) & (np.sign(Tcrops)!=np.sign(Tgrass)) : # only when targets have different signs
            print('Transfering. Initial targets: ' + str(Tcrops) + ', ' + str(Tgrass))
               
            if Tcrops<0: # grass will be lost in favour of crops
                print('Grass will be transformed into crops') 
                df_targets.loc[k,'T_crops_end'] = np.min([Tsum,0])
                df_targets.loc[k,'T_grass_end'] = np.max([Tsum,0]) 
                df_targets.loc[k,'T_grass2crops'] = area_transfer = Tgrass - df_targets.loc[k,'T_grass_end']
                index_bool_selec = (index_bool_country) & (frac_grass>0)
                index_notsorted = np.argwhere(index_bool_selec) # [ [x_i,y_i] , ...] format 
                # ranking grassland by a mean of normalised suitability of crops and opposite of normalised suitability of grass
                maxcrops = np.nanmax(suit_crops[index_bool_selec])
                if(maxcrops==0): maxcrops=1 # avoiding dividing by 0; all suit crops is 0
                suit_selec_crops = suit_grass[index_bool_selec]/maxcrops
                maxgrass = np.nanmax(suit_grass[index_bool_selec])
                if(maxgrass==0): maxgrass=1 # avoiding dividing by 0; all suit grass is 0
                suit_selec_grass_op = 1. - suit_grass[index_bool_selec]/maxgrass
                suit_compos = mean(suit_selec_crops, suit_selec_grass_op) # 1D array; only those that have grass in the country
                index_sorted = tuple(np.transpose( index_notsorted[ np.unravel_index(np.argsort(suit_compos.ravel())[::-1], suit_compos.shape) ] )) # [[x],[y]] format; sorted list of tuple indexes
                area_cum = np.cumsum(area_grass[index_sorted]) # cumulating area according with previous sorting
                index_chosen = tuple(np.array(index_sorted)[ :, 0:np.sum(area_cum < area_transfer) +1]) # indexes of pixels that will be transformed
                # transfering
                print(str(len(index_chosen[0])) + ' /' + str(len(index_sorted[0])) + ' pixels transformed')
                frac_crops_transf[index_chosen] += frac_grass[index_chosen]  # updating frac
                frac_grass_transf[index_chosen] = 0.
                    
            elif Tgrass<0: # crops will be lost in favour of grass
                print('Crops will be transformed into grass')
                df_targets.loc[k,'T_crops_end'] = np.max([Tsum,0])
                df_targets.loc[k,'T_grass_end'] = np.min([Tsum,0])
                df_targets.loc[k,'T_crops2grass'] = area_transfer = Tcrops - df_targets.loc[k,'T_crops_end']
                index_bool_selec = (index_bool_country) & (frac_crops>0)
                index_notsorted = np.argwhere(index_bool_selec) # [ [x_i,y_i] , ...] format           
                # ranking cropland by a mean of normalised suitability of grass and opposite of normalised suitability of crops
                maxgrass = np.nanmax(suit_grass[index_bool_selec])
                if(maxgrass==0): maxgrass=1 # avoiding dividing by 0; all suit grass is 0
                suit_selec_grass = suit_grass[index_bool_selec]/maxgrass
                maxcrops = np.nanmax(suit_crops[index_bool_selec])
                if(maxcrops==0): maxcrops=1 # avoiding dividing by 0; all suit crops is 0
                suit_selec_crops_op = 1. - suit_crops[index_bool_selec]/maxcrops
                suit_compos = mean(suit_selec_grass, suit_selec_crops_op) # 1D array; only those that have crops in the country
                index_sorted = tuple(np.transpose( index_notsorted[ np.unravel_index(np.argsort(suit_compos.ravel())[::-1], suit_compos.shape) ] )) # [[x],[y]] format; sorted list of tuple indexes
                area_cum = np.cumsum(area_crops[index_sorted]) # cumulating area according with previous sorting
                index_chosen = tuple(np.array(index_sorted)[ :, 0:np.sum(area_cum < area_transfer) +1]) # indexes of pixels that will be transformed
                # transfering
                print(str(len(index_chosen[0])) + ' /' + str(len(index_sorted[0])) + ' pixels transformed')
                frac_grass_transf[index_chosen] += frac_crops[index_chosen]  # updating frac
                frac_crops_transf[index_chosen] = 0.
                    
            else:
                print('Error in initial targets: ' + str(Tcrops) + ', ' + str(Tgrass) )
        
        
        ######  CREATING prop_convert AND spvar  ######################################################################################        
        
        Tcrops = df_targets.loc[k,'T_crops_end']
        Tgrass = df_targets.loc[k,'T_grass_end']
        T = np.array([Tcrops, Tgrass])          
        if (Tcrops>0) & (Tgrass>0):  # restauration for both 
            index_bool_country_antro = (index_bool_country) & (frac_antro>0.) # pixels with known antropic area
            propconvert_crops[index_bool_country_antro] = frac_crops[index_bool_country_antro] / (frac_antro[index_bool_country_antro])
            propconvert_grass[index_bool_country_antro] = frac_grass[index_bool_country_antro] / (frac_antro[index_bool_country_antro])
            # dominance
            index_bool_cropsdominant = (index_bool_country) & (propconvert_crops>threshold_dominance)
            propconvert_crops[index_bool_cropsdominant] = 1.
            propconvert_grass[index_bool_cropsdominant] = 0.
            spvar_grass[index_bool_cropsdominant] = 0.
            print('Crops dominating: ' + str(np.sum(index_bool_cropsdominant)) + ' pixels')
            index_bool_grassdominant = (index_bool_country) & (propconvert_grass>threshold_dominance)
            propconvert_grass[index_bool_grassdominant] = 1.
            propconvert_crops[index_bool_grassdominant] = 0.
            spvar_crops[index_bool_grassdominant] = 0. 
            print('Grass dominating: ' + str(np.sum(index_bool_grassdominant)) + ' pixels')       
        elif ((Tcrops<0)&(Tgrass<0)) | (np.sign(Tcrops)*np.sign(Tgrass)==0): # explicitly cover all possible allowed cases to avoid problems
            Tsum = Tcrops + Tgrass
            if Tsum!=0: propconvert_crops[index_bool_country] = Tcrops/Tsum  
            if Tsum!=0: propconvert_grass[index_bool_country] = Tgrass/Tsum
            if (Tgrass>0)&(Tcrops==0): spvar_crops[index_bool_country] = 0.
            if (Tcrops>0)&(Tgrass==0): spvar_grass[index_bool_country] = 0.
        else:
            print('Error in region ' + str(country_code) + '. Targets after transformation: ' + str(Tcrops) + ', ' + str(Tgrass))
    
    print('')
    
    ######  CREATING matrix (just masking by suitability threshold)
    area_matrix_crops = np.zeros(countries_code_map.shape)
    area_matrix_crops[suit_crops>threshold_matrix_crops] = area_pixel
    area_matrix_grass = np.zeros(countries_code_map.shape)
    area_matrix_grass[suit_grass>threshold_matrix_grass] = area_pixel
    
    # Writing            
    df_targets.to_csv(country_targets_transformed_file, index=False, na_rep='nan')
    array2tif(frac_crops_transf_file, frac_crops_transf, countries_code_map_file)
    array2tif(frac_grass_transf_file, frac_grass_transf, countries_code_map_file)
    array2tif(propconvert_crops_file, propconvert_crops, countries_code_map_file)       
    array2tif(propconvert_grass_file, propconvert_grass, countries_code_map_file)         
    array2tif(spvar_crops_file, spvar_crops, countries_code_map_file) 
    array2tif(spvar_grass_file, spvar_grass, countries_code_map_file) 
    array2tif(area_matrix_crops_file, area_matrix_crops, countries_code_map_file) 
    array2tif(area_matrix_grass_file, area_matrix_grass, countries_code_map_file) 
    
    # Just for debugging
    #frac_crops_delta = frac_crops_transf - frac_crops
    #frac_grass_delta = frac_grass_transf - frac_grass
    #directory_debug = output_folder + 'debug/'
    #if not os.path.exists(directory_debug): os.makedirs(directory_debug)
    #array2tif(directory_debug + 'Delta_frac_crops.tif', frac_crops_delta, countries_code_map_file)
    #array2tif(directory_debug + 'Delta_frac_grass.tif', frac_grass_delta, countries_code_map_file)
    
    print('')
    print(datetime.now())

    if isinstance(verbose_log, str):    
        sys.stdout = old_stdout
        log_file.close()
        
    #os.system('cp ' + input_csv + ' ' + output_folder) # just in case


