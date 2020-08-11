# -*- coding: utf-8 -*-

import numpy as np
import rasterio
import pandas as pd


def tif2array(input_file, masked=True, nodata_value=np.nan, band=1, model_map=None, band_model_map=1, verbose=True):
    if verbose==True: print('Reading ' + input_file)
    with rasterio.open(input_file,'r') as ds:
        array=ds.read(band, masked=masked) # take the first layer 'band' (1 is the first)
    if masked==True: array.data[array.mask]=nodata_value # if masked, replace nodata values by nodata_value        
    if model_map is not None:
        with rasterio.open(model_map,'r') as mm:
            model=mm.read(band_model_map, masked=True) # take the first layer 'band_model_map' (1 is the first)
        array.data[model.mask]=np.nan # replace nodata_value model_map pixel value of input_file by nan (typically they are oceans)
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

def fill_average(input_array, model, weights, list_model=None, verbose=False):
# replace non-positive values of input array by the weighted average of the regions defined by model
# it also provide two lists of 2D-lists with the codes in model (or list_model) and the averages, and the codes and the sum of fractions
# if list_model is provided, it only take into account the values in model on that list
#  input_array, model, weights have to have the same size
    inp=input_array.copy()
    output = inp.copy()
    averages=[]
    sums_frac=[]
    inp[(np.isnan(inp)&(~np.isnan(model)))] = -1 # pixels that will be replaced 
    inp[np.isnan(model)]=-9 # pixels outside of the regiones (e.g. oceans)
    if (verbose==True): print('Total to be replaced: '+ str(np.sum(inp==-1)) + ' (' + str(int(100.*np.sum(inp==-1)/np.sum(~np.isnan(model)))) + '%)')
    if list_model==None:
        list_model=np.unique(model)
        list_model=list_model[~np.isnan(list_model)]
    for code in list_model:
        bool_region = model==code
        bool_valid =  bool_region & (inp>0) # pixels that are used to calculate the average
        bool_replace = bool_region & (inp==-1) # pixels to be replaced
        w = weights[bool_valid]
        sum_frac=np.sum(w)
        average = np.average(inp[bool_valid], weights=w) if np.sum(sum_frac)>0 else 0
        output[bool_replace]=average
        averages.append([code, average])
        sums_frac.append([code, sum_frac])
        if (verbose==True) & (np.sum(bool_region)>0): print( 'Replaced ' + str(np.sum(bool_replace)) + ' (' + str(int(100.*np.sum(bool_replace)/np.sum(bool_region))) + '%) of region ' + str(code) + ' with ' + str(average) )
    return output, averages, sums_frac 

def fill_regions(input_array, list_regions_values, model, verbose=False):
# replace non-positive pixel of input array by a value according to the region (list_regions_values) where it is defined by model array.
    output = input_array.copy()
    for code_value in list_regions_values:
        code=code_value[0]
        value=code_value[1]
        bool_region_index = model==code
        bool_index = ( ~(output>0) & (model==code) )
        if (verbose==True):
            n_tot = np.sum(bool_region_index)
            if (n_tot==0): n_tot=-1. # to avoid dividing by zero
            print('Filling ' + str(np.sum(bool_index)) + ' (' + str(int(100.*np.sum(bool_index)/n_tot)) + '%) of region ' + str(code) + ' with ' + str(value) )
        output[bool_index] = value
    return output