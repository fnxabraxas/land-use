# Land Use 
Set of scripts and functions to complement land use optimization models.

- aggregated_suitability: generation of raster maps and regional target tables based on suitability to be used on the target equations of the optimizer. It is assumed that pasture and agriculture suitability can be aggregated. It includes scripts, auxiliary functions, a short summary on how to use the outputs, and a full mathematical development of the model. It is necessary to include a csv file with the paths to the inputs files (an example is provided).

- separated_suitability: generation of raster maps and regional target tables based on suitability to be used on the target equations of the optimizer. Pasture and agriculture suitability are computed separately. It includes scripts, auxiliary functions, a short summary on how to use the outputs, and a full mathematical development of the model. It is necessary to include a csv file with the paths to the inputs files (an example is provided).

- separated_area: generation of objects necessary to set the regional targets equations of the optimizer assuming that these targets are based on area. Pasture and agriculture are initially evaluated separately. Documentation is still under construction.

