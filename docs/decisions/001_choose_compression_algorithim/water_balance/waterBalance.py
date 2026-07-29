from pathlib import Path

import dask
import xarray as xr
from tqdm import tqdm

dask.config.set(**{'array.slicing.split_large_chunks': True})
import time
import warnings

import pandas as pd

warnings.simplefilter("ignore")


def timing_decorator(func):
    '''@timing_decorator ontop of function you want to time'''
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        execution_time_min = execution_time / 60
        print(f"\n{func.__name__} took: {execution_time_min:.1f}min / {execution_time:.1f}s to run ")
        return result    
    return wrapper

class waterBalance:
    """Get Water Balance for a given simulation directory.
    """
    def waterBalance(simDirectory, cellAreaFile, saveFolder):
        simDirectory = Path(simDirectory)
        saveFolder = Path(saveFolder)
        saveFolder.mkdir(parents=True, exist_ok=True)
        tpFiles = sorted(simDirectory.glob('**/*netcdf/precipitation_annuaTot_output.nc'))
        evapFiles = sorted(simDirectory.glob('**/*netcdf/totalEvaporation_annuaTot_output.nc'))
        runoffFiles = sorted(simDirectory.glob('**/*netcdf/totalRunoff_monthTot_output.nc'))

        def readCellArea(cellAreaFile, tp_ds):            
            cellAreaFile = Path(cellAreaFile)
            cellArea_ds = xr.open_dataset(cellAreaFile, chunks='auto').to_array().squeeze()
            cellArea_ds = cellArea_ds.sel(lat=slice(tp_ds.lat.max(), tp_ds.lat.min()), lon=slice(tp_ds.lon.min(), tp_ds.lon.max()))
            return cellArea_ds
        
        def mergeSimFiles(files):
            def annualMean(files):
                ds = xr.open_mfdataset(files, chunks={}, engine='h5netcdf', parallel=True, concat_dim="time", combine="nested").sel(time=slice('1985-01-01', None))
                if 'annua' in files[0]:
                    return ds.mean('time', skipna=False).compute()
                
                if 'month' in files[0]:
                    ds = ds.resample(time='Y').sum(skipna=False)
                    return ds.mean('time', skipna=False).compute()
            clones = {}
            for nc_file in files:
                parent_dir = str(nc_file.parents[0])
                
                if parent_dir in clones:
                    clones[parent_dir].append(str(nc_file))
                else:
                    clones[parent_dir] = [str(nc_file)]
            count = 0
            for clone, files in tqdm(clones.items(), desc="Reading:"):
                count = count + 1
                maskArray =annualMean(files)
                if count == 1: 
                    ds_final=maskArray
                else:
                    ds_final = ds_final.combine_first(maskArray)
                    
            return ds_final

        tp_ds, evap_ds, runoff_ds = mergeSimFiles(tpFiles), mergeSimFiles(evapFiles), mergeSimFiles(runoffFiles)
        cellArea_ds = readCellArea(cellAreaFile, tp_ds) 

        tp_ds = (tp_ds.reindex(lat=cellArea_ds.lat, lon=cellArea_ds.lon, method='nearest') * cellArea_ds)*1E-9
        tp_ds = tp_ds.chunk({'lat': 5000, 'lon': 5000})
        tp_ds.to_zarr(saveFolder / 'tp.zarr', mode='w')
        
        evap_ds = (evap_ds.reindex(lat=cellArea_ds.lat, lon=cellArea_ds.lon, method='nearest') * cellArea_ds)*1E-9
        evap_ds = evap_ds.chunk({'lat': 5000, 'lon': 5000})
        evap_ds.to_zarr(saveFolder / 'evap.zarr', mode='w')

        runoff_ds = (runoff_ds.reindex(lat=cellArea_ds.lat, lon=cellArea_ds.lon, method='nearest') * cellArea_ds)*1E-9
        runoff_ds = runoff_ds.chunk({'lat': 5000, 'lon': 5000})
        runoff_ds.to_zarr(saveFolder / 'runoff.zarr', mode='w')

        def calcWaterBalance(tp_ds, evap_ds, runoff_ds):
            tp = tp_ds.to_array().squeeze().sum().values
            evap = evap_ds.to_array().squeeze().sum().values
            runoff = runoff_ds.to_array().squeeze().sum().values
            delta_s = tp - evap - runoff
            wb_df = pd.DataFrame({'precip': [tp],
                                  'evap': [evap],
                                  'runoff': [runoff], 
                                  'DeltaWaterStor': [delta_s]})
            wb_df = wb_df.round(0).astype(int)
            wb_df.to_json(saveFolder / 'waterBalance.json')
            print(wb_df)
            
        calcWaterBalance(tp_ds, evap_ds, runoff_ds)

