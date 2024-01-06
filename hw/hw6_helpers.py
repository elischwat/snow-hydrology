import pandas as pd 
import datetime
import xarray as xr
import os
import urllib
import geopy
import geopy.distance
import shapely.geometry
import py3dep
import pytz
import numpy as np
import rasterio
import geopandas as gpd
from sklearn import linear_model

from astral import LocationInfo
from astral.sun import sun

from metpy.units import units

import turbpy

ROTATION_SUPPORTED_MEASUREMENTS = [
     'u',     'v',     
     'u_w_',     'v_w_', 
     'u_tc_',     'v_tc_',    
    'u_h2o_',     'v_h2o_',    
    ]

def download_sos_data_day(date = '20221101', local_download_dir = 'sosnoqc', cache=False,  planar_fit = False):
    """Download a netcdf file from the ftp url provided by the Earth Observing Laboratory at NCAR.
    Data is the daily data reynolds averaged to 5 minutes.

    Args:
        date (str, optional): String version of a date. Defaults to '20221101'.
        local_download_dir (str, optional): Directory to which files will be downloaded. Defaults to 'sosnoqc'; this directory will be created if it does not already exist.

    Returns:
        _type_: _description_
    """
    base_url = 'ftp.eol.ucar.edu'
    if planar_fit:
        path = 'pub/archive/isfs/projects/SOS/netcdf/noqc_geo_tiltcor/'
    else:
        path = 'pub/archive/isfs/projects/SOS/netcdf/noqc_geo'
    
    if planar_fit:
        file_example =  f'isfs_sos_tiltcor_{date}.nc'

    else:
        file_example = f'isfs_{date}.nc'

    os.makedirs(local_download_dir, exist_ok=True)

    full_file_path = os.path.join('ftp://', base_url, path, file_example)
    if planar_fit:
        download_file_path = os.path.join(local_download_dir, 'planar_fit', file_example)
    else:
        download_file_path = os.path.join(local_download_dir, file_example)
    

    if cache and os.path.isfile(download_file_path):
        print(f"Caching...skipping download for {date}")
    else:
        urllib.request.urlretrieve(
            full_file_path,
            download_file_path   
        )

    return download_file_path

def height_from_variable_name(name):
    """Parse instrument/sensor height from EOL variable names.

    Args:
        name (_type_): _description_

    Returns:
        _type_: _description_
    """
    # handle the soil moisture depths
    if '_0_6cm' in name:
        return -0.006
    elif '_1_9cm' in name:
        return -0.019
    elif '_3_1cm' in name:
        return -0.031
    elif '_4_4cm' in name:
        return -0.044
    elif '_8_1cm' in name:
        return -0.081
    elif '_9_4cm' in name:
        return -0.094
    elif '_10_6cm' in name:
        return -.106
    elif '_11_9cm' in name:
        return -.119
    elif '_18_1cm' in name:
        return -.181
    elif '_19_4cm' in name:
        return -.194
    elif '_20_6cm' in name:
        return -.206
    elif '_21_9cm' in name:
        return -.219
    elif '_28_1cm' in name:
        return -.281
    elif '_29_4cm' in name:
        return -.294
    elif '_30_6cm' in name:
        return -.306
    elif '_31_9cm' in name:
        return -.319
    #snow temperature depths - these should be handled before tower depths because, 
    # for example '_4m_' is in '_0_4m_'
    elif '_0_4m_' in name:
        return 0.4
    elif '_0_5m_' in name:
        return 0.5
    elif '_0_6m_' in name:
        return 0.6
    elif '_0_7m_' in name:
        return 0.7
    elif '_0_8m_' in name:
        return 0.8
    elif '_0_9m_' in name:
        return 0.9
    elif '_1_0m_' in name:
        return 1.0
    elif '_1_1m_' in name:
        return 1.1
    elif '_1_2m_' in name:
        return 1.2
    elif '_1_3m_' in name:
        return 1.3
    elif '_1_4m_' in name:
        return 1.4
    elif '_1_5m_' in name:
        return 1.5
    # tower depths
    elif '_1m_' in name:
        return 1
    elif '_2m_' in name:
        return 2
    elif '_3m_' in name:
        return 3
    elif '_4m_' in name:
        return 4
    elif '_5m_' in name:
        return 5
    elif '_6m_' in name:
        return 6
    elif '_7m_' in name:
        return 7
    elif '_8m_' in name:
        return 8
    elif '_9m_' in name:
        return 9
    elif '_10m_' in name:
        return 10
    elif '_11m_' in name:
        return 11
    elif '_12m_' in name:
        return 12
    elif '_13m_' in name:
        return 13
    elif '_14m_' in name:
        return 14
    elif '_15m_' in name:
        return 15
    elif '_16m_' in name:
        return 16
    elif '_17m_' in name:
        return 17
    elif '_18m_' in name:
        return 18
    elif '_19m_' in name:
        return 19
    elif '_20m_' in name:
        return 20
    # surface measurements
    elif name.startswith('Tsurf_'):
        return 0

def tower_from_variable_name(name):
    """Parse instrument/sensor tower from EOL variable names.

    Args:
        name (_type_): _description_

    Returns:
        _type_: _description_
    """
    if name.endswith('_d'):
        return 'd'
    elif name.endswith('_c'):
        return 'c'
    elif name.endswith('_ue'):
        return 'ue'
    elif name.endswith('uw'):
        return 'uw'

def measurement_from_variable_name(name):
    """Provide plain text measurement name from EOL variable names.

    Args:
        name (_type_): _description_

    Returns:
        _type_: _description_
    """
    # VARIABLE NAMES THAT COME FROM THE SOSNOQC DATASETS
    if any([prefix in name for prefix in ['SF_avg_1m_ue', 'SF_avg_2m_ue']]): # these are the only two 
        return 'snow flux'
    elif any([prefix in name for prefix in ['P_10m_', 'P_20m_']]):
        return 'pressure'
    elif any([prefix in name for prefix in ['dir_1m_','dir_2m_','dir_3m_','dir_5m_','dir_10m_','dir_15m_','dir_20m_']]):
        return 'wind direction'
    elif any([prefix in name for prefix in ['spd_1m_', 'spd_2m_', 'spd_3m_', 'spd_5m_', 'spd_10m_', 'spd_15m_', 'spd_20m_']]):
        return 'wind speed'
    elif any([prefix in name for prefix in ['u_1m_','u_2m_','u_3m_','u_5m_','u_10m_','u_15m_','u_20m_']]):
        return 'u'
    elif any([prefix in name for prefix in ['v_1m_','v_2m_','v_3m_','v_5m_','v_10m_','v_15m_','v_20m_']]):
        return 'v'
    elif any([prefix in name for prefix in ['w_1m_','w_2m_','w_3m_','w_5m_','w_10m_','w_15m_','w_20m_']]):
        return 'w'
    elif any([prefix in name for prefix in ['u_u__1m_', 'u_u__2m_', 'u_u__3m_', 'u_u__5m_', 'u_u__10m_', 'u_u__15m_', 'u_u__20m_']]):
        return 'u_u_'
    elif any([prefix in name for prefix in ['v_v__1m_', 'v_v__2m_', 'v_v__3m_', 'v_v__5m_', 'v_v__10m_', 'v_v__15m_', 'v_v__20m_']]):
        return 'v_v_'
    elif any([prefix in name for prefix in ['w_w__1m_', 'w_w__2m_', 'w_w__3m_', 'w_w__5m_', 'w_w__10m_', 'w_w__15m_', 'w_w__20m_']]):
        return 'w_w_'
    elif any([prefix in name for prefix in ['u_w__1m_','u_w__2m_','u_w__3m_','u_w__5m_','u_w__10m_','u_w__15m_','u_w__20m_']]):
        return 'u_w_'
    elif any([prefix in name for prefix in ['v_w__1m_','v_w__2m_','v_w__3m_','v_w__5m_','v_w__10m_','v_w__15m_','v_w__20m_']]):
        return 'v_w_'
    elif any([prefix in name for prefix in ['u_v__1m_','u_v__2m_','u_v__3m_','u_v__5m_','u_v__10m_','u_v__15m_','u_v__20m_']]):
        return 'u_v_'
    elif any([prefix in name for prefix in ['u_tc__1m_','u_tc__2m_','u_tc__3m_','u_tc__5m_','u_tc__10m_','u_tc__15m_','u_tc__20m_']]):
        return 'u_tc_'
    elif any([prefix in name for prefix in ['v_tc__1m_','v_tc__2m_','v_tc__3m_','v_tc__5m_','v_tc__10m_','v_tc__15m_','v_tc__20m_']]):
        return 'v_tc_'
    elif any([prefix in name for prefix in ['w_tc__1m_','w_tc__2m_','w_tc__3m_','w_tc__5m_','w_tc__10m_','w_tc__15m_','w_tc__20m_']]):
        return 'w_tc_'
    elif any([prefix in name for prefix in ['u_h2o__1m_','u_h2o__2m_','u_h2o__3m_','u_h2o__5m_','u_h2o__10m_','u_h2o__15m_','u_h2o__20m_']]):
        return 'u_h2o_'
    elif any([prefix in name for prefix in ['v_h2o__1m_','v_h2o__2m_','v_h2o__3m_','v_h2o__5m_','v_h2o__10m_','v_h2o__15m_','v_h2o__20m_']]):
        return 'v_h2o_'
    elif any([prefix in name for prefix in ['w_h2o__1m_','w_h2o__2m_','w_h2o__3m_','w_h2o__5m_','w_h2o__10m_','w_h2o__15m_','w_h2o__20m_']]):
        return 'w_h2o_'
    elif any([prefix in name for prefix in ['T_1m_', 'T_2m_', 'T_3m_', 'T_4m_', 'T_5m_', 'T_6m_', 'T_7m_', 'T_8m_', 'T_9m_', 'T_10m_', 'T_11m_', 'T_12m_', 'T_13m_', 'T_14m_', 'T_15m_', 'T_16m_', 'T_17m_', 'T_18m_', 'T_19m_', 'T_20m_']]):
        return 'temperature'
    elif any([prefix in name for prefix in ['RH_1m_', 'RH_2m_', 'RH_3m_', 'RH_4m_', 'RH_5m_', 'RH_6m_', 'RH_7m_', 'RH_8m_', 'RH_9m_', 'RH_10m_', 'RH_11m_', 'RH_12m_', 'RH_13m_', 'RH_14m_', 'RH_15m_', 'RH_16m_', 'RH_17m_', 'RH_18m_', 'RH_19m_', 'RH_20m_']]):
        return 'RH'
    elif any([prefix in name for prefix in ['tc_1m', 'tc_2m', 'tc_3m', 'tc_5m', 'tc_10m', 'tc_15m', 'tc_20m']]):
        return 'virtual temperature'
    elif any([prefix in name for prefix in ['Tsoil_3_1cm_d', 'Tsoil_8_1cm_d', 'Tsoil_18_1cm_d', 'Tsoil_28_1cm_d', 'Tsoil_4_4cm_d', 'Tsoil_9_4cm_d', 'Tsoil_19_4cm_d', 'Tsoil_29_4cm_d', 'Tsoil_0_6cm_d',  'Tsoil_10_6cm_d', 'Tsoil_20_6cm_d', 'Tsoil_30_6cm_d', 'Tsoil_1_9cm_d', 'Tsoil_11_9cm_d', 'Tsoil_21_9cm_d', 'Tsoil_31_9cm_d']]):
        return 'soil temperature'
    elif name == 'Gsoil_d':
        return 'ground heat flux'
    elif name == 'Qsoil_d':   
        return 'soil moisture'
    elif name == 'Rsw_in_9m_d':
        return 'shortwave radiation incoming'
    elif name == 'Rsw_out_9m_d':
        return 'shortwave radiation outgoing'
    elif name in ['Vtherm_c', 'Vtherm_d', 'Vtherm_ue', 'Vtherm_uw']:
        return "Vtherm"
    elif name in ['Vpile_c', 'Vpile_d', 'Vpile_ue', 'Vpile_uw']:
        return "Vpile"
    elif name in ['IDir_c', 'IDir_d', 'IDir_ue', 'IDir_uw']:
        return "IDir"
    elif any([prefix in name for prefix in ['Tsnow_0_4m_', 'Tsnow_0_5m_', 'Tsnow_0_6m_', 'Tsnow_0_7m_', 'Tsnow_0_8m_', 'Tsnow_0_9m_', 'Tsnow_1_0m_', 'Tsnow_1_1m_', 'Tsnow_1_2m_', 'Tsnow_1_3m_', 'Tsnow_1_4m_', 'Tsnow_1_5m_']]):
        return 'snow temperature'
    # VARIABLE NAMES THAT do not COME FROM THE SOSNOQC DATASETS but we add and use a naming schema consistent with SOSNOQC dataset naming schema
    elif any([prefix in name for prefix in ['Tpot_1m_', 'Tpot_2m_', 'Tpot_3m_', 'Tpot_4m_', 'Tpot_5m_', 'Tpot_6m_', 'Tpot_7m_', 'Tpot_8m_', 'Tpot_9m_', 'Tpot_10m_', 'Tpot_11m_', 'Tpot_12m_', 'Tpot_13m_', 'Tpot_14m_', 'Tpot_15m_', 'Tpot_16m_', 'Tpot_17m_', 'Tpot_18m_', 'Tpot_19m_', 'Tpot_20m_']]):
        return 'potential temperature'
    elif name == 'Rlw_in_9m_d':
        return 'longwave radiation incoming'
    elif name == 'Rlw_out_9m_d':
        return 'longwave radiation outgoing'
    elif name in ['Tsurf_c', 'Tsurf_d', 'Tsurf_ue', 'Tsurf_uw', 'Tsurf_rad_d']:
        return "surface temperature"
    elif any([prefix in name for prefix in ['tke_1m_',    'tke_2m_',    'tke_3m_',    'tke_5m_',    'tke_10m_',    'tke_15m_',    'tke_20m_']]):
        return "turbulent kinetic energy"
    

        
def merge_datasets_with_different_variables(ds_list, dim):
    """ Take a list of datasets and merge them using xr.merge. First check that the two datasets
    have the same data vars. If they do not, missing data vars in each dataset are added with nan values
    so that the two datasets have the same set of data vars. NOTE: This gets slow with lots of datasets

    Args:
        ds_list (_type_): _description_
        dim (_type_): _description_
    """
    def _merge_datasets_with_different_variables(ds1, ds2, dim):
        vars1 = set(ds1.data_vars)
        vars2 = set(ds2.data_vars)
        in1_notin2 = vars1.difference(vars2)
        in2_notin1 = vars2.difference(vars1)
        # add vars with NaN values to ds1
        for v in in2_notin1:
            ds1[v] = xr.DataArray(coords=ds1.coords, dims=ds1.dims)
        # add vars with NaN values to ds2
        for v in in1_notin2:
            ds2[v] = xr.DataArray(coords=ds2.coords, dims=ds2.dims)
        return xr.concat([ds1, ds2], dim=dim)

    new_ds = ds_list.pop(0)
    while ds_list:
        new_ds = _merge_datasets_with_different_variables(
            new_ds,
            ds_list.pop(0),
            dim=dim
        )
    return new_ds

def modify_df_timezone(df, source_tz, target_tz, time_col='time'):
    df = df.copy()
    df[time_col] = df[time_col].dt.tz_localize(source_tz).dt.tz_convert(target_tz).dt.tz_localize(None)
    return df

def get_tidy_dataset(ds, variable_names):
    """Convert an SoS netcdf xr.DataSet into a dataframe with time, height, tower, and measurement as indexes in a tidy dataset.

    Args:
        ds (_type_): Dataset to convert
        variable_names (_type_): Variable names that you want operated on. Variables may not be supported.
    """
    tidy_df = ds[variable_names].to_dataframe().reset_index().melt(id_vars='time', value_vars=variable_names)
    tidy_df['height'] = tidy_df['variable'].apply(height_from_variable_name)
    tidy_df['tower'] = tidy_df['variable'].apply(tower_from_variable_name)
    tidy_df['measurement'] = tidy_df['variable'].apply(measurement_from_variable_name)
    return tidy_df


def apogee2temp(ds,tower):
    # hard-coded sensor-specific calibrations
    Vref = 2.5
    ID = ds[f"IDir_{tower}"]
    sns = [136, 137, 138, 139, 140]
    im = [ sns.index(x) if x in sns else None for x in ID ][0]
    # unclear if we want these, or scaled up versions
    mC0 = [57508.575,56653.007,58756.588,58605.7861, 58756.588][im] * 1e5
    mC1 = [289.12189,280.03380,287.12487,285.00285, 287.12487][im] * 1e5
    mC2 = [2.16807,2.11478,2.11822,2.08932, 2.11822][im] * 1e5
    bC0 = [-168.3687,-319.9362,-214.5312,-329.6453, -214.5312][im]* 1e5
    bC1 = [-0.22672,-1.23812,-0.59308,-1.24657, -0.59308][im]* 1e5
    bC2 = [0.08927,0.08612,0.10936,0.09234, 0.10936][im]* 1e5
    # read data
    Vtherm = ds[f"Vtherm_{tower}"]
    Vpile = ds[f"Vpile_{tower}"]*1000
    # calculation of detector temperature from Steinhart-Hart
    Rt = 24900.0/((Vref/Vtherm) - 1)
    Ac = 1.129241e-3
    Bc = 2.341077e-4
    Cc = 8.775468e-8
    TDk = 1/(Ac + Bc*np.log(Rt) + Cc*(np.log(Rt)**3))
    TDc = TDk - 273.15
    # finally, calculation of "target" temperature including thermopile measurement
    m = mC2*TDc**2 + mC1*TDc + mC0
    b = bC2*TDc**2 + bC1*TDc + bC0
    TTc = (TDk**4 + m*Vpile + b)**0.25 - 273.15
    # sufs = suffixes(TTc,leadch='') # get suffixes
    # dimnames(TTc)[[2]] = paste0("Tsfc.Ap.",sufs)
    TTc = TTc * units('celsius')
    return TTc