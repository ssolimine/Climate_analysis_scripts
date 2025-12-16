import os
import xarray as xr
import numpy as np
import pandas as pd
import pyresample.kd_tree as pykd
import pytz
from netCDF4 import Dataset
from pyresample.geometry import GridDefinition

"""
This script resamples the spatial grid of various ERA5 datasets and the MSWEP precipitation dataset onto the spatial 
grid of the CLAUS dataset (0.33 degree lat lon resolution). The kd-tree nearest neighbor method is used to interpolate 
individual grid cell values from each ERA5 dataset onto the closest corresponding CLAUS (WIG filtered) grid cell location.
Input is a monthly NetCDF file of each variable with the exact same temporal, 3-hourly resolution and includes any 
missing dates (i.e. the time dimension of all datasets align perfectly). 
The output is new monthly NetCDF files of the spatially resampled data with the same temporal resolution as the input.
"""

# Define the domain of interest (Equatorial Africa)
lat_min, lat_max = -5, 5
lon_min, lon_max = 5, 40

# Define the years and months to process
years = range(1984, 2010)  # Adjust range as needed
months = [2, 3, 4, 5]  # February through May


# Function to convert time array to datetime.datetime objects
def convert_time(time_array):
    return np.array([pd.to_datetime(t).to_pydatetime() for t in time_array])


def write_monthly_nc(output_dir, year, month, times, lats, lons, data, var_name, units, long_name, levels=None):
    """
    Writes interpolated data to a NetCDF file for a given variable.
    Handles variables with or without pressure levels.
    """
    os.makedirs(output_dir, exist_ok=True)
    if levels is not None:
        output_file = os.path.join(output_dir, f'{year}_{month:02d}_3hourly_{var_name}.nc')
    else:
        output_file = os.path.join(output_dir, f'{year}_{month:02d}_3hourly_{var_name}.nc')
    print(f"Writing NetCDF file for {year}-{month:02d}, variable: {var_name}")

    # Check dimensions before writing
    expected_shape = (len(times), len(levels), len(lats), len(lons)) if levels is not None else (
        len(times), len(lats), len(lons))
    if data.shape != expected_shape:
        raise ValueError(f"Shape mismatch: Expected {expected_shape}, got {data.shape}")

    with Dataset(output_file, 'w', format='NETCDF4') as ncfile:
        # Dimensions
        time_dim = ncfile.createDimension('time', len(times))
        lat_dim = ncfile.createDimension('lat', len(lats))
        lon_dim = ncfile.createDimension('lon', len(lons))
        if levels is not None:
            level_dim = ncfile.createDimension('level', len(levels))

        # Variables
        times_var = ncfile.createVariable('time', 'f8', ('time',))
        lats_var = ncfile.createVariable('lat', 'f4', ('lat',))
        lons_var = ncfile.createVariable('lon', 'f4', ('lon',))
        if levels is not None:
            levels_var = ncfile.createVariable('level', 'f4', ('level',))

        if levels is not None:
            data_var = ncfile.createVariable(var_name, 'f4', ('time', 'level', 'lat', 'lon'), fill_value=np.nan)
        else:
            data_var = ncfile.createVariable(var_name, 'f4', ('time', 'lat', 'lon'), fill_value=np.nan)

        # Write data
        times_var[:] = np.array([pd.Timestamp(t).replace(tzinfo=pytz.UTC).timestamp() for t in times])  # Convert to seconds since 1970
        lats_var[:] = lats
        lons_var[:] = lons
        if levels is not None:
            levels_var[:] = levels
        if levels is not None:
            data_var[:, :, :, :] = data
        else:
            data_var[:, :, :] = data

        # Add attributes
        times_var.units = 'seconds since 1970-01-01 00:00:00 UTC'
        lats_var.units = 'degrees_north'
        lons_var.units = 'degrees_east'
        if levels is not None:
            levels_var.units = 'hectopascal'
        data_var.units = units
        data_var.long_name = long_name

        print(f"NetCDF file written successfully for {year}-{month:02d}, variable: {var_name}.")


# Define the output directories for each resampled variable
output_dir_wig = '/Path/to/Interpolated_grids/WIG_filtered_square/Equatorial_Africa'
output_dir_mswep = '/Path/to/Interpolated_grids/MSWEP/Equatorial_Africa'
output_dir_cape = '/Path/to/Interpolated_grids/CAPE/Equatorial_Africa'
output_dir_cin = '/Path/to/Interpolated_grids/CIN/Equatorial_Africa'
output_dir_evap = '/Path/to/Interpolated_grids/Evaporation/Equatorial_Africa'
output_dir_q = '/Path/to/Interpolated_grids/q/Equatorial_Africa'
output_dir_div = '/Path/to/Interpolated_grids/Divergence/Equatorial_Africa'
output_dir_pwat = '/Path/to/Interpolated_grids/PWAT/Equatorial_Africa'
output_dir_slhf = '/Path/to/Interpolated_grids/SLHF/Equatorial_Africa'
output_dir_soil = '/Path/to/Interpolated_grids/Soil_Moisture/Equatorial_Africa'
output_dir_temp = '/Path/to/Interpolated_grids/Temperature/Equatorial_Africa'
output_dir_temp2m = '/Path/to/Interpolated_grids/Temperature_2m/Equatorial_Africa'
output_dir_uwind = '/Path/to/Interpolated_grids/Uwind/Equatorial_Africa'
output_dir_vwind = '/Path/to/Interpolated_grids/Vwind/Equatorial_Africa'
output_dir_vert = '/Path/to/Interpolated_grids/Vertical/Equatorial_Africa'
output_dir_shf = '/Path/to/Interpolated_grids/SHF/Equatorial_Africa'

# Process each year and month
for year in years:
    for month in months:
        print(f"Processing year: {year}, month: {month:02d}")

        # Define paths for all datasets
        # WIG filtered temperature anomaly data
        wig_file = f'/Path/to/Monthly_data/WIG_filtered_square/{year:04d}_{month:02d}_WIG.nc'
        # Precipitation data
        mswep_file = f'/Path/to/Monthly_data/MSWEP/{year:04d}_{month:02d}_3hourly_MSWEP.nc'
        # CAPE data
        cape_file = f'/Path/to/Monthly_data/CAPE/Equatorial_Africa/{year:04d}_{month:02d}_3hourly_CAPE.nc'
        # CIN data
        cin_file = f'/Path/to/Monthly_data/CIN/Equatorial_Africa/{year:04d}_{month:02d}_3hourly_CIN.nc'
        # Evaporation data
        evap_file = f'/Path/to/Monthly_data/Evaporation/Equatorial_Africa/{year:04d}_{month:02d}_3hourly_evaporation.nc'
        # Mixing ratio (q) data
        q_file = f'/Path/to/Monthly_data/q/Equatorial_Africa/{year:04d}_{month:02d}_3hourly_q.nc'
        # Divergence data
        div_file = f'/Path/to/Monthly_data/Divergence/Equatorial_Africa/{year:04d}_{month:02d}_3hourly_divergence.nc'
        # Precipitable water data
        pwat_file = f'/Path/to/Monthly_data/PWAT/Equatorial_Africa/{year:04d}_{month:02d}_3hourly_pwat.nc'
        # Surface Latent Heat Flux data
        slhf_file = f'/Path/to/Monthly_data/SLHF/Equatorial_Africa/{year:04d}_{month:02d}_3hourly_SLHF.nc'
        # Surface Sensible Heat Flux data
        shf_file = f'/Path/to/Monthly_data/SHF/Equatorial_Africa/{year:04d}_{month:02d}_3hourly_SHF.nc'
        # Soil moisture data
        soil_file = f'/Path/to/Monthly_data/Soil_moisture_L1/{year:04d}_{month:02d}_3hourly_SWVL1.nc'
        # Temperature data
        temp_file = f'/Path/to/Monthly_data/Temperature/Equatorial_Africa/{year:04d}_{month:02d}_3hourly_temperature.nc'
        # 2m temp data
        temp2m_file = f'/Path/to/Monthly_data/Temperature/Equatorial_Africa/{year:04d}_{month:02d}_3hourly_2mTemp.nc'
        # U-Component wind data
        uwind_file = f'/Path/to/Monthly_data/Uwind/Equatorial_Africa/{year:04d}_{month:02d}_3hourly_uwind.nc'
        # V-Component wind data
        vwind_file = f'/Path/to/Monthly_data/Vwind/Equatorial_Africa/{year:04d}_{month:02d}_3hourly_vwind.nc'
        # Vertical velocity data
        vert_file = f'/Path/to/Monthly_data/Vertical/Equatorial_Africa/{year:04d}_{month:02d}_3hourly_vertical.nc'

        # Extract lat, lon, and time from WIG filtered data
        ds_wig = xr.open_dataset(wig_file)
        subset_wig = ds_wig.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
        lat_wig, lon_wig = subset_wig.lat, subset_wig.lon
        wig_date = convert_time(ds_wig.time.values)

        # Extract lat, lon, and time from precipitation data
        ds_mswep = xr.open_dataset(mswep_file)
        subset_mswep = ds_mswep.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
        lat_mswep, lon_mswep = subset_mswep.lat, subset_mswep.lon
        mswep_date = convert_time(ds_mswep.time.values)

        # Extract lat, lon, and time from CAPE data
        ds_cape = xr.open_dataset(cape_file)
        subset_cape = ds_cape.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
        lat_cape, lon_cape = subset_cape.lat, subset_cape.lon
        cape_date = convert_time(ds_cape.time.values)

        # Extract lat, lon, and time from CIN data
        ds_cin = xr.open_dataset(cin_file)
        subset_cin = ds_cin.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
        lat_cin, lon_cin = subset_cin.lat, subset_cin.lon
        cin_date = convert_time(ds_cin.time.values)

        # Extract lat, lon, and time from evaporation data
        ds_evap = xr.open_dataset(evap_file)
        subset_evap = ds_evap.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
        lat_evap, lon_evap = subset_evap.lat, subset_evap.lon
        evap_date = convert_time(ds_evap.time.values)

        # Extract lat, lon, and time from soil moisture data
        ds_soil = xr.open_dataset(soil_file)
        subset_soil = ds_soil.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
        lat_soil, lon_soil = subset_soil.lat, subset_soil.lon
        soil_date = convert_time(ds_soil.time.values)

        # Extract lat, lon, and time from surface latent heat flux data
        ds_slhf = xr.open_dataset(slhf_file)
        subset_slhf = ds_slhf.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
        lat_slhf, lon_slhf = subset_slhf.lat, subset_slhf.lon
        slhf_date = convert_time(ds_slhf.time.values)

        # Extract lat, lon, and time from surface sensible heat flux data
        ds_shf = xr.open_dataset(shf_file)
        subset_shf = ds_shf.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
        lat_shf, lon_shf = subset_shf.lat, subset_shf.lon
        shf_date = convert_time(ds_shf.time.values)

        # Extract lat, lon, and time from mixing ratio (q) data
        ds_q = xr.open_dataset(q_file)
        subset_q = ds_q.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
        lat_q, lon_q = subset_q.lat, subset_q.lon
        q_date = convert_time(ds_q.time.values)
        q_levels = subset_q.level.values

        # Extract lat, lon, and time from Divergence data
        ds_div = xr.open_dataset(div_file)
        subset_div = ds_div.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
        lat_div, lon_div = subset_div.lat, subset_div.lon
        div_date = convert_time(ds_div.time.values)
        div_levels = subset_div.level.values

        # Extract lat, lon, and time from PWAT data
        ds_pwat = xr.open_dataset(pwat_file)
        subset_pwat = ds_pwat.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
        lat_pwat, lon_pwat = subset_pwat.lat, subset_pwat.lon
        pwat_date = convert_time(ds_pwat.time.values)

        # Extract lat, lon, and time from temperature data
        ds_temp = xr.open_dataset(temp_file)
        subset_temp = ds_temp.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
        lat_temp, lon_temp = subset_temp.lat, subset_temp.lon
        temp_date = convert_time(ds_temp.time.values)
        temp_levels = subset_temp.level.values

        # Extract lat, lon, and time from 2m temperature data
        ds_temp2m = xr.open_dataset(temp2m_file)
        subset_temp2m = ds_temp2m.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
        lat_temp2m, lon_temp2m = subset_temp2m.lat, subset_temp2m.lon
        temp2m_date = convert_time(ds_temp2m.time.values)

        # Extract lat, lon, and time from U-component wind data
        ds_uwind = xr.open_dataset(uwind_file)
        subset_uwind = ds_uwind.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
        lat_uwind, lon_uwind = subset_uwind.lat, subset_uwind.lon
        uwind_date = convert_time(ds_uwind.time.values)
        uwind_levels = subset_uwind.level.values

        # Extract lat, lon, and time from V-component wind data
        ds_vwind = xr.open_dataset(vwind_file)
        subset_vwind = ds_vwind.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
        lat_vwind, lon_vwind = subset_vwind.lat, subset_vwind.lon
        vwind_date = convert_time(ds_vwind.time.values)
        vwind_levels = subset_vwind.level.values

        # Extract lat, lon, and time from vertical velocity data
        ds_vert = xr.open_dataset(vert_file)
        subset_vert = ds_vert.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
        lat_vert, lon_vert = subset_vert.lat, subset_vert.lon
        vert_date = convert_time(ds_vert.time.values)
        vert_levels = subset_vert.level.values

        # Generate grid definitions for resampling all fields onto WIG grid using their lat lon locations
        lon_mesh_wig, lat_mesh_wig = np.meshgrid(lon_wig, lat_wig)
        lon_mesh_mswep, lat_mesh_mswep = np.meshgrid(lon_mswep, lat_mswep)
        lon_mesh_cape, lat_mesh_cape = np.meshgrid(lon_cape, lat_cape)
        lon_mesh_cin, lat_mesh_cin = np.meshgrid(lon_cin, lat_cin)
        lon_mesh_evap, lat_mesh_evap = np.meshgrid(lon_evap, lat_evap)
        lon_mesh_q, lat_mesh_q = np.meshgrid(lon_q, lat_q)
        lon_mesh_div, lat_mesh_div = np.meshgrid(lon_div, lat_div)
        lon_mesh_pwat, lat_mesh_pwat = np.meshgrid(lon_pwat, lat_pwat)
        lon_mesh_slhf, lat_mesh_slhf = np.meshgrid(lon_slhf, lat_slhf)
        lon_mesh_shf, lat_mesh_shf = np.meshgrid(lon_shf, lat_shf)
        lon_mesh_soil, lat_mesh_soil = np.meshgrid(lon_soil, lat_soil)
        lon_mesh_temp, lat_mesh_temp = np.meshgrid(lon_temp, lat_temp)
        lon_mesh_temp2m, lat_mesh_temp2m = np.meshgrid(lon_temp2m, lat_temp2m)
        lon_mesh_uwind, lat_mesh_uwind = np.meshgrid(lon_uwind, lat_uwind)
        lon_mesh_vwind, lat_mesh_vwind = np.meshgrid(lon_vwind, lat_vwind)
        lon_mesh_vert, lat_mesh_vert = np.meshgrid(lon_vert, lat_vert)

        # Grid definitions used in kd-tree interpolation
        wig_grid = GridDefinition(lons=lon_mesh_wig, lats=lat_mesh_wig)
        mswep_grid = GridDefinition(lons=lon_mesh_mswep, lats=lat_mesh_mswep)
        cape_grid = GridDefinition(lons=lon_mesh_cape, lats=lat_mesh_cape)
        cin_grid = GridDefinition(lons=lon_mesh_cin, lats=lat_mesh_cin)
        evap_grid = GridDefinition(lons=lon_mesh_evap, lats=lat_mesh_evap)
        q_grid = GridDefinition(lons=lon_mesh_q, lats=lat_mesh_q)
        div_grid = GridDefinition(lons=lon_mesh_div, lats=lat_mesh_div)
        pwat_grid = GridDefinition(lons=lon_mesh_pwat, lats=lat_mesh_pwat)
        slhf_grid = GridDefinition(lons=lon_mesh_slhf, lats=lat_mesh_slhf)
        shf_grid = GridDefinition(lons=lon_mesh_shf, lats=lat_mesh_shf)
        soil_grid = GridDefinition(lons=lon_mesh_soil, lats=lat_mesh_soil)
        temp_grid = GridDefinition(lons=lon_mesh_temp, lats=lat_mesh_temp)
        temp2m_grid = GridDefinition(lons=lon_mesh_temp2m, lats=lat_mesh_temp2m)
        uwind_grid = GridDefinition(lons=lon_mesh_uwind, lats=lat_mesh_uwind)
        vwind_grid = GridDefinition(lons=lon_mesh_vwind, lats=lat_mesh_vwind)
        vert_grid = GridDefinition(lons=lon_mesh_vert, lats=lat_mesh_vert)

        # Extract the gridded datasets for resampling
        wig_temp = subset_wig.filtered_temp.values
        mswep_precip = subset_mswep.precipitation.values
        cape = subset_cape.cape.values
        cin = subset_cin.cin.values
        evap = subset_evap.evaporation.values
        q = subset_q.q.values
        div = subset_div.d.values
        pwat = subset_pwat.tcwv.values
        slhf = subset_slhf.slhf.values
        shf = subset_shf.sshf.values
        soil = subset_soil.swvl1.values
        temp = subset_temp.t.values
        temp2m = subset_temp2m.t.values
        uwind = subset_uwind.u.values
        vwind = subset_vwind.v.values
        vert = subset_vert.w.values

        # Resample each gridded dataset onto the wig_temp grid (CLAUS grid) using pyresample.kd_tree module
        # For radius_of_influence calculation -> ROI ≈ 0.75 * target-pixel diagonal
        # WIG res -> 0.33 deg (≈36.7km). Diagonal = sqrt(36.7^2 * 36.7^2) ≈ 52km
        # ROI = 0.75 * 52km = 39000m
        mswep_to_wig = np.zeros_like(wig_temp)
        for i in range(mswep_to_wig.shape[0]):
            mswep_to_wig[i] = pykd.resample_nearest(mswep_grid, mswep_precip[i], wig_grid, radius_of_influence=39000)

        cape_to_wig = np.zeros_like(wig_temp)
        for i in range(cape_to_wig.shape[0]):
            cape_to_wig[i] = pykd.resample_nearest(cape_grid, cape[i], wig_grid, radius_of_influence=39000)

        cin_to_wig = np.zeros_like(wig_temp)
        for i in range(cin_to_wig.shape[0]):
            cin_to_wig[i] = pykd.resample_nearest(cin_grid, cin[i], wig_grid, radius_of_influence=39000)

        evap_to_wig = np.zeros_like(wig_temp)
        for i in range(evap_to_wig.shape[0]):
            evap_to_wig[i] = pykd.resample_nearest(evap_grid, evap[i], wig_grid, radius_of_influence=39000)

        slhf_to_wig = np.zeros_like(wig_temp)
        for i in range(slhf_to_wig.shape[0]):
            slhf_to_wig[i] = pykd.resample_nearest(slhf_grid, slhf[i], wig_grid, radius_of_influence=39000)

        shf_to_wig = np.zeros_like(wig_temp)
        for i in range(shf_to_wig.shape[0]):
            shf_to_wig[i] = pykd.resample_nearest(shf_grid, shf[i], wig_grid, radius_of_influence=39000)

        soil_to_wig = np.zeros_like(wig_temp)
        for i in range(soil_to_wig.shape[0]):
            soil_to_wig[i] = pykd.resample_nearest(soil_grid, soil[i], wig_grid, radius_of_influence=39000)

        pwat_to_wig = np.zeros_like(wig_temp)
        for i in range(pwat_to_wig.shape[0]):
            pwat_to_wig[i] = pykd.resample_nearest(pwat_grid, pwat[i], wig_grid, radius_of_influence=39000)

        temp2m_to_wig = np.zeros_like(wig_temp)
        for i in range(temp2m_to_wig.shape[0]):
            temp2m_to_wig[i] = pykd.resample_nearest(temp2m_grid, temp2m[i], wig_grid, radius_of_influence=39000)

        q_to_wig = np.zeros((q.shape[0], q.shape[1], wig_temp.shape[1], wig_temp.shape[2]))
        for t in range(q.shape[0]):  # Time
            for l in range(q.shape[1]):  # Levels
                q_to_wig[t, l] = pykd.resample_nearest(q_grid, q[t, l], wig_grid, radius_of_influence=39000)

        div_to_wig = np.zeros((div.shape[0], div.shape[1], wig_temp.shape[1], wig_temp.shape[2]))
        for t in range(div.shape[0]):  # Time
            for l in range(div.shape[1]):  # Levels
                div_to_wig[t, l] = pykd.resample_nearest(div_grid, div[t, l], wig_grid, radius_of_influence=39000)

        temp_to_wig = np.zeros((temp.shape[0], temp.shape[1], wig_temp.shape[1], wig_temp.shape[2]))
        for t in range(temp.shape[0]):  # Time
            for l in range(temp.shape[1]):  # Levels
                temp_to_wig[t, l] = pykd.resample_nearest(temp_grid, temp[t, l], wig_grid, radius_of_influence=39000)

        uwind_to_wig = np.zeros((uwind.shape[0], uwind.shape[1], wig_temp.shape[1], wig_temp.shape[2]))
        for t in range(uwind.shape[0]):  # Time
            for l in range(uwind.shape[1]):  # Levels
                uwind_to_wig[t, l] = pykd.resample_nearest(uwind_grid, uwind[t, l], wig_grid, radius_of_influence=39000)

        vwind_to_wig = np.zeros((vwind.shape[0], vwind.shape[1], wig_temp.shape[1], wig_temp.shape[2]))
        for t in range(vwind.shape[0]):  # Time
            for l in range(vwind.shape[1]):  # Levels
                vwind_to_wig[t, l] = pykd.resample_nearest(vwind_grid, vwind[t, l], wig_grid, radius_of_influence=39000)

        vert_to_wig = np.zeros((vert.shape[0], vert.shape[1], wig_temp.shape[1], wig_temp.shape[2]))
        for t in range(vert.shape[0]):  # Time
            for l in range(vert.shape[1]):  # Levels
                vert_to_wig[t, l] = pykd.resample_nearest(vert_grid, vert[t, l], wig_grid, radius_of_influence=39000)

        # Write NetCDF files for each variable
        write_monthly_nc(output_dir_mswep, year, month, mswep_date, lat_wig, lon_wig, mswep_to_wig, 'precipitation', 'mm/hr',
                         'MSWEP Precipitation')
        write_monthly_nc(output_dir_cape, year, month, cape_date, lat_wig, lon_wig, cape_to_wig, 'cape', 'J/kg',
                         'Convective Available Potential Energy')
        write_monthly_nc(output_dir_cin, year, month, cin_date, lat_wig, lon_wig, cin_to_wig, 'cin', 'J/kg',
                         'Convective Inhibition')
        write_monthly_nc(output_dir_evap, year, month, evap_date, lat_wig, lon_wig, evap_to_wig, 'evaporation',
                         'm of water equivalent', 'Evaporation')
        write_monthly_nc(output_dir_pwat, year, month, pwat_date, lat_wig, lon_wig, pwat_to_wig, 'pwat',
                         'kg m**-2', 'Total column vertically-integrated water vapor')
        write_monthly_nc(output_dir_q, year, month, q_date, lat_wig, lon_wig, q_to_wig, 'q', 'kg/kg', 'Mixing Ratio',
                         levels=q_levels)
        write_monthly_nc(output_dir_div, year, month, div_date, lat_wig, lon_wig, div_to_wig, 'd', 's**-1', 'Divergence',
                         levels=div_levels)
        write_monthly_nc(output_dir_slhf, year, month, slhf_date, lat_wig, lon_wig, slhf_to_wig, 'slhf', 'J m**-2',
                         'Surface Latent Heat Flux')
        write_monthly_nc(output_dir_shf, year, month, shf_date, lat_wig, lon_wig, shf_to_wig, 'sshf', 'J m**-2',
                         'Surface Sensible Heat Flux')
        write_monthly_nc(output_dir_soil, year, month, soil_date, lat_wig, lon_wig, soil_to_wig, 'swvl1', 'm**3 m**-3',
                         'Volumetric Soil Moisture')
        write_monthly_nc(output_dir_temp, year, month, temp_date, lat_wig, lon_wig, temp_to_wig, 't', 'K', 'Temperature',
                         levels=temp_levels)
        write_monthly_nc(output_dir_temp2m, year, month, temp2m_date, lat_wig, lon_wig, temp2m_to_wig, 't2m', 'K',
                         '2m Temperature')
        write_monthly_nc(output_dir_uwind, year, month, uwind_date, lat_wig, lon_wig, uwind_to_wig, 'u', 'm/s', 'U-Component of Wind',
                         levels=uwind_levels)
        write_monthly_nc(output_dir_vwind, year, month, vwind_date, lat_wig, lon_wig, vwind_to_wig, 'v', 'm/s', 'V-Component of Wind',
                         levels=vwind_levels)
        write_monthly_nc(output_dir_vert, year, month, vert_date, lat_wig, lon_wig, vert_to_wig, 'w', 'pa/s', 'Vertical velocity',
                         levels=vert_levels)

