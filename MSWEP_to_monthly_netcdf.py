import os
import glob
import numpy as np
import xarray as xr
import pandas as pd
from netCDF4 import Dataset
from datetime import datetime, timedelta
import time
import pytz

"""
This script takes individual global, 3-hourly Multi-Source Weighted-Ensemble Precipitation (MSWEP) files [The latest 
update now provides hourly MSWEP files] and bins them into monthly NetCDF files for the equatorial Africa region 
with a 3-hourly time step per file for the years 1998 through 2009. An array of NaNs is used to fill any missing 
times in the MSWEP dataset. 
"""


# Process_file function to subset region
def process_file(file_path):
    ds = xr.open_dataset(file_path)
    # Subset to the desired region
    ds_subset = ds.sel(lat=slice(15, -15),
                       lon=slice(5, 40))  # Slicing for latitudes from 15S to 15N and longitudes 5E to 40E
    rainfall_rate = ds_subset['precipitation'].values
    lat = ds_subset['lat'].values
    lon = ds_subset['lon'].values
    # Extract and convert time
    time_var = ds_subset['time'].values[0]
    date_time = pd.to_datetime(time_var).to_pydatetime()
    return date_time, rainfall_rate, lat, lon


# Function to get the list of NetCDF files
def get_files(data_dir):
    file_pattern = os.path.join(data_dir, '*.nc')
    return sorted(glob.glob(file_pattern))


# Function to write the final NetCDF file with the composite data
def write_monthly_nc(output_dir, year, month, times, lats, lons, precip):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{year}_{month:02d}_3hourly_MSWEP.nc')
    # Debugging the sizes before writing to NetCDF
    print(f"Writing NetCDF file for {year}-{month:02d}")
    print(f"Time array size: {len(times)}")
    print(f"Precipitation array shape: {precip.shape}")
    with Dataset(output_file, 'w', format='NETCDF4') as ncfile:
        # Dimensions
        time_dim = ncfile.createDimension('time', len(times))
        lat_dim = ncfile.createDimension('lat', lats.shape[0])
        lon_dim = ncfile.createDimension('lon', lons.shape[0])
        # Variables
        times_var = ncfile.createVariable('time', 'f8', ('time',))
        lats_var = ncfile.createVariable('lat', 'f4', ('lat',))
        lons_var = ncfile.createVariable('lon', 'f4', ('lon',))
        precip_var = ncfile.createVariable('precipitation', 'f4', ('time', 'lat', 'lon'), fill_value=9999.0)
        # Write data
        times_var[:] = np.array([t.replace(tzinfo=pytz.UTC).timestamp() for t in times])  # Make sure time is in UTC
        lats_var[:] = lats
        lons_var[:] = lons
        precip_var[:, :, :] = precip
        # Add units attributes
        times_var.units = 'seconds since 1970-01-01 00:00:00 UTC'
        lats_var.units = 'degrees_north'
        lons_var.units = 'degrees_east'
        precip_var.units = 'mm/hr'
        print(f"NetCDF file written successfully with shape {precip.shape}")


# Function to fill missing dates and log if any irregularities
def fill_missing_dates(start_date, end_date, existing_dates):
    current_date = start_date
    all_dates = []
    while current_date <= end_date:
        all_dates.append(current_date)
        current_date += timedelta(days=1)
    missing_dates = [date for date in all_dates if date not in existing_dates]
    return missing_dates, all_dates


# Process yearly data without averaging (MSWEP is already 3-hourly)
def process_yearly_data(base_data_dir, output_dir, years):
    for year in years:
        yearly_data_dir = os.path.join(base_data_dir, str(year))
        file_list = get_files(yearly_data_dir)

        for month in range(1, 13):  # Process each month
            print(f"Starting data processing for year: {year}, month: {month}")
            start_time = time.time()
            monthly_times = []
            monthly_precip = []
            lats, lons = None, None

            # Process each file
            for file_path in file_list:
                date_time, rainfall_rate, lat, lon = process_file(file_path)
                if date_time.year == year and date_time.month == month:
                    monthly_times.append(date_time)
                    if lats is None or lons is None:
                        lats, lons = lat, lon

                    # Ensure rainfall_rate has correct shape
                    rainfall_rate = np.squeeze(rainfall_rate)
                    monthly_precip.append(rainfall_rate)

            # Step 1: Fill missing dates
            start_date = datetime(year, month, 1)
            next_month = (start_date.replace(day=28) + timedelta(days=4)).replace(day=1)
            end_date = next_month - timedelta(minutes=30)
            missing_dates, all_dates = fill_missing_dates(start_date, end_date, monthly_times)

            # Insert missing data (NaNs) for the missing dates
            for missing_date in missing_dates:
                monthly_times.append(missing_date)
                monthly_precip.append(np.full((len(lats), len(lons)), np.nan))  # Insert NaNs for missing data

            # Sort by time to align data
            sorted_indices = np.argsort(monthly_times)
            monthly_times = np.array(monthly_times)[sorted_indices]
            monthly_precip = np.array(monthly_precip)[sorted_indices]

            # Check the shape of monthly precipitation array
            print(f"Shape of monthly precipitation array: {monthly_precip.shape}")

            # Write to NetCDF
            write_monthly_nc(output_dir, year, month, monthly_times, lats, lons, monthly_precip)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Year {year} Month {month} completed in {elapsed_time:.2f} seconds")


# Process functions to create monthly netcdf of 3-hourly MSWEP precipitation
base_data_dir = '/Volumes/Seagate/MSWEP'
output_dir = '/Users/User/MSWEP_monthly_data/'
years = range(1998, 2010)  # Process 1998 through 2009
process_yearly_data(base_data_dir, output_dir, years)
