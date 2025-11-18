import os
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pyresample.kd_tree as pykd
from skimage import measure
from datetime import datetime as dt
from pyresample.geometry import GridDefinition
from matplotlib.colors import TwoSlopeNorm

'''
This script composites rainfall around quasi–two-day WIG wave events over the Congo (5°S–5°N, 5–40°E) and visualizes the results.
It loads WIG-filtered brightness temperatures (res 0.33°) and MSWEP precipitation (res 0.1°), optionally filters by MJO phase, and
resamples MSWEP onto the WIG grid. Wave peaks are detected per grid cell by labeling contiguous times when WIG-Tb exceeds a
threshold (In this case σ ≈ ±7.92 K) and taking the segment centroids. For each identified peak at a chosen base point (0°N, 20°E), the code
builds lagged (−24…+24 h, 3-hourly) rainfall composites and forms anomalies relative to the full FMAM climatology. 
It then produces: (1) a five-panel map of lagged precipitation anomalies, (2) a base-point lagged precipitation anomaly
time series, (3) a map of the fractional contribution of WIG-associated rainfall with permutation-test significance stippling,
and (4) a histogram of local solar time for peak WIG occurrence (diurnal climatology).
'''

# Define the domain Congo
lat_min = -5
lat_max = 5
lon_min = 5
lon_max = 40

# Define the lag times relative to WIG wave peak in 3-hour increments: -24, through +24 hours
lags = [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]

# Initialize the array to store the composite rainfall data (Africa) - grid size empirically determined
composite_rainfall_total = np.zeros((31, 105))
count_total = np.zeros((31, 105))
count_precip_total = np.zeros((31, 105))  # Count number of time-steps with precipitation
composite_rainfall_wig = np.zeros((len(lags), 31, 105))  # (lag time, lat, lon)
count_wig = np.zeros((len(lags), 31, 105))  # Count the number of valid entries for averaging
count_precip_wig = np.zeros((len(lags), 31, 105))  # Count number of wig time-steps with precipitation

lst = np.zeros(len(lags))  # Record local solar time for each wave
lst_count = np.zeros(len(lags))
lst_list = [[] for i in range(len(lags))]


# Extract MJO phase from the "rmmindex.txt" file and isolate the corresponding dates
def extract_column(file_path, column_number):
    values = []
    with open(file_path, 'r') as file:
        for line in file:
            columns = line.strip().split()
            if len(columns) > column_number:
                values.append(float(columns[column_number]))
    return values


# Function extracts dates associated with MJO phases
def extract_dates(file_path):
    dates = []
    with open(file_path, 'r') as file:
        for line in file:
            columns = line.strip().split()
            if len(columns) >= 3:
                year = int(columns[0])
                month = int(columns[1])
                day = int(columns[2])
                date = dt(year, month, day)
                dates.append(date)
    return dates


# Load MJO phase data
fpath = 'rmmindex.txt'
column_index = 5  # Column associated with MJO phase index
mjoPhase = np.array(extract_column(fpath, column_index))
mjoDates = extract_dates(fpath)
# Identify indices for MJO phases 2 and 5/6
phase2Index = np.where(mjoPhase == 2.0)[0]  # Enhanced phase
phase56Index = np.where(np.logical_or(mjoPhase == 5.0, mjoPhase == 6.0))[0]  # Suppressed phase
# Dates associated with MJO phases
mjoPhase2Dates = [mjoDates[index] for index in phase2Index]
mjoPhase56Dates = [mjoDates[index] for index in phase56Index]


# Function to convert time array to datetime.datetime objects
def convert_time(time_array):
    return np.array([pd.to_datetime(t).to_pydatetime() for t in time_array])


# Function to filter data based on MJO phase
def filter_dates_by_phase(dates, phase=None):
    if phase == 'enhanced':
        selected_dates = sorted(set(mjoPhase2Dates))
    elif phase == 'suppressed':
        selected_dates = sorted(set(mjoPhase56Dates))
    else:
        return np.arange(len(dates))
    selected_dates_set = {dt.date() for dt in selected_dates}
    return np.array([i for i, dt in enumerate(dates) if dt.date() in selected_dates_set])


# Phase selection: 'all', 'enhanced', or 'suppressed'
# selected_phase = 'enhanced'
# selected_phase = 'suppressed'
selected_phase = 'all'

# Loop through each year and month
for year in range(2001, 2010):  # 9 years from 2001 through 2009
    for month in range(2, 6):  # FMAM only

        print(f"Starting data processing for year: {year}, month: {month}")

        # Generate the file paths
        wig_file = f'/path/to/WIG_filtered_cloud_temps/{year:04d}_{month:02d}_WIG.nc'
        # Africa
        mswep_file = f'/path/to/MSWEP_precipitation/{year:04d}_{month:02d}_3hourly_MSWEP.nc'

        # Check if files exist before processing
        if not os.path.exists(wig_file) or not os.path.exists(mswep_file):
            print(f"Skipping {year}-{month:02d} due to missing files.")
            continue

        # Extract lat, lon, and time from WIG filtered data
        ds_wig = xr.open_dataset(wig_file)
        subset_wig = ds_wig.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
        lat_wig, lon_wig = subset_wig.lat, subset_wig.lon
        wig_date = convert_time(ds_wig.time.values)

        # Extract lat, lon, and time from MSWEP data
        ds_mswep = xr.open_dataset(mswep_file)
        subset_mswep = ds_mswep.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
        lat_mswep, lon_mswep = subset_mswep.lat, subset_mswep.lon
        mswep_date = convert_time(ds_mswep.time.values)

        # Filter dates based on the selected MJO phase
        filtered_indices = np.array(sorted(filter_dates_by_phase(mswep_date, selected_phase)))

        # Generate grid definitions for resampling MSWEP precip data onto WIG grid
        lon_mesh_mswep, lat_mesh_mswep = np.meshgrid(lon_mswep, lat_mswep)
        lon_mesh_wig, lat_mesh_wig = np.meshgrid(lon_wig, lat_wig)
        wig_grid = GridDefinition(lons=lon_mesh_wig, lats=lat_mesh_wig)
        mswep_grid = GridDefinition(lons=lon_mesh_mswep, lats=lat_mesh_mswep)

        # Resample MSWEP data to WIG grid (data must be on the same grid for direct comparison)
        # For radius_of_influence calculation -> ROI ≈ 0.75 * target-pixel diagonal
        # WIG res -> 0.33 deg (≈36.7km); MSWEP res -> 0.1 deg (≈11.1km). diagonal = sqrt(36.7^2 * 36.7^2) ≈ 52km
        # ROI = 0.75 * 52km = 39000m
        mswep_precip = subset_mswep.precipitation.values
        mswep_to_wig = np.zeros_like(subset_wig.filtered_temp)
        for i in range(mswep_to_wig.shape[0]):
            mswep_to_wig[i] = pykd.resample_nearest(mswep_grid, mswep_precip[i], wig_grid, radius_of_influence=39000)

        wig_temp = subset_wig.filtered_temp.values
        wig_temp[wig_temp == 9999.0] = np.nan

        # wig_temp_mean = wig_temp.copy()

        # Create wave index array based on WIG-Tb standard deviation
        # Previous calculations find SD of 7.92 kelvin in the Congo
        wave_index = (wig_temp <= -7.92).astype(int)

        # Find peaks in wave_index
        wig_grid = wave_index.copy()
        peak_points = np.zeros_like(wig_temp)

        for i in range(wig_temp.shape[1]):
            for j in range(wig_temp.shape[2]):
                wig_labels, nwig_labels = measure.label(wave_index[:, i, j] > 0,
                                                        return_num=True)  # labels times with wave present

                for k in range(1, nwig_labels):
                    mask = (wig_labels == k).astype(np.uint8)  # Wave mask through time
                    length = np.sum(mask)
                    if length < 0:
                        wig_grid[:, i, j][wig_labels == k] = 0
                        wig_labels[wig_labels == k] = 0

                wig_labels_filtered, nwig_labels_filtered = measure.label(wig_labels > 0,
                                                                          return_num=True)  # Filtered waves in time

                wig_labels_filtered = wig_labels_filtered.reshape(1, peak_points.shape[0])
                props = measure.regionprops(wig_labels_filtered)

                cen_points = []
                for m in range(nwig_labels_filtered):
                    cen_points.append((props[m].centroid[1], props[m].centroid[0]))  # Center point of each wave (peaks)
                cen_points = np.array(cen_points).reshape(len(cen_points), 2)
                cen_points = np.floor(cen_points)

                for p in range(cen_points.shape[0]):
                    index = int(cen_points[p][0])
                    peak_points[index, i, j] = 2  # Identify loc of wave peaks in time for each grid cell (random num)

        # Define the base grid point near 0° latitude, 20°E longitude (Africa)
        base_lat, base_lon = 0.0, 20.0

        # Find the indices of the base grid point in the WIG filtered data
        lat_idx = np.argmin(np.abs(lat_wig.values - base_lat))
        lon_idx = np.argmin(np.abs(lon_wig.values - base_lon))

        # Process filtered dates only
        for t in filtered_indices:
            # Skip time steps with all NaN's
            if np.isnan(np.sum(mswep_to_wig[t])):
                print(f"MSWEP NaN value for {year}_{month:02d} index {t}. Skipping...")
                continue

            composite_rainfall_total += mswep_to_wig[t]  # Store rainfall totals for each grid
            count_total += 1  # Number of observations per grid cell
            count_precip_total += (mswep_to_wig[t] >= 1).astype(int)  # Number of precip. observations per grid cell

            if peak_points[t, lat_idx, lon_idx] == 2:  # Check if there's a WIG wave peak at the base grid point
                for i, lag in enumerate(lags):
                    t_lag = t + lag  # Calculate the corresponding time index for the lag

                    if 0 <= t_lag < peak_points.shape[0]:  # Ensure we stay within the bounds of the dataset
                        # Accumulate rainfall data for all grid points, but relative to the base grid point's lag time
                        if np.isnan(np.sum(mswep_to_wig[t_lag])):
                            print(f"MSWEP NaN value for {year}_{month:02d} lag index {t_lag}. Skipping...")
                            continue

                        composite_rainfall_wig[i] += mswep_to_wig[t_lag]  # Store rainfall associated with WIG waves
                        count_wig[i] += 1  # Number observation times associated with WIG waves per grid cell
                        count_precip_wig[i] += (mswep_to_wig[t_lag] >= 1).astype(int)  # Num of WIG precip obs per grid
                        lst[i] += np.floor(mswep_date[t_lag].hour + (base_lon/15))  # Calculate local solar t for avging
                        lst_count[i] += 1
                        lst_list[i].append(np.round((mswep_date[t_lag].hour + (base_lon / 15)) % 24))  # all local solar t

# Avoid division by zero
count_wig[count_wig == 0] = np.nan

# Calculate the average rainfall for each lag time
composite_rainfall_total_mean = composite_rainfall_total / count_total
composite_rainfall_wig_mean = composite_rainfall_wig / count_wig

# Replace nan values with 0 where no averages could be computed
composite_rainfall_total_mean = np.nan_to_num(composite_rainfall_total_mean)
composite_rainfall_wig_mean = np.nan_to_num(composite_rainfall_wig_mean)

# At this point, composite_rainfall contains the 10-year average composite
# Compute rainfall anomalies with respect to lag times
composite_rainfall_anomaly = np.zeros_like(composite_rainfall_wig_mean)
for i in range(composite_rainfall_wig_mean.shape[0]):
    composite_rainfall_anomaly[i] = composite_rainfall_wig_mean[i] - composite_rainfall_total_mean

# Calculate the latitudinal mean centered on the equator for each time lag
composite_rainfall_anomaly_latmean = composite_rainfall_anomaly.mean(axis=1)

# Calculate percentage of rainfall associated with WIG waves
area_grid = np.ones_like(composite_rainfall_total)
area_grid *= (36.63 * 1000) ** 2  # Calculate area of each grid cell; multiply by cos(lat) near equator = negligible dif

composite_rainfall_wig_total = composite_rainfall_wig[4:9].sum(axis=0)  # Sum rainfall rate for each grid box for lags
                                                                        # -12 thru 0 hours (peak wig rainfall)
                        # Can be any range, but the larger the window, the less statistically significant; the smaller,
                        # the less WIG rainfall captured. The defined 12-hour window is based on previous WIG analysis

total_rain_vol_wig = (composite_rainfall_wig_total / 1000) * area_grid  # Calculate total WIG rainfall in cubic meters
composite_rainfall_total_hours = composite_rainfall_total

total_rain_vol = (composite_rainfall_total_hours / 1000) * area_grid  # Calculate total rainfall in cubic meters
total_rain_wig_percentage = (total_rain_vol_wig / total_rain_vol) * 100  # Percentage of rain associated with WIG waves

# Use these stats for permutation test of statistical significance
count_precip_wig_total = count_precip_wig[4:9].sum(axis=0)  # Number of WIG rain observations from lag -12 thru 0 hour
n_total_days = (count_precip_total / 8)  # Total number of rain days (8 observations per day)
n_wig_rain_days = (count_precip_wig_total / 8)  # Number of rain days associated with WIG waves
observed_rainfall_wig = (
            total_rain_vol_wig / total_rain_vol)  # Observed proportion of rainfall associated with WIG waves (ratio)
expected_rainfall_wig = (count_precip_wig_total / 8) / (
            count_precip_total / 8)  # Expected proportion of rainfall (same as rainy day proportion)

# --- --- --- --- --- --- --- --- --- --- --- Plot results --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

# Five panel plot of composite rainfall rate anomalies associated with WIG waves
# Define parameters
time_range = np.arange(-24, 30, 3)
vmin, vmax = -0.8, 0.8  # Define min and max precip. anomalies
indices_to_plot = [0, 4, 8, 12, -2]  # Indices for time steps in plot

# Create the figure with 5 vertically stacked subplots
fig, axes = plt.subplots(5, 1, figsize=(8, 12), subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)
for idx, ax in zip(indices_to_plot, axes):
    # Extract the specific timestep of composite_rainfall_anomaly
    anomaly_timestep = composite_rainfall_anomaly[idx, :, :]

    # Use TwoSlopeNorm to center the colormap around 0
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    # Plot the data with Normalize scaling
    plot = ax.contourf(lon_wig, lat_wig, anomaly_timestep, levels=np.linspace(vmin, vmax, 11),
                       transform=ccrs.PlateCarree(), cmap='RdBu', norm=norm, extend='both')

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)

    # Plot a star at the base latitude and longitude point
    base_lat = lat_wig[lat_idx]  # Get the latitude of the base point
    base_lon = lon_wig[lon_idx]  # Get the longitude of the base point
    ax.plot(base_lon, base_lat, marker='*', color='black', markersize=12, transform=ccrs.PlateCarree(),
            label='Base Point')

    gl = ax.gridlines(draw_labels=True, alpha=0)
    gl.top_labels = False
    gl.right_labels = False
    if ax != axes[-1]:
        gl.bottom_labels = False
    gl.left_labels = False

    ax.set_yticks([5.0, 2.5, 0, -2.5, -5.0], crs=ccrs.PlateCarree())
    ax.set_yticklabels(['5°N', '2.5°N', '0°', '2.5°S', '5°S'])
    ax.set_extent([lon_wig.min(), lon_wig.max(), lat_wig.min(), lat_wig.max()], crs=ccrs.PlateCarree())
    ax.set_title(f"Composite Rainfall Anomaly - Time lag ({time_range[idx]} hours)", fontsize=15)

cbar = fig.colorbar(plot, ax=axes, orientation='vertical', pad=0.05, extend='both')
cbar.set_label('Precipitation Anomaly (mm/hr)', fontsize=15)
ticks = np.linspace(vmin, vmax, 9)
cbar.set_ticks(ticks)
cbar.set_ticklabels([f'{tick:.1f}' for tick in ticks])  # Format the ticks with one decimal place
plt.show()

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

# Plot lagged composite time series centered over base longitude
# Also calculate fractional area under the curve for the precipitation dataset

precip_ar = composite_rainfall_anomaly[:, lat_idx-5:lat_idx+5, lon_idx-5:lon_idx+5].mean(axis=(1, 2))

tm_lgs = np.arange(-24, 27, 3)  # Time lags
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
# Plot with tm_lgs as the x-axis
ax.plot(tm_lgs, precip_ar, color='tab:blue')
ax.set_xlabel('Lag (hours)')
ax.set_ylabel('Precipitation Anomaly (mm/hr)')
# Set x-axis limits and ticks (6-hour increments from -24 to 24)
ax.set_xlim([-24, 24])
ax.set_xticks(np.arange(-24, 25, 6))
ax.set_ylim([-0.4, 0.8])
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params(axis='both', which='both', direction='in')
ax.tick_params(axis='both', which='major', length=6, width=1)
ax.tick_params(axis='both', which='minor', length=3, width=1)
ax.tick_params(axis='y')
ax.grid(color='lightgrey', axis='both')
ax.set_axisbelow(True)
ax.tick_params(which='minor', length=3)
plt.axvline(x=0, color='k', linestyle='--')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Composite precipitation anomaly at base grid point (Congo)')
plt.show()

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

# Plot estimated percentage of rainfall associated with WIG waves (WIG precip. defined from Lags -12h thru 0h)
# A permutation test is used to determine statistical significance of fractional precipitation estimates

# Number of permutations for the test
num_permutations = 1000


# Function to perform permutation test and calculate p-values
def permutation_test(observed_rainfall_wig, n_wig_rain_days, n_total_days, num_permutations):
    observed_fraction = observed_rainfall_wig  # Observed proportion of rainfall associated with WIG waves
    # Initialize array to store permuted fractions
    # Initialize p-value array with zeros
    p_values_perm = np.zeros_like(observed_rainfall_wig)
    permuted_fractions = np.zeros((num_permutations,) + observed_fraction.shape)
    # Perform permutations
    for p in range(num_permutations):
        # Shuffle WIG-associated rain days randomly across the time dimension for each grid point
        shuffled_labels = np.random.permutation(n_wig_rain_days.flatten())
        shuffled_labels = shuffled_labels.reshape(n_wig_rain_days.shape)
        # Calculate the permuted proportion of WIG-associated rainfall
        permuted_fractions[p] = shuffled_labels / n_total_days
    # Calculate p-values by comparing observed values to the permuted values
    for i in range(observed_fraction.shape[0]):  # Loop over latitudes
        for j in range(observed_fraction.shape[1]):  # Loop over longitudes
            # Count how many permuted values are greater than or equal to the observed
            p_values_perm[i, j] = (np.sum(permuted_fractions[:, i, j] >= observed_fraction[i, j]) + 1) / (num_permutations + 1)
    return p_values_perm


# Example usage:
# observed_rainfall_wig: Proportion of rainfall associated with WIG waves
# n_wig_rain_days: Array of number of WIG wave rain days
# n_total_days: Array of total rain days
# num_permutations: Number of times to permute the data
p_values = permutation_test(observed_rainfall_wig, n_wig_rain_days, n_total_days, num_permutations)

# Plot rainfall associated with WIG waves along with statistical significance
p_threshold = 0.05  # Define the significance threshold
vmin, vmax = 0.0, 50.0
plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
plot = ax.contourf(lon_wig, lat_wig, total_rain_wig_percentage, levels=np.linspace(vmin, vmax, 11),
                   transform=ccrs.PlateCarree(), cmap='YlGnBu', extend='max')

contours = ax.contour(lon_wig, lat_wig, total_rain_wig_percentage, levels=np.linspace(vmin, vmax, 11),
                      colors='black', linewidths=1, transform=ccrs.PlateCarree())

ax.clabel(contours, inline=True, fontsize=8, fmt='%1.1f')

# Plot a star at the base latitude and longitude point
base_lat = lat_wig[lat_idx]  # Get the latitude of the base point
base_lon = lon_wig[lon_idx]  # Get the longitude of the base point
ax.plot(base_lon, base_lat, marker='*', color='black', markersize=12, transform=ccrs.PlateCarree(),
        label='Base Point')

cbar = plt.colorbar(plot, ax=ax, orientation='horizontal', pad=0.05)
cbar.set_label(r'Precipitation %', fontsize=14)
ax.coastlines()
ax.add_feature(cfeature.BORDERS)
ax.set_yticks([5.0, 2.5, 0, -2.5, -5.0], crs=ccrs.PlateCarree())
ax.set_yticklabels(['5°N', '2.5°N', '0°', '2.5°S', '5°S'])
ax.set_xticks(ax.get_xticks(), crs=ccrs.PlateCarree())
ax.set_xticklabels([f'{int(lon)}°E' if lon >= 0 else f'{abs(int(lon))}°W' for lon in ax.get_xticks()])
ax.set_extent([lon_wig.min(), lon_wig.max(), lat_wig.min(), lat_wig.max()], crs=ccrs.PlateCarree())
lon_wig_flat, lat_wig_flat = np.meshgrid(lon_wig, lat_wig)
lon_wig_flat = lon_wig_flat.flatten()
lat_wig_flat = lat_wig_flat.flatten()

# Mask the areas where p_values_perm is not significant
sig_mask = p_values.flatten() < p_threshold
# Plot stippling where p-values are below threshold (significant regions)
ax.scatter(lon_wig_flat[sig_mask], lat_wig_flat[sig_mask], color='crimson', s=10, marker='x', alpha=0.5,
           transform=ccrs.PlateCarree())
# Title
plt.title(f"Fractional rainfall contribution estimates", fontsize=14)
plt.tight_layout()
plt.show()

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

# Generate histogram of WIG wave peak timing in local solar time at base grid location (WIG diurnal cycle climatology)
lst_values = lst_list[8]  # All times of peak wave passage

filtered_values = [value for value in lst_values if 0 <= value <= 23]

fig, ax = plt.subplots(1, 1, figsize=(9, 6))
counts, bins, patches = ax.hist(
    filtered_values,
    bins=np.arange(-0.5, 24.5, 1),
    edgecolor='black',
    color='slategrey',
    density=True
)

# Convert density (fraction) to percent
for patch in patches:
    patch.set_height(patch.get_height() * 100)

# Manually set updated heights
# ax.set_ylim(0, max(p.get_height() for p in patches) * 1.1)
ax.set_ylim(0, 45)
ax.set_xticks(np.arange(0, 24))
ax.set_xlabel('Hour (LST)', fontsize=13)
ax.set_ylabel('Percentage of total (%)', fontsize=13)
plt.title('Diurnal climatology of peak WIG wave occurrence at base grid location (Congo)', fontsize=13)

# Add n = ... and σ box
n = len(filtered_values)
sigma = -7.92
ax.text(
    0.98, 0.95, f"$n = {n}$\n$\sigma = {sigma:.2f}$ K",
    transform=ax.transAxes,
    fontsize=12,
    verticalalignment='top',
    horizontalalignment='right',
    multialignment='left',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='grey', alpha=0.8)
)
plt.show()
