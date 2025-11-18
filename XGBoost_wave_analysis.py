import xarray as xr
import numpy as np
import pandas as pd
from skimage import measure
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from datetime import datetime as dt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import shap

'''
This script creates a lagged-forecast dataset around strong WIG waves occurring near the base grid point 0°N, 20°E
(central Congo), removes the diurnal cycle from multiple ERA5 fields, aggregates single and multi level variables over 
specified pressure levels, regions, and specific lag windows,then trains an XGBoost regression model where SHAP tools 
are used to diagnose which atmospheric variables are most strongly associated with the lead up to strong WIG waves.
'''

# --- Feature preparation parameters ---
base_lat, base_lon = 0.0, 20.0

# --- Define the averaging window ---
average_lag_window = range(-7, -4)  # e.g. Index -7, -4 -> lags -21 through -15 hours)
lat_window, lon_window = 5.0, 5.0  # +/-5 degree latitude/longitude windows around base grid point

# --- Define the time-of-day filter (modify as needed) ---
# allowed_hours = [0, 3, 6]  # e.g.  only include times at 00z, 03z, 06z
allowed_hours = [0, 3, 6, 9, 12, 15, 18, 21]  # Include all times

# # --- Define evaporation regions ---
# Option to include other equatorial African regions
evap_regions = {
    # "evap_highlands": {"lat": slice(-5, 5), "lon": slice(30, 40)},
    # "evap_east_congo": {"lat": slice(-5, 5), "lon": slice(26, 30)},
    "evap": {"lat": slice(-5, 5), "lon": slice(15, 25)},
    # "evap_west_congo": {"lat": slice(-5, 5), "lon": slice(5, 14)}
}

# --- Define soil moisture regions ---
soil_regions = {
    # "soil_highlands": {"lat": slice(-5, 5), "lon": slice(30, 40)},
    # "soil_east_congo": {"lat": slice(-5, 5), "lon": slice(26, 30)},
    "soil": {"lat": slice(-5, 5), "lon": slice(15, 25)},
    # "soil_west_congo": {"lat": slice(-5, 5), "lon": slice(5, 14)}
}

# --- Define surface latent heat regions ---
slhf_regions = {
    # "slhf_highlands": {"lat": slice(-5, 5), "lon": slice(30, 40)},
    # "slhf_east_congo": {"lat": slice(-5, 5), "lon": slice(26, 30)},
    "surface_lhf": {"lat": slice(-5, 5), "lon": slice(15, 25)},
    # "slhf_west_congo": {"lat": slice(-5, 5), "lon": slice(5, 14)}
}

# # --- Define sensible heat regions ---
# sshf_regions = {
#     # "sshf_highlands": {"lat": slice(-5, 5), "lon": slice(30, 40)},
#     # "sshf_east_congo": {"lat": slice(-5, 5), "lon": slice(26, 30)},
#     "surface_shf": {"lat": slice(-5, 5), "lon": slice(15, 25)},
#     # "sshf_west_congo": {"lat": slice(-5, 5), "lon": slice(5, 14)}
# }

# --- Load land-sea mask data ---
lsm_file = f'/path/to/land_sea_mask.nc'
ds_lsm = xr.open_dataset(lsm_file)

# --- Threshold for SSHF and SLHF filtering ---
threshold = 1000  # Absolute value threshold

# --- Initialize DataFrame to store features and targets ---
feature_df_raw = []
strong_event_metadata = []


# Diurnal cycle removal function adapted to work on time-series arrays
def remove_diurnal_cycle(array, nharm=2):
    if np.isnan(array).all():
        return array  # Return if all values are NaN

    valid_times = ~np.isnan(array)
    valid_array = array[valid_times]

    if len(valid_array) < (2 * nharm + 1):
        return array  # Not enough data to fit harmonics

    # Scale by number of time steps per day (8 for 3-hourly data)
    timesteps_per_day = 8
    hours = np.arange(len(valid_array)) % timesteps_per_day  # Adjust to 8 unique values per day

    X = np.ones((len(valid_array), 2 * nharm + 1))
    for i in range(1, nharm + 1):
        X[:, 2 * i - 1] = np.sin(2 * np.pi * i * hours / timesteps_per_day)
        X[:, 2 * i] = np.cos(2 * np.pi * i * hours / timesteps_per_day)

    # Perform least squares fit to remove the cycle
    coefficients = np.linalg.lstsq(X, valid_array, rcond=None)[0]
    diurnal_cycle = X @ coefficients

    # Replace valid data points and retain NaNs
    adjusted_array = array.copy()
    adjusted_array[valid_times] = valid_array - diurnal_cycle

    return adjusted_array


def apply_diurnal_cycle_removal(ds, variable_name, nharm=2):
    """
    Apply diurnal cycle removal to an xarray dataset for a given variable,
    while preserving the original dimension order.
    """
    ds_corrected = xr.apply_ufunc(
        remove_diurnal_cycle,
        ds[variable_name],
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,                     # Apply to each grid point independently
        dask="allowed",                     # Allow parallel execution for large datasets
        keep_attrs=True,                    # Preserve variable attributes
        kwargs={"nharm": nharm},            # Explicitly pass nharm as an argument to the remove_diurnal_cycle function
    )

    # Ensure the output has the same dimension order as the input
    return ds_corrected.transpose(*ds[variable_name].dims)


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


# --- Load MJO phase data ---
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
        selected_dates = set(mjoPhase2Dates)
    elif phase == 'suppressed':
        selected_dates = set(mjoPhase56Dates)
    else:
        return np.arange(len(dates))  # No filtering, return all indices
    filtered_idx = []
    for i, dt1 in enumerate(selected_dates):
        for j, dt2 in enumerate(dates):
            if dt1.year == dt2.year and dt1.month == dt2.month and dt1.day == dt2.day:
                filtered_idx.append(j)
    return np.array(filtered_idx)


# --- Phase selection: 'all', 'enhanced', or 'suppressed' ---
# selected_phase = 'enhanced'
# selected_phase = 'suppressed'
selected_phase = 'all'  # Select all phases

# --- Process each year and month ---
for year in range(2001, 2010):
    for month in [2, 3, 4, 5]:
        print(f"Processing year: {year}, month: {month:02d}")

        # Load WIG wave data
        wig_file = f'/path/to/WIG_filtered_OLR/{year:04d}_{month:02d}_WIG.nc'
        ds_wig = xr.open_dataset(wig_file)
        subset_wig = ds_wig.sel(lat=slice(base_lat - lat_window, base_lat + lat_window),
                                lon=slice(base_lon - lon_window, base_lon + lon_window))
        wig_temp = subset_wig.filtered_temp.values
        wig_temp[wig_temp == 9999.0] = np.nan
        wig_temp *= -1  # Reverse the sign of WIG anomalies (more intuitive results; positive SHAP val -> stronger WIG)
        wig_time = pd.to_datetime(subset_wig.time.values)
        lat_wig, lon_wig = subset_wig.lat, subset_wig.lon

        # Extract hours from time
        wig_hours = wig_time.hour

        # Find the indices corresponding to allowed hours
        time_filter_indices = np.where(np.isin(wig_hours, allowed_hours))[0]

        # Convert WIG time array to datetime objects for comparison
        wig_time_dt = convert_time(wig_time)  # Convert to datetime format
        selected_indices = filter_dates_by_phase(wig_time_dt, selected_phase)  # Get indices of relevant MJO phase

        # Load all atmospheric variables (DO NOT FILTER BY HOUR)
        feature_datasets = {}

        var_files_single_lev = {
            "cape": "CAPE",
            "cin": "CIN",
            # "precipitation": "MSWEP",
            "pwat": "PWAT",
            # "t2m": "Temperature_2m",
        }

        var_files_multi_lev = {
            "q": "q",
            "d": "Divergence",
            "u": "Uwind",
            "v": "Vwind",
            "w": "Vertical",
            "t": "Temperature"
        }

        # Variables that need diurnal cycle removal at single levels
        diurnal_vars_single = ["cape", "cin", "precipitation", "t2m"]

        # Multi-level variables that need diurnal cycle removal at specific levels
        diurnal_vars_3_levels = ["d", "u", "v", "w"]  # Apply at 950, 900, 850 hPa
        diurnal_vars_2_levels = ["q"]  # Apply at 950, 850 hPa

        # Define target levels
        target_levels_3 = [950., 900., 850.]  # Levels for Uwind, Vwind, Vertical, Divergence
        target_levels_2 = [950., 850.]  # Levels for q

        # Load single-level variables (apply diurnal cycle removal where needed)
        for var_name, folder in var_files_single_lev.items():
            file_path = f"/path/to/ERA5/{folder}/Equatorial_Africa/{year:04d}_{month:02d}_3hourly_{var_name}.nc"
            ds_var = xr.open_dataset(file_path)
            subset_var = ds_var.sel(lat=slice(base_lat - lat_window, base_lat + lat_window),
                                    lon=slice(base_lon - lon_window, base_lon + lon_window))
            ds_var["time"] = pd.to_datetime(ds_var.time.values)  # Ensure time is in datetime format

            # Apply diurnal cycle removal ONLY to selected variables
            if var_name in diurnal_vars_single:
                subset_var[var_name] = apply_diurnal_cycle_removal(subset_var, var_name)

            feature_datasets[var_name] = subset_var  # Store in dataset dictionary

        use_selected_levels_only = False  # Set to False to apply diurnal removal to ALL levels

        for var_name, folder in var_files_multi_lev.items():
            file_path = f"/path/to/ERA5/{folder}/Equatorial_Africa/{year:04d}_{month:02d}_3hourly_{var_name}.nc"
            ds_var = xr.open_dataset(file_path)
            subset_var = ds_var.sel(lat=slice(base_lat - lat_window, base_lat + lat_window),
                                    lon=slice(base_lon - lon_window, base_lon + lon_window))
            ds_var["time"] = pd.to_datetime(ds_var.time.values)  # Ensure time is in datetime format

            # Determine levels for diurnal correction
            if use_selected_levels_only:
                if var_name in diurnal_vars_3_levels:
                    target_levels = target_levels_3
                elif var_name in diurnal_vars_2_levels:
                    target_levels = target_levels_2
                else:
                    target_levels = []

                level_indices = [np.where(ds_var.level.values == lvl)[0][0] for lvl in target_levels if
                                 lvl in ds_var.level.values]
            else:
                level_indices = list(range(len(ds_var.level)))

            # Apply diurnal cycle removal to the selected levels or all levels
            for idx in level_indices:
                subset_var[var_name].values[:, idx, :, :] = apply_diurnal_cycle_removal(
                    subset_var.isel(level=idx), var_name)

            feature_datasets[var_name] = subset_var

        # Extract level arrays for reference
        uwind_levels = feature_datasets["u"].level.values
        vwind_levels = feature_datasets["v"].level.values
        vert_levels = feature_datasets["w"].level.values
        div_levels = feature_datasets["d"].level.values
        temp_levels = feature_datasets["t"].level.values
        q_levels = feature_datasets["q"].level.values

        # Load Soil, SLHF, and SSHF datasets
        soil_file = f"/path/to/ERA5/variable/Equatorial_Africa/{year:04d}_{month:02d}_3hourly_swvl1.nc"
        slhf_file = f"/path/to/ERA5/variable/Equatorial_Africa/{year:04d}_{month:02d}_3hourly_slhf.nc"
        # sshf_file = f"/Volumes/Seagate/Interpolated_grids/SHF/Equatorial_Africa/{year:04d}_{month:02d}_3hourly_sshf.nc"
        evap_file = f"/path/to/ERA5/variable/Equatorial_Africa/{year:04d}_{month:02d}_3hourly_evaporation.nc"

        ds_soil = xr.open_dataset(soil_file)
        ds_slhf = xr.open_dataset(slhf_file)
        # ds_sshf = xr.open_dataset(sshf_file)
        ds_evap = xr.open_dataset(evap_file)

        ds_soil["swvl1"] = apply_diurnal_cycle_removal(ds_soil, "swvl1", nharm=2)
        ds_slhf["slhf"] = apply_diurnal_cycle_removal(ds_slhf, "slhf", nharm=2)
        # ds_sshf["sshf"] = apply_diurnal_cycle_removal(ds_sshf, "sshf", nharm=2)
        ds_evap["evaporation"] = apply_diurnal_cycle_removal(ds_evap, "evaporation", nharm=2)

        # Apply land-sea mask to soil, SLHF, and SSHF datasets
        expanded_lsm = np.repeat(np.squeeze(ds_lsm.lsm.values)[np.newaxis, :, :], ds_soil.dims['time'], axis=0)
        expanded_lsm[expanded_lsm > 0.9] = 1

        ds_soil["swvl1"] = ds_soil.swvl1.where(expanded_lsm == 1, np.nan)
        ds_slhf["slhf"] = ds_slhf.slhf.where(expanded_lsm == 1, np.nan)
        # ds_sshf["sshf"] = ds_sshf.sshf.where(expanded_lsm == 1, np.nan)
        ds_evap["evaporation"] = ds_evap.evaporation.where(expanded_lsm == 1, np.nan)

        # # Apply filtering of "near zero" values
        # ds_slhf["slhf"] = ds_slhf.slhf.where(np.abs(ds_slhf.slhf) >= threshold, np.nan)
        # ds_sshf["sshf"] = ds_sshf.sshf.where(np.abs(ds_sshf.sshf) >= threshold, np.nan)

        # Find the indices of the closest grid point to the base latitude and longitude
        base_lat_idx = np.argmin(np.abs(lat_wig.values - base_lat))
        base_lon_idx = np.argmin(np.abs(lon_wig.values - base_lon))

        # Detect peak events with strong intensity (>= 7 K)
        wave_index = (wig_temp[:, base_lat_idx, base_lon_idx] >= 7).astype(int)
        labels, n_labels = measure.label(wave_index, return_num=True)

        for label in range(1, n_labels + 1):

            # Extract time indices for the current wave event
            time_indices = np.where(labels == label)[0]
            if len(time_indices) == 0:
                continue

            # Find the time of peak intensity within this wave event
            peak_time_idx = time_indices[np.argmax(wig_temp[time_indices, base_lat_idx, base_lon_idx])]
            peak_time = wig_time[peak_time_idx]
            peak_intensity = wig_temp[peak_time_idx, base_lat_idx, base_lon_idx]

            # Record metadata for strong events
            strong_event_metadata.append({
                "peak_time": peak_time,
                "wig_anomaly": peak_intensity
            })

        # Set this flag to toggle between extracting all WIG values or only strong events
        extract_all_wig = False  # Set to False to only use strong WIG values (>=threshold or <=-threshold)

        # If only strong WIG values should be used, identify those time indices
        thr = 7.0  # Set WIG anomaly threshold
        series = wig_temp[:, base_lat_idx, base_lon_idx]

        if not extract_all_wig:
            # Store only first half of peak and trough values beyond the set threshold
            # Recall sign of WIG anomalies were flipped for SHAP analysis. Positive anomalies = enhanced WIG phases

            # Positive events (ridges -> enhanced WIG phase)
            labels_pos, n_pos = measure.label((series >= thr).astype(int), return_num=True)
            keep_pos = []
            for lab in range(1, n_pos + 1):
                seg = np.where(labels_pos == lab)[0]  # contiguous block where series >= thr
                if seg.size == 0:
                    continue
                peak_idx_in_seg = seg[np.argmax(series[seg])]  # Wave peak
                keep_pos.extend(seg[seg <= peak_idx_in_seg])  # keep first wave half up to peak (inclusive)

            # Negative events (troughs -> suppressed WIG phase)
            labels_neg, n_neg = measure.label((series <= -thr).astype(int), return_num=True)
            keep_neg = []
            for lab in range(1, n_neg + 1):
                seg = np.where(labels_neg == lab)[0]  # contiguous block where series <= -thr
                if seg.size == 0:
                    continue
                trough_idx_in_seg = seg[np.argmin(series[seg])]  # Wave trough
                keep_neg.extend(seg[seg <= trough_idx_in_seg])  # keep first wave half down to trough (inclusive)

            # Merge and sort unique indices
            candidate_indices = np.array(sorted(set(keep_pos + keep_neg)), dtype=int)

            # Use these as the reference times for feature extraction
            strong_indices = candidate_indices
        else:
            strong_indices = np.arange(len(wig_time))  # Use all time steps

        # Iterate only over the lead-up samples (to peak/trough)
        for time_idx in strong_indices:
            intensity = series[time_idx]
            if np.isnan(intensity):
                continue
            current_time = wig_time[time_idx]
            wave_features = {"wave_intensity": intensity, "time": current_time}

            # Store lagged feature values for mean computation
            lagged_values = {feature: [] for feature in [
                "cape", "cin", "pwat", "uwind upper", "uwind mid", "uwind lower", "vwind upper",
                "vwind mid", "vwind lower", "omega upper", "omega mid", "omega lower", "div upper", "div mid",
                "div lower", "temp upper", "temp mid", "temp lower", "q950hPa", "q850hPa", "q500hPa",  # "temp2m", "mswep"
            ]}

            # Initialize storage for regional lagged values
            for region_set in [evap_regions, soil_regions, slhf_regions]:
                for region_name in region_set.keys():
                    lagged_values[region_name] = []

            # Add lagged features for all datasets
            for lag in average_lag_window:
                lag_time = current_time + pd.Timedelta(hours=lag * 3)
                lag_hour = lag_time.hour

                if lag_time not in wig_time:
                    continue  # Ensure lag_time is in the correct MJO phase
                if lag_hour not in allowed_hours:
                    continue  # Skip lags that do not match allowed hours

                # Check if selected_indices is empty before filtering
                if len(selected_indices) == 0:
                    continue  # No valid MJO phase dates, skip this iteration

                # Check if lag_time is in the selected MJO phase dates
                if lag_time not in wig_time_dt[selected_indices]:
                    continue  # Skip this lag_time if it does not belong to the selected MJO phase

                lag_idx = np.where(wig_time == lag_time)[0][0]

                # Add features for each dataset

                # WIG data
                # wave_features[f"wig_lag{lag*3}"] = np.nanmean(wig_temp[lag_idx])

                # Store values for averaging
                lagged_values["cape"].append(np.nanmean(feature_datasets["cape"].cape.values[lag_idx]))
                lagged_values["cin"].append(np.nanmean(feature_datasets["cin"].cin.values[lag_idx]))
                lagged_values["pwat"].append(np.nanmean(feature_datasets["pwat"].pwat.values[lag_idx]))
                # lagged_values["temp2m"].append(np.nanmean(feature_datasets["t2m"].t2m.values[lag_idx]))
                # lagged_values["mswep"].append(feature_datasets["precipitation"].precipitation.values[lag_idx, base_lat_idx, base_lon_idx])

                # Add evaporation features and calculate mean
                for region_name, region_coords in evap_regions.items():
                    region_evap = ds_evap.sel(lat=region_coords["lat"], lon=region_coords["lon"]).evaporation.values
                    lagged_values[region_name].append(np.nanmean(region_evap[lag_idx]))

                # Add soil moisture features and calculate mean
                for region_name, region_coords in soil_regions.items():
                    region_soil = ds_soil.sel(lat=region_coords["lat"], lon=region_coords["lon"]).swvl1.values
                    lagged_values[region_name].append(np.nanmean(region_soil[lag_idx]))

                # Add surface latent heat flux (SLHF) features and calculate mean
                for region_name, region_coords in slhf_regions.items():
                    region_slhf = ds_slhf.sel(lat=region_coords["lat"], lon=region_coords["lon"]).slhf.values
                    lagged_values[region_name].append(np.nanmean(region_slhf[lag_idx]))

                # # Add sensible heat flux (SSHF) features and calculate mean
                # for region_name, region_coords in sshf_regions.items():
                #     region_sshf = ds_sshf.sel(lat=region_coords["lat"], lon=region_coords["lon"]).sshf.values
                #     lagged_values[region_name].append(np.nanmean(region_sshf[lag_idx]))

                # Aggregate Uwind, Divergence, and Temperature features based on atmospheric levels
                # Define levels for each aggregation (hPa)
                upper_uwind_levels = [250., 200., 150.]
                mid_uwind_levels = [650., 550., 450.]
                lower_uwind_levels = [950., 900., 850.]

                upper_vwind_levels = [250., 200., 150.]
                mid_vwind_levels = [650., 550., 450.]
                lower_vwind_levels = [950., 900., 850.]

                upper_omega_levels = [250., 200., 150.]
                mid_omega_levels = [650., 550., 450.]
                lower_omega_levels = [950., 900., 850.]

                upper_div_levels = [250., 200., 150.]
                mid_div_levels = [650., 550., 450.]
                lower_div_levels = [950., 900., 850.]

                upper_temp_levels = [350., 200.]
                mid_temp_levels = [750., 550.]
                lower_temp_levels = [950., 850.]

                # Calculate mean values for Uwind
                lagged_values[f"uwind upper"].append(np.nanmean(
                    [feature_datasets["u"].u.values[lag_idx, np.where(uwind_levels == lvl)[0][0]] for lvl in upper_uwind_levels]))
                lagged_values[f"uwind mid"].append(np.nanmean(
                    [feature_datasets["u"].u.values[lag_idx, np.where(uwind_levels == lvl)[0][0]] for lvl in mid_uwind_levels]))
                lagged_values[f"uwind lower"].append(np.nanmean(
                    [feature_datasets["u"].u.values[lag_idx, np.where(uwind_levels == lvl)[0][0]] for lvl in lower_uwind_levels]))

                # Calculate mean values for Vwind
                lagged_values[f"vwind upper"].append(np.nanmean(
                    [feature_datasets["v"].v.values[lag_idx, np.where(vwind_levels == lvl)[0][0]] for lvl in upper_vwind_levels]))
                lagged_values[f"vwind mid"].append(np.nanmean(
                    [feature_datasets["v"].v.values[lag_idx, np.where(vwind_levels == lvl)[0][0]] for lvl in mid_vwind_levels]))
                lagged_values[f"vwind lower"].append(np.nanmean(
                    [feature_datasets["v"].v.values[lag_idx, np.where(vwind_levels == lvl)[0][0]] for lvl in lower_vwind_levels]))

                # Calculate mean values for vertical velocity
                lagged_values[f"omega upper"].append(np.nanmean(
                    [feature_datasets["w"].w.values[lag_idx, np.where(vert_levels == lvl)[0][0]] for lvl in upper_omega_levels]))
                lagged_values[f"omega mid"].append(np.nanmean(
                    [feature_datasets["w"].w.values[lag_idx, np.where(vert_levels == lvl)[0][0]] for lvl in mid_omega_levels]))
                lagged_values[f"omega lower"].append(np.nanmean(
                    [feature_datasets["w"].w.values[lag_idx, np.where(vert_levels == lvl)[0][0]] for lvl in lower_omega_levels]))

                # Calculate mean values for Divergence
                lagged_values[f"div upper"].append(np.nanmean(
                    [feature_datasets["d"].d.values[lag_idx, np.where(div_levels == lvl)[0][0]] for lvl in upper_div_levels]))
                lagged_values[f"div mid"].append(np.nanmean(
                    [feature_datasets["d"].d.values[lag_idx, np.where(div_levels == lvl)[0][0]] for lvl in mid_div_levels]))
                lagged_values[f"div lower"].append(np.nanmean(
                    [feature_datasets["d"].d.values[lag_idx, np.where(div_levels == lvl)[0][0]] for lvl in lower_div_levels]))

                # Calculate mean values for Temperature
                lagged_values[f"temp upper"].append(np.nanmean(
                    [feature_datasets["t"].t.values[lag_idx, np.where(temp_levels == lvl)[0][0]] for lvl in upper_temp_levels]))
                lagged_values[f"temp mid"].append(np.nanmean(
                    [feature_datasets["t"].t.values[lag_idx, np.where(temp_levels == lvl)[0][0]] for lvl in mid_temp_levels]))
                lagged_values[f"temp lower"].append(np.nanmean(
                    [feature_datasets["t"].t.values[lag_idx, np.where(temp_levels == lvl)[0][0]] for lvl in lower_temp_levels]))

                # Mixing Ratio (q) for all levels
                for level_idx, level in enumerate(q_levels):
                    lagged_values[f"q{int(level)}hPa"].append(np.nanmean(feature_datasets["q"].q.values[lag_idx, level_idx]))

            # Compute and store mean values within the lag window
            for feature, values in lagged_values.items():
                wave_features[f"{feature} {average_lag_window[0]*3}to {average_lag_window[-1]*3}h"] = np.nanmean(values)

            # Append computed features to dataset
            feature_df_raw.append(wave_features)

# --- Convert to DataFrame ---
feature_df = pd.DataFrame(feature_df_raw)

# --- Drop rows with any missing values ---
feature_df.dropna(inplace=True)

# --- Separate features and target ---
X = feature_df.drop(columns=["wave_intensity"])
y = feature_df["wave_intensity"]

# --- Exclude WIG lag and MSWEP lag features ---
# excluded_columns = [col for col in X.columns if col.startswith("wig_lag") or col.startswith("mswep_lag")]
excluded_columns = [col for col in X.columns if col.startswith("wig_lag")]
X = X.drop(columns=excluded_columns)

# --- Drop the 'time' column if not needed ---
X = X.drop(columns=["time"], errors='ignore')

# --- Split data into training and testing sets BEFORE normalization ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Fit the scaler on training data and transform both train and test sets (normalize the datasets)---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Convert scaled data back to DataFrame format for SHAP ---
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# --- Train XGBoost model ---
model = XGBRegressor()
model.fit(X_train_scaled, y_train)

# --- Evaluate the model ---
y_pred = model.predict(X_test_scaled)

# --- Calculate RMSE and R² scores ---
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# --- Print overall model stats ---
print("\n===== Wave Intensity Summary (Overall) =====")
print(f"Minimum wave intensity: {y.min():.4f}")
print(f"Maximum wave intensity: {y.max():.4f}")
print(f"Mean wave intensity: {y.mean():.4f}")
print(f"Standard deviation: {y.std():.4f}")

# --- Compute summary statistics for training and test sets ---
y_train_mean, y_train_std = y_train.mean(), y_train.std()
y_test_mean, y_test_std = y_test.mean(), y_test.std()

# --- Print training and test model stats ---
print("\n===== Wave Intensity Summary (Training Set) =====")
print(f"Training Set Size: {len(y_train)} samples")
print(f"Mean wave intensity: {y_train_mean:.4f}")
print(f"Standard deviation: {y_train_std:.4f}")

print("\n===== Wave Intensity Summary (Test Set) =====")
print(f"Test Set Size: {len(y_test)} samples")
print(f"Mean wave intensity: {y_test_mean:.4f}")
print(f"Standard deviation: {y_test_std:.4f}")

# --- Print model performance stats ---
print("\n===== Model Performance on Test Data =====")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")

# --- SHAP analysis ---
explainer = shap.Explainer(model, X_train_scaled)
shap_values = explainer(X_test_scaled)

# --- Visualize global feature importance ---
shap.summary_plot(shap_values, X_test_scaled)
