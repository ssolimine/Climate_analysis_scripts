import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime as dt
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes

GeoAxes._pcolormesh_patched = Axes.pcolormesh

"""
Script for calculating time-height lagged regressions for the months Feb - May and years 1998 through 2009;
can be changed as desired.
ERA5 atmospheric variables are regressed against quasi two-day WIG wave filtered brightness temperatures (from CLAUS).
rmm and lat-lon indices are for equatorial Africa.
"""

# Define the domain and base grid point
lat_min_height, lat_max_height = -2, 2
lon_min_height, lon_max_height = 18, 22

base_lat, base_lon = 0.0, 20.0


class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        return np.ma.masked_array(super().__call__(value, clip=clip))


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
    Apply diurnal cycle removal to ds[variable_name],
    works with (time, level, lat, lon).
    """
    ds_corrected = xr.apply_ufunc(
        remove_diurnal_cycle,
        ds[variable_name],
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,                       # Apply to each grid point independently
        dask="allowed",                       # Allow parallel execution for large datasets
        keep_attrs=True,                      # Preserve variable attributes
        kwargs={"nharm": nharm},            # Explicitly pass nharm as an argument to the remove_diurnal_cycle function
    )
    # Ensure the output has the same dimension order as the input
    return ds_corrected.transpose(*ds[variable_name].dims)


# Define yearly phase window depending on how many months of data being analyzed per year
def window_phase(time_index, start_month=2, end_month=5):
    # time_index: pandas.DatetimeIndex
    years = time_index.year
    # Start: Feb 1 of that year; End: May 31 of that year
    start = pd.to_datetime([f"{y}-{start_month:02d}-01" for y in years])
    # last day of end_month per year
    end = pd.to_datetime([pd.Timestamp(f"{y}-{end_month:02d}-01")
                          + pd.offsets.MonthEnd(0) for y in years])
    # clamp times to [start, end] just in case
    clamped = np.minimum(np.maximum(time_index.values, start.values), end.values)
    num_days = (end - start).days.values + 1   # 120 or 121 depending on leap year (Feb - May)
    day_in_window = (pd.to_datetime(clamped) - start).days.values.astype(float)
    phase = day_in_window / num_days          # ∈ [0, 1]
    return phase


def build_design_matrix_from_phase(phase, hour, nharm_s=3, nharm_d=2):
    cols = [np.ones_like(phase, dtype=float)]
    for k in range(1, nharm_s + 1):
        ang = 2 * np.pi * k * phase
        cols.append(np.sin(ang)); cols.append(np.cos(ang))
    for m in range(1, nharm_d + 1):
        ang = 2 * np.pi * m * (hour / 24.0)
        cols.append(np.sin(ang)); cols.append(np.cos(ang))
    return np.column_stack(cols)


def remove_seasonal_and_diurnal(series_1d, time_1d, nharm_s=3, nharm_d=2):
    if series_1d.ndim != 1:
        raise ValueError("remove_seasonal_and_diurnal expects a 1D series.")
    mask = ~np.isnan(series_1d)
    if mask.sum() < max(5, 2 * (nharm_s + nharm_d) + 1):
        return series_1d
    t_index = pd.to_datetime(time_1d)
    hour = t_index.hour.values.astype(float)
    phase = window_phase(t_index)           # ∈ [0, 1] within Feb–May per year
    X = build_design_matrix_from_phase(phase, hour, nharm_s=nharm_s, nharm_d=nharm_d)
    y = series_1d.astype(float)
    beta = np.linalg.lstsq(X[mask], y[mask], rcond=None)[0]
    fit = X @ beta
    out = series_1d.copy()
    out[mask] = y[mask] - fit[mask]
    return out


def apply_seasonal_diurnal_removal(ds, variable_name, nharm_s=3, nharm_d=2):
    """
    Apply joint seasonal+diurnal removal to ds[variable_name].
    Works with (time, level, lat, lon).
    This application should not be necessary for high frequency two-day waves analyzed over one
    season (e.g. Feb-May). Apply these functions for yearly or low frequency wave data (slow)
    nharm_s and nharm_d are the number of seasonal and diurnal harmonics included, respectively.
    """
    residual = xr.apply_ufunc(
        remove_seasonal_and_diurnal,
        ds[variable_name],   # DataArray to clean
        ds["time"],          # Pass time axis alongside
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="allowed",
        keep_attrs=True,
        kwargs=dict(nharm_s=nharm_s, nharm_d=nharm_d),
    )
    return residual.transpose(*ds[variable_name].dims)


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
        selected_dates = sorted(set(mjoPhase2Dates))  # Phase 2 enhanced for Congo
    elif phase == 'suppressed':
        selected_dates = sorted(set(mjoPhase56Dates))  # Phases 5 and 6 suppressed for Congo
    else:
        return np.arange(len(dates))
    selected_dates_set = {dt.date() for dt in selected_dates}
    return np.array([i for i, dt in enumerate(dates) if dt.date() in selected_dates_set])


# Storage lists for concatenation
# Empty time-height lists
wig_height_all = []
wig_height_time_all = []

q_height_all = []
q_height_time_all = []

div_height_all = []
div_height_time_all = []

temp_height_all = []
temp_height_time_all = []

omega_height_all = []
omega_height_time_all = []

# Process each year and month
for year in range(1998, 2010):
    for month in [2, 3, 4, 5]:
        print(f"Processing year: {year}, month: {month:02d}")

        wig_file = f'/path/to/wave_filtered_data/{year:04d}_{month:02d}_WIG.nc'

        # ----- Load data for time-height lagged regressions -----

        # Load WIG wave data
        ds_wig_height = xr.open_dataset(wig_file)
        subset_wig_height = ds_wig_height.sel(lat=slice(lat_min_height, lat_max_height),
                                              lon=slice(lon_min_height, lon_max_height))
        wig_height_temp = subset_wig_height.filtered_temp.values
        wig_height_temp[wig_height_temp == 9999.0] = np.nan
        wig_height_time = pd.to_datetime(subset_wig_height.time.values)
        lat_wig_height, lon_wig_height = subset_wig_height.lat, subset_wig_height.lon

        # --- Load Specific humidity (q) data ---

        q_height_file = f'/path/to/era5_data/{year:04d}_{month:02d}_3hourly_q.nc'
        ds_q_height = xr.open_dataset(q_height_file)

        # Apply seasonal + diurnal cycle removal or just diurnal cycle removal
        # ds_q_height["q"] = apply_seasonal_diurnal_removal(ds_q_height, "q", nharm_s=3, nharm_d=2)
        ds_q_height["q"] = apply_diurnal_cycle_removal(ds_q_height, "q")

        subset_q_height = ds_q_height.sel(lat=slice(lat_min_height, lat_max_height),
                                          lon=slice(lon_min_height, lon_max_height))
        q_height = subset_q_height.q.values
        q_height_levels = subset_q_height.level.values
        q_height_time = pd.to_datetime(subset_q_height.time.values)

        # --- Load divergence data ---

        div_height_file = f'/path/to/era5_data/{year:04d}_{month:02d}_3hourly_d.nc'
        ds_div_height = xr.open_dataset(div_height_file)

        # Apply diurnal cycle removal
        # ds_div_height["d"] = apply_seasonal_diurnal_removal(ds_div_height, "d", nharm_s=3, nharm_d=2)
        ds_div_height["d"] = apply_diurnal_cycle_removal(ds_div_height, "d")

        subset_div_height = ds_div_height.sel(lat=slice(lat_min_height, lat_max_height),
                                              lon=slice(lon_min_height, lon_max_height))
        div_height = subset_div_height.d.values
        div_height_levels = subset_div_height.level.values
        div_height_time = pd.to_datetime(subset_div_height.time.values)

        # --- Load temperature data ---

        temp_height_file = f'/path/to/era5_data/{year:04d}_{month:02d}_3hourly_t.nc'
        ds_temp_height = xr.open_dataset(temp_height_file)

        # Apply diurnal cycle removal
        # ds_temp_height["t"] = apply_seasonal_diurnal_removal(ds_temp_height, "t", nharm_s=3, nharm_d=2)
        ds_temp_height["t"] = apply_diurnal_cycle_removal(ds_temp_height, "t")

        subset_temp_height = ds_temp_height.sel(lat=slice(lat_min_height, lat_max_height),
                                                lon=slice(lon_min_height, lon_max_height))
        temp_height = subset_temp_height.t.values
        temp_height_levels = subset_temp_height.level.values
        temp_height_time = pd.to_datetime(subset_temp_height.time.values)

        # --- Load omega data ---

        omega_height_file = f'/path/to/era5_data/{year:04d}_{month:02d}_3hourly_w.nc'
        ds_omega_height = xr.open_dataset(omega_height_file)

        # Apply diurnal cycle removal
        # ds_omega_height["w"] = apply_seasonal_diurnal_removal(ds_omega_height, "w", nharm_s=3, nharm_d=2)
        ds_omega_height["w"] = apply_diurnal_cycle_removal(ds_omega_height, "w")

        subset_omega_height = ds_omega_height.sel(lat=slice(lat_min_height, lat_max_height),
                                                  lon=slice(lon_min_height, lon_max_height))
        omega_height = subset_omega_height.w.values
        omega_height_levels = subset_omega_height.level.values
        omega_height_time = pd.to_datetime(subset_omega_height.time.values)

        # Store for concatenation

        # ---- Store time-height data ----
        wig_height_all.append(wig_height_temp)
        wig_height_time_all.append(wig_height_time)

        q_height_all.append(q_height)
        q_height_time_all.append(q_height_time)

        div_height_all.append(div_height)
        div_height_time_all.append(div_height_time)

        temp_height_all.append(temp_height)
        temp_height_time_all.append(temp_height_time)

        omega_height_all.append(omega_height)
        omega_height_time_all.append(omega_height_time)

# Concatenate across all years to create one long time series
# selected_phase = 'enhanced'
# selected_phase = 'suppressed'
selected_phase = 'all'

# Time-height data
wig_height_final = np.concatenate(wig_height_all, axis=0)
wig_height_time_final = np.concatenate(wig_height_time_all, axis=0)

wig_height_time_dt_final = convert_time(wig_height_time_final)  # Convert to datetime format
selected_indices = filter_dates_by_phase(wig_height_time_dt_final, selected_phase)  # Get indices of relevant MJO phase

q_height_final = np.concatenate(q_height_all, axis=0)
q_height_time_final = np.concatenate(q_height_time_all, axis=0)

div_height_final = np.concatenate(div_height_all, axis=0)
div_height_time_final = np.concatenate(div_height_time_all, axis=0)

temp_height_final = np.concatenate(temp_height_all, axis=0)
temp_height_time_final = np.concatenate(temp_height_time_all, axis=0)

omega_height_final = np.concatenate(omega_height_all, axis=0)
omega_height_time_final = np.concatenate(omega_height_time_all, axis=0)

# Filter data by selected phase

# --- Time-height data ---

wig_height_final = wig_height_final[selected_indices]
wig_height_time_final = wig_height_time_final[selected_indices]

q_height_final = q_height_final[selected_indices]
q_height_time_final = q_height_time_final[selected_indices]

div_height_final = div_height_final[selected_indices]
div_height_time_final = div_height_time_final[selected_indices]

temp_height_final = temp_height_final[selected_indices]
temp_height_time_final = temp_height_time_final[selected_indices]

omega_height_final = omega_height_final[selected_indices]
omega_height_time_final = omega_height_time_final[selected_indices]

# Find the indices of the closest grid point to the base latitude and longitude

base_lat_height_idx = np.argmin(np.abs(lat_wig_height.values - base_lat))
base_lon_height_idx = np.argmin(np.abs(lon_wig_height.values - base_lon))


def time_height_lag_regression(wig, era5, base_y, base_x, max_lag=12):
    # Time/height lag regression
    baseY1 = base_y
    baseY2 = base_y + 1
    baseX = base_x
    maxLag = max_lag

    lagRegHeight = np.zeros((2 * maxLag + 1, era5.shape[1], era5.shape[2], era5.shape[3]))
    wig_temp_clean = np.nan_to_num(wig)  # Convert NaNs to 0 to avoid matrix errors

    for lag in np.arange(-maxLag, maxLag + 1, 1):
        dot_prod = wig_temp_clean[maxLag:era5.shape[0] - maxLag, baseY1:baseY2, baseX] \
            .T.dot(era5[maxLag + lag:era5.shape[0] - maxLag + lag].transpose(1, 2, 0, 3))

        c = np.linalg.inv(wig_temp_clean[maxLag:era5.shape[0] - maxLag, baseY1:baseY2, baseX]
                          .T.dot(wig_temp_clean[maxLag:era5.shape[0] - maxLag, baseY1:baseY2, baseX])) \
            .dot(dot_prod.transpose(1, 2, 0, 3))

        c = c.transpose(1, 2, 0, 3)

        std_array = np.zeros((1, 1))
        # std_array -= np.std(wig_temp_clean[maxLag:era5.shape[0] - maxLag, baseY1:baseY2, baseX])
        std_array -= 15
        lagRegHeight[lag + maxLag] = std_array.dot(c)

    lagRegHeightComplete = np.swapaxes(lagRegHeight[:, :, baseY1, baseX], 0, 1)

    return lagRegHeightComplete


lagReg_q = time_height_lag_regression(wig_height_final, q_height_final, base_lat_height_idx, base_lon_height_idx)

lagReg_div = time_height_lag_regression(wig_height_final, div_height_final, base_lat_height_idx, base_lon_height_idx)

lagReg_temp = time_height_lag_regression(wig_height_final, temp_height_final, base_lat_height_idx, base_lon_height_idx)

lagReg_omega = time_height_lag_regression(wig_height_final, omega_height_final, base_lat_height_idx,
                                          base_lon_height_idx)


def time_height_regression_plot(era5_lag, var_name=None, var_units=None, save_path=None):
    minval = -abs(era5_lag).max()
    maxval = abs(era5_lag).max()

    # Define contour intervals explicitly
    intrvl = (maxval - minval) / 16
    levels = np.arange(minval, maxval + intrvl, intrvl)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Contourf plot with centered normalization
    cb = ax.contourf(
        era5_lag,
        levels=levels,
        cmap='seismic',
        norm=MidpointNormalize(vmin=minval, vmax=maxval, midpoint=0),
        # extend='both'
    )

    ax.contour(era5_lag, levels=np.arange(minval, maxval, intrvl), colors='black', linestyles='solid',
               negative_linestyles='solid', linewidths=0.3)

    # Colorbar setup with explicit labeling
    clb_ticks = np.linspace(minval, maxval, num=9)
    clb = plt.colorbar(cb, ticks=clb_ticks)
    clb.ax.set_yticklabels([f'{tick:.2f}' for tick in clb_ticks])

    # # Explicitly label min/max on colorbar
    clb.ax.set_yticklabels(
        [f'{minval:.2f}'] + [f'{tick:.2f}' for tick in clb_ticks[1:-1]] + [f'{maxval:.2f}']
    )
    clb.set_label(f'( {var_units} )', fontsize=14)

    # Set proper pressure levels (y-axis)
    pressure_levels = np.array([
        1000., 975., 950., 925., 900., 875., 850., 825., 800., 775., 750., 700., 650.,
        600., 550., 500., 450., 400., 350., 300., 250., 225., 200., 175., 150.
    ])

    # Desired ticks every 100 hPa starting from 1000 hPa
    ytick_values = [1000, 900, 800, 700, 600, 500, 400, 300, 200]
    ytick_indices = [np.where(pressure_levels == p)[0][0] for p in ytick_values]
    ax.set_yticks(ytick_indices)
    ax.set_yticklabels([str(int(p)) for p in ytick_values])

    # X-axis ticks for lag
    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_xticklabels([-36, -30, -24, -18, -12, -6, 0, 6, 12, 18, 24, 30, 36])
    ax.set_ylabel('Pressure (hPa)', fontsize=14)
    ax.set_xlabel('Lag (hours)', fontsize=14)
    ax.set_title(f"Time-height lag regressed {var_name} (Congo)", fontsize=14)
    # plt.title('Lag-Height Regression of Temperature', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


time_height_regression_plot(lagReg_q * 1000, var_name="specific humidity", var_units='g / kg',
                            save_path="/savepath/Time-height_lag_q.png")
time_height_regression_plot(lagReg_temp, var_name="temperature", var_units='K',
                            save_path="/savepath/Time-height_lag_temp.png")
time_height_regression_plot(lagReg_div, var_name="divergence",
                            var_units='s\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE}',
                            save_path="/savepath/Time-height_lag_divergence.png")


# --- Separate function for plotting Omega time-height regressions ---


def time_height_regression_plot_omega(era5_lag, var_name=None, var_units=None, save_path=None):
    minval = -abs(era5_lag).max()
    maxval = abs(era5_lag).max()
    intrvl = (maxval - minval) / 16
    levels = np.arange(minval, maxval + intrvl, intrvl)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    # Filled contours
    cb = ax.contourf(
        era5_lag,
        levels=levels,
        cmap='seismic',
        norm=MidpointNormalize(vmin=minval, vmax=maxval, midpoint=0),
    )
    # Line contours
    ax.contour(
        era5_lag,
        levels=levels,
        colors='black',
        linewidths=0.3
    )
    # Colorbar
    clb_ticks = np.linspace(minval, maxval, num=9)
    clb = plt.colorbar(cb, ticks=clb_ticks)
    clb.ax.set_yticklabels(
        [f'{minval:.2f}'] + [f'{tick:.2f}' for tick in clb_ticks[1:-1]] + [f'{maxval:.2f}']
    )
    clb.set_label(f'( {var_units} )', fontsize=14)
    # Pressure levels
    pressure_levels = np.array([1000., 950., 900., 850., 800., 750., 650., 550., 450.,
                                350., 250., 200., 150.])
    ytick_values = [1000, 900, 800, 750, 650, 500, 450, 350, 200]
    # Safely find tick indices that exist in pressure_levels
    ytick_indices = []
    ytick_labels = []
    for p in ytick_values:
        idx = np.where(pressure_levels == p)[0]
        if idx.size > 0:
            ytick_indices.append(idx[0])
            ytick_labels.append(str(int(p)))
    ax.set_yticks(ytick_indices)
    ax.set_yticklabels(ytick_labels)
    # Lag ticks
    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_xticklabels([-36, -30, -24, -18, -12, -6, 0, 6, 12, 18, 24, 30, 36])
    ax.set_ylabel('Pressure (hPa)', fontsize=14)
    ax.set_xlabel('Lag (hours)', fontsize=14)
    ax.set_title(f"Time-height lag regressed {var_name} (Congo)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


time_height_regression_plot_omega((lagReg_omega * 3600) / 100, var_name="omega", var_units='hPa / hr',
                                  save_path="/savepath/Time-height_lag_omega.png")

