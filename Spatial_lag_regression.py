import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
from scipy import stats
from datetime import datetime as dt
from skimage.transform import radon
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes

GeoAxes._pcolormesh_patched = Axes.pcolormesh

"""
Script for calculating spatial lagged regressions for the months Feb - May and years 1998 through 2009;
can be changed as desired.
ERA5 atmospheric variables are regressed against quasi two-day WIG wave filtered brightness temperatures (from CLAUS).
rmm and lat-lon indices are for equatorial Africa.
"""

# Define the domain and base grid point
lat_min_spatial, lat_max_spatial = -5, 5
lon_min_spatial, lon_max_spatial = 5, 40

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
    if mask.sum() < max(5, 2*(nharm_s+nharm_d)+1):
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
    season (e.g. Feb-May). Apply these functions for yearly or low frequency wave data (slow process)
    nharm_s and nharm_d are the number of seasonal and diurnal harmonics included.
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
# Empty spatial lists
wig_spatial_all = []
wig_spatial_time_all = []

t_spatial_all = []
t_spatial_time_all = []

q_spatial_all = []
q_spatial_time_all = []

div_spatial_all = []
div_spatial_time_all = []

uwind_spatial_all = []
uwind_spatial_time_all = []

vwind_spatial_all = []
vwind_spatial_time_all = []

# Process each year and month
for year in range(1998, 2010):
    for month in [2, 3, 4, 5]:
        print(f"Processing year: {year}, month: {month:02d}")

        wig_file = f'/path/to/wave_filtered_data/{year:04d}_{month:02d}_WIG.nc'

        # ----- Load data for spatial lagged regressions -----

        # Load WIG wave data
        ds_wig_spatial = xr.open_dataset(wig_file)
        subset_wig_spatial = ds_wig_spatial.sel(lat=slice(lat_min_spatial, lat_max_spatial),
                                                lon=slice(lon_min_spatial, lon_max_spatial))
        wig_spatial_temp = subset_wig_spatial.filtered_temp.values
        wig_spatial_temp[wig_spatial_temp == 9999.0] = np.nan
        wig_spatial_time = pd.to_datetime(subset_wig_spatial.time.values)
        lat_wig_spatial, lon_wig_spatial = subset_wig_spatial.lat, subset_wig_spatial.lon

        #  -- Load temp data --

        t_spatial_file = f'/path/to/era5_data/{year:04d}_{month:02d}_3hourly_t.nc'
        ds_t_spatial = xr.open_dataset(t_spatial_file)

        # Apply diurnal cycle removal
        # ds_t_spatial["t"] = apply_seasonal_diurnal_removal(ds_t_spatial, "t", nharm_s=3, nharm_d=2)
        ds_t_spatial["t"] = apply_diurnal_cycle_removal(ds_t_spatial, "t")

        subset_t_spatial = ds_t_spatial.sel(lat=slice(lat_min_spatial, lat_max_spatial),
                                            lon=slice(lon_min_spatial, lon_max_spatial))
        t_spatial = subset_t_spatial.t.values
        t_spatial_levels = subset_t_spatial.level.values
        t_spatial_time = pd.to_datetime(subset_t_spatial.time.values)

        #  -- Load specific humidity data --

        q_spatial_file = f'/path/to/era5_data/{year:04d}_{month:02d}_3hourly_q.nc'
        ds_q_spatial = xr.open_dataset(q_spatial_file)

        # Apply diurnal cycle removal
        # ds_q_spatial["q"] = apply_seasonal_diurnal_removal(ds_q_spatial, "q", nharm_s=3, nharm_d=2)
        ds_q_spatial["q"] = apply_diurnal_cycle_removal(ds_q_spatial, "q")

        subset_q_spatial = ds_q_spatial.sel(lat=slice(lat_min_spatial, lat_max_spatial),
                                            lon=slice(lon_min_spatial, lon_max_spatial))
        q_spatial = subset_q_spatial.q.values
        q_spatial_levels = subset_q_spatial.level.values
        q_spatial_time = pd.to_datetime(subset_q_spatial.time.values)

        #  -- Load divergence data --

        div_spatial_file = f'/path/to/era5_data/{year:04d}_{month:02d}_3hourly_d.nc'
        ds_div_spatial = xr.open_dataset(div_spatial_file)

        # Apply diurnal cycle removal
        # ds_div_spatial["d"] = apply_seasonal_diurnal_removal(ds_div_spatial, "d", nharm_s=3, nharm_d=2)
        ds_div_spatial["d"] = apply_diurnal_cycle_removal(ds_div_spatial, "d")

        subset_div_spatial = ds_div_spatial.sel(lat=slice(lat_min_spatial, lat_max_spatial),
                                                lon=slice(lon_min_spatial, lon_max_spatial))
        div_spatial = subset_div_spatial.d.values
        div_spatial_levels = subset_div_spatial.level.values
        div_spatial_time = pd.to_datetime(subset_div_spatial.time.values)

        #  -- Load Uwind data --

        uwind_spatial_file = f'/path/to/era5_data/{year:04d}_{month:02d}_3hourly_u.nc'
        ds_uwind_spatial = xr.open_dataset(uwind_spatial_file)

        # Apply diurnal cycle removal
        # ds_uwind_spatial["u"] = apply_seasonal_diurnal_removal(ds_uwind_spatial, "u", nharm_s=3, nharm_d=2)
        ds_uwind_spatial["u"] = apply_diurnal_cycle_removal(ds_uwind_spatial, "u")

        subset_uwind_spatial = ds_uwind_spatial.sel(lat=slice(lat_min_spatial, lat_max_spatial),
                                                    lon=slice(lon_min_spatial, lon_max_spatial))
        uwind_spatial = subset_uwind_spatial.u.values
        uwind_spatial_levels = subset_uwind_spatial.level.values
        uwind_spatial_time = pd.to_datetime(subset_uwind_spatial.time.values)

        #  -- Load Vwind data --

        vwind_spatial_file = f'/path/to/era5_data/{year:04d}_{month:02d}_3hourly_v.nc'
        ds_vwind_spatial = xr.open_dataset(vwind_spatial_file)

        # Apply diurnal cycle removal
        # ds_vwind_spatial["v"] = apply_seasonal_diurnal_removal(ds_vwind_spatial, "v", nharm_s=3, nharm_d=2)
        ds_vwind_spatial["v"] = apply_diurnal_cycle_removal(ds_vwind_spatial, "v")

        subset_vwind_spatial = ds_vwind_spatial.sel(lat=slice(lat_min_spatial, lat_max_spatial),
                                                    lon=slice(lon_min_spatial, lon_max_spatial))
        vwind_spatial = subset_vwind_spatial.v.values
        vwind_spatial_levels = subset_vwind_spatial.level.values
        vwind_spatial_time = pd.to_datetime(subset_vwind_spatial.time.values)

        # Store for concatenation

        # ---- Store spatial data ----
        wig_spatial_all.append(wig_spatial_temp)
        wig_spatial_time_all.append(wig_spatial_time)

        t_spatial_all.append(t_spatial)
        t_spatial_time_all.append(t_spatial_time)

        q_spatial_all.append(q_spatial)
        q_spatial_time_all.append(q_spatial_time)

        div_spatial_all.append(div_spatial)
        div_spatial_time_all.append(div_spatial_time)

        uwind_spatial_all.append(uwind_spatial)
        uwind_spatial_time_all.append(uwind_spatial_time)

        vwind_spatial_all.append(vwind_spatial)
        vwind_spatial_time_all.append(vwind_spatial_time)

# Concatenate across all years to create one long time series
# selected_phase = 'enhanced'
# selected_phase = 'suppressed'
selected_phase = 'all'

# Spatial data
wig_spatial_final = np.concatenate(wig_spatial_all, axis=0)
wig_spatial_time_final = np.concatenate(wig_spatial_time_all, axis=0)

wig_spatial_time_dt_final = convert_time(wig_spatial_time_final)  # Convert to datetime format
selected_indices = filter_dates_by_phase(wig_spatial_time_dt_final, selected_phase)  # Get indices of relevant MJO phase

t_spatial_final = np.concatenate(t_spatial_all, axis=0)
t_spatial_time_final = np.concatenate(t_spatial_time_all, axis=0)

q_spatial_final = np.concatenate(q_spatial_all, axis=0)
q_spatial_time_final = np.concatenate(q_spatial_time_all, axis=0)

div_spatial_final = np.concatenate(div_spatial_all, axis=0)
div_spatial_time_final = np.concatenate(div_spatial_time_all, axis=0)

uwind_spatial_final = np.concatenate(uwind_spatial_all, axis=0)
uwind_spatial_time_final = np.concatenate(uwind_spatial_time_all, axis=0)

vwind_spatial_final = np.concatenate(vwind_spatial_all, axis=0)
vwind_spatial_time_final = np.concatenate(vwind_spatial_time_all, axis=0)

# Filter data by selected phase
wig_spatial_final = wig_spatial_final[selected_indices]
wig_spatial_time_final = wig_spatial_time_final[selected_indices]

q_spatial_final = q_spatial_final[selected_indices]
q_spatial_time_final = q_spatial_time_final[selected_indices]

div_spatial_final = div_spatial_final[selected_indices]
div_spatial_time_final = div_spatial_time_final[selected_indices]

t_spatial_final = t_spatial_final[selected_indices]
t_spatial_time_final = t_spatial_time_final[selected_indices]

uwind_spatial_final = uwind_spatial_final[selected_indices]
uwind_spatial_time_final = uwind_spatial_time_final[selected_indices]

vwind_spatial_final = vwind_spatial_final[selected_indices]
vwind_spatial_time_final = vwind_spatial_time_final[selected_indices]

# Find the indices of the closest grid point to the base latitude and longitude

base_lat_spatial_idx = np.argmin(np.abs(lat_wig_spatial.values - base_lat))
base_lon_spatial_idx = np.argmin(np.abs(lon_wig_spatial.values - base_lon))


def spatial_lag_regression(wig, era5, levels, pressure_lev, base_y, base_x, max_lag=12):
    # Creates 2D spatial lag regression
    baseY1 = base_y
    baseY2 = base_y + 1
    baseX = base_x
    maxlag = max_lag

    pressure = levels
    level = np.where(pressure == pressure_lev)[0][0]

    wig_temp_clean = np.nan_to_num(wig)  # Convert NaNs to 0 to avoid matrix errors
    era_reg = era5[:, level, :, :]  # Specify height with second dimension
    lagRegSpatial = np.zeros((2 * maxlag + 1, era_reg.shape[1], era_reg.shape[2]))

    for lag in np.arange(-maxlag, maxlag + 1, 1):
        dot_prod = wig_temp_clean[maxlag:era_reg.shape[0] - maxlag, baseY1:baseY2, baseX] \
            .T.dot(era_reg[maxlag + lag:era_reg.shape[0] - maxlag + lag].transpose(1, 0, 2))
        c = np.linalg.inv(wig_temp_clean[maxlag:era_reg.shape[0] - maxlag, baseY1:baseY2, baseX]
                          .T.dot(wig_temp_clean[maxlag:era_reg.shape[0] - maxlag, baseY1:baseY2, baseX])).dot(
            dot_prod.transpose(1, 0, 2))
        c = c.transpose(1, 0, 2)
        std_array = np.zeros((1, 1))
        # std_array -= np.std(wig_temp_clean[maxlag:era_reg.shape[0] - maxlag, baseY1:baseY2, baseX])
        std_array -= 8
        lagRegSpatial[lag + maxlag] = std_array.dot(c)

    # Creates grid of statistical significance (P-values)
    A_grid = np.zeros_like(wig_temp_clean[maxlag:era_reg.shape[0] - maxlag, :, :])
    for i in range(A_grid.shape[1]):
        for j in range(A_grid.shape[2]):
            A_grid[:, i, j] = wig_temp_clean[maxlag:era_reg.shape[0] - maxlag, baseY1,
                              baseX]  # Fill grid with base grid point values

    pval = np.zeros_like(lagRegSpatial)
    N = 0
    for lag in np.arange(-maxlag, maxlag + 1, 1):
        for j in range(pval.shape[1]):
            for k in range(pval.shape[2]):
                pval[N, j, k] = \
                stats.pearsonr(A_grid[:, j, k], era_reg[maxlag + lag:era_reg.shape[0] - maxlag + lag, j, k])[1]
        N += 1
    pval_sig = pval.copy()
    pval_sig[pval > 0.05] = np.nan

    return lagRegSpatial, pval_sig


lagRegSpatial_temp, pval_temp = spatial_lag_regression(wig_spatial_final, t_spatial_final, t_spatial_levels, 950.,
                                                       base_lat_spatial_idx, base_lon_spatial_idx)

lagRegSpatial_q, pval_q = spatial_lag_regression(wig_spatial_final, q_spatial_final, q_spatial_levels, 950.,
                                                 base_lat_spatial_idx, base_lon_spatial_idx)

lagRegSpatial_q500, pval_q500 = spatial_lag_regression(wig_spatial_final, q_spatial_final, q_spatial_levels, 500.,
                                                       base_lat_spatial_idx, base_lon_spatial_idx)

lagRegSpatial_q850, pval_q850 = spatial_lag_regression(wig_spatial_final, q_spatial_final, q_spatial_levels, 850.,
                                                       base_lat_spatial_idx, base_lon_spatial_idx)

lagRegSpatial_div, pval_div = spatial_lag_regression(wig_spatial_final, div_spatial_final, div_spatial_levels, 200.,
                                                     base_lat_spatial_idx, base_lon_spatial_idx)

lagRegSpatial_div950, pval_div950 = spatial_lag_regression(wig_spatial_final, div_spatial_final, div_spatial_levels,
                                                           950.,
                                                          base_lat_spatial_idx, base_lon_spatial_idx)

lagRegSpatial_uwind200, pval_uwind200 = spatial_lag_regression(wig_spatial_final, uwind_spatial_final,
                                                               uwind_spatial_levels, 200., base_lat_spatial_idx,
                                                               base_lon_spatial_idx)

lagRegSpatial_uwind950, pval_uwind950 = spatial_lag_regression(wig_spatial_final, uwind_spatial_final,
                                                               uwind_spatial_levels, 950., base_lat_spatial_idx,
                                                               base_lon_spatial_idx)

lagRegSpatial_uwind500, pval_uwind500 = spatial_lag_regression(wig_spatial_final, uwind_spatial_final,
                                                               uwind_spatial_levels, 550., base_lat_spatial_idx,
                                                               base_lon_spatial_idx)

lagRegSpatial_uwind850, pval_uwind850 = spatial_lag_regression(wig_spatial_final, uwind_spatial_final,
                                                               uwind_spatial_levels, 850., base_lat_spatial_idx,
                                                               base_lon_spatial_idx)

lagRegSpatial_vwind200, pval_vwind200 = spatial_lag_regression(wig_spatial_final, vwind_spatial_final,
                                                               vwind_spatial_levels, 200., base_lat_spatial_idx,
                                                               base_lon_spatial_idx)

lagRegSpatial_vwind950, pval_vwind950 = spatial_lag_regression(wig_spatial_final, vwind_spatial_final,
                                                               vwind_spatial_levels, 950., base_lat_spatial_idx,
                                                               base_lon_spatial_idx)

lagRegSpatial_vwind500, pval_vwind500 = spatial_lag_regression(wig_spatial_final, vwind_spatial_final,
                                                               vwind_spatial_levels, 550., base_lat_spatial_idx,
                                                               base_lon_spatial_idx)

lagRegSpatial_vwind850, pval_vwind850 = spatial_lag_regression(wig_spatial_final, vwind_spatial_final,
                                                               vwind_spatial_levels, 850., base_lat_spatial_idx,
                                                               base_lon_spatial_idx)

lagRegSpatial_q500_72, pval_q500_72 = spatial_lag_regression(wig_spatial_final, q_spatial_final, q_spatial_levels, 500.,
                                                             base_lat_spatial_idx, base_lon_spatial_idx, max_lag=24)

lagRegSpatial_div_72, pval_div_72 = spatial_lag_regression(wig_spatial_final, div_spatial_final, div_spatial_levels,
                                                           200.,
                                                           base_lat_spatial_idx, base_lon_spatial_idx, max_lag=24)


def spatial_regression_loop(era_regress, lat_max, lat_min, lon_max, lon_min, pval, vmin, vmax,
                            lag_window, var_name=None, var_units=None,
                            uwind=None, vwind=None, pval_u=None, pval_v=None, scale=15):
    lat = np.linspace(lat_min, lat_max, era_regress.shape[1])  # Assuming (time, lat, lon)
    lon = np.linspace(lon_min, lon_max, era_regress.shape[2])
    LON, LAT = np.meshgrid(lon, lat)

    levels = np.linspace(vmin, vmax, 21)
    plt.figure(figsize=(10, 6))
    for i, lag in enumerate(lag_window):
        plt.clf()
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_title(f"Regressed {var_name} anomalies (Lag {lag} hours)", fontsize=14)
        ax.set_extent([lon_min_spatial, lon_max_spatial, lat_min_spatial, lat_max_spatial])
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
        contour = ax.contourf(LON, LAT, era_regress[i], levels=levels, cmap='RdBu_r', norm=norm, extend='both')
        sig = np.where(~np.isnan(pval[i]), 1, np.nan)
        ax.contourf(LON, LAT, sig, levels=[0.5, 1.5], colors='none', hatches=['...'])

        # Wind Vectors (if provided)
        if uwind is not None and vwind is not None and pval_u is not None and pval_v is not None:
            # Sanity check: warn if uwind and vwind look suspiciously similar
            if np.allclose(uwind[i], vwind[i], atol=1e-3, equal_nan=True):
                print(f"Warning: uwind and vwind are nearly identical at lag {lag}h — check your inputs!")
            # Valid where both pvals are significant and neither u nor v is NaN
            sig_mask = np.logical_or((pval_u[i] < 0.05), (pval_v[i] < 0.05))
            sig_mask &= ~np.isnan(uwind[i]) & ~np.isnan(vwind[i])
            u_sig = np.where(sig_mask, uwind[i], np.nan)
            v_sig = np.where(sig_mask, vwind[i], np.nan)
            # Downsample for clarity
            stride = 3
            u_sig_plot = u_sig[::stride, ::stride]
            v_sig_plot = v_sig[::stride, ::stride]
            LON_plot = LON[::stride, ::stride]
            LAT_plot = LAT[::stride, ::stride]
            # print("Quiver vectors to plot:", np.sum(~np.isnan(u_sig_plot) & ~np.isnan(v_sig_plot)))  # For debugging
            ax.quiver(
                LON_plot, LAT_plot, u_sig_plot, v_sig_plot,
                scale=scale, width=0.0025, headwidth=3, transform=ccrs.PlateCarree()
            )
        # Star at base point
        star_lat = lat[base_lat_spatial_idx]
        star_lon = lon[base_lon_spatial_idx]
        ax.plot(star_lon, star_lat, marker='*', color='black', markersize=12, transform=ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle='--', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}
        cbar = plt.colorbar(contour, orientation='horizontal', pad=0.07, shrink=0.8)
        cbar.set_label(f"({var_units})", fontsize=12)
        plt.pause(1)
    plt.show()


lags = np.arange(-36, 39, 3)
# lags = np.arange(-72, 75, 3)

spatial_regression_loop(lagRegSpatial_temp, lat_max_spatial, lat_min_spatial, lon_max_spatial, lon_min_spatial,
                        pval_temp, -0.5, 0.5, lags, var_name="temperature", var_units='K',
                        uwind=lagRegSpatial_uwind950,
                        vwind=lagRegSpatial_vwind950,
                        pval_u=pval_uwind950,
                        pval_v=pval_vwind950,
                        scale=5)

spatial_regression_loop(lagRegSpatial_q * 1000, lat_max_spatial, lat_min_spatial, lon_max_spatial, lon_min_spatial,
                        pval_q, -0.12, 0.12, lags, var_name="specific humidity", var_units="g/kg",
                        uwind=lagRegSpatial_uwind950,
                        vwind=lagRegSpatial_vwind950,
                        pval_u=pval_uwind950,
                        pval_v=pval_vwind950,
                        scale=5)

spatial_regression_loop(lagRegSpatial_div, lat_max_spatial, lat_min_spatial, lon_max_spatial, lon_min_spatial,
                        pval_div, lagRegSpatial_div.min(), lagRegSpatial_div.max(), lags, var_name="divergence",
                        var_units='s\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE}',
                        uwind=lagRegSpatial_uwind200,
                        vwind=lagRegSpatial_vwind200,
                        pval_u=pval_uwind200,
                        pval_v=pval_vwind200,
                        scale=18)


# Hovmöller storage dictionary for Hovmöller diagram of ERA5 fields (q500 or whichever you choose)
q500_hovmoller = {lag: None for lag in lags}

# Compute latitude-averaged velocity potential for each lag
for i, lag in enumerate(lags):
    q500_hovmoller[lag] = np.nanmean(lagRegSpatial_q500[i] * 1000, axis=0)  # Average over latitude

# Stack data into a 2D time-longitude array
hovmoller_matrix = np.stack([q500_hovmoller[lag] for lag in lags], axis=0)

# Create corresponding time labels
time_labels = np.array(lags)  # Lags as time axis
lon_labels = lon_wig_spatial  # Longitude axis

# Define y-axis ticks explicitly at every 12 hours, including endpoints
# y_ticks = np.arange(-72, 73, 24)
y_ticks = np.arange(-36, 39, 12)
vmin, vmax = -0.15, 0.15
norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)

plt.figure(figsize=(12, 6))
# Plot Hovmöller Diagram
plt.contourf(lon_labels, time_labels, hovmoller_matrix, cmap="BrBG", levels=21, norm=norm, extend="both")
cbar = plt.colorbar(label=r"Specific humidity (g/kg)")

# Set y-axis ticks and labels explicitly
plt.yticks(y_ticks, labels=y_ticks)

# Labels and Formatting
plt.xlabel("Longitude (°E)")
plt.ylabel("Lag Time (hours)")
plt.title("Hovmöller Diagram of Specific humidity at 500 hPa")
plt.gca().invert_yaxis()  # Negative lags at the top
plt.savefig("/Figs/Hovmöller_q500.png",
            dpi=300, bbox_inches="tight")
plt.show()


# --- Measure phase speed of wave ---

# Build radon transform and supplemental data arrays
theta = np.linspace(0., 180., 900, endpoint=False)
ones = np.ones_like(hovmoller_matrix)  # For normalization of radon transform
radon_transform = radon(hovmoller_matrix, theta=theta)
radon_ones = radon(ones, theta=theta)  # Radon transform of ones array
radon_norm = radon_transform / radon_ones  # Normalize radon transform
radon_sum = np.sum((radon_norm ** 2), axis=0)  # Calculate sum of square for finding theta index
theta_max = np.where(radon_sum == np.nanmax(radon_sum))  # Locate theta max for phase speed formula

# Divide theta max by 5 to get degree value from index number (theta index res is 1/5th degree e.g. 900/5 = 180)
# Longitude resolution is 0.33 degrees; 1 degree is 111367 meters
# Time resolution is 3 hourly or 1/8th of a day, therefore, time resolution in seconds is 86400s/8 = 10800s
phase_speed_full_dataset = (np.tan(np.deg2rad(theta_max[0][0] / 5)) * 0.33 * 111367) / 10800

print(f"Wave phase speed = {phase_speed_full_dataset} m/s")


# --- Plotting 5-panel view of spatial regressions ---


def regression_5panel_plot(era_regress, lat_min, lat_max, lon_min, lon_max,
                           pval, vmin, vmax, lag_window, indices_to_plot,
                           var_name=None, var_units=None, round=True,
                           uwind=None, vwind=None, pval_u=None, pval_v=None,
                           scale=15, color='BrBG', lev=950, save_path=None):
    # Construct lat/lon meshgrid
    lat = np.linspace(lat_min, lat_max, era_regress.shape[1])
    lon = np.linspace(lon_min, lon_max, era_regress.shape[2])
    LON, LAT = np.meshgrid(lon, lat)

    # Create 5 vertical panels
    fig, axes = plt.subplots(5, 1, figsize=(8, 14), subplot_kw={'projection': ccrs.PlateCarree()},
                             constrained_layout=True)

    for ax, idx in zip(axes, indices_to_plot):
        lag = lag_window[idx]

        norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
        contour = ax.contourf(LON, LAT, era_regress[idx], levels=np.linspace(vmin, vmax, 21),
                              cmap=color, norm=norm, extend='both')

        # Hatching for significance
        sig = np.where(~np.isnan(pval[idx]), 1, np.nan)
        ax.contourf(LON, LAT, sig, levels=[0.5, 1.5], colors='none', hatches=['...'])

        # Wind vectors (either u or v significant)
        if uwind is not None and vwind is not None and pval_u is not None and pval_v is not None:
            if np.allclose(uwind[idx], vwind[idx], atol=1e-3, equal_nan=True):
                print(f"uwind ≈ vwind at lag {lag}h — check inputs!")

            sig_mask = np.logical_or((pval_u[idx] < 0.05), (pval_v[idx] < 0.05))
            sig_mask &= ~np.isnan(uwind[idx]) & ~np.isnan(vwind[idx])
            u_sig = np.where(sig_mask, uwind[idx], np.nan)
            v_sig = np.where(sig_mask, vwind[idx], np.nan)

            stride = 4
            ax.quiver(LON[::stride, ::stride], LAT[::stride, ::stride],
                      u_sig[::stride, ::stride], v_sig[::stride, ::stride],
                      scale=scale, width=0.0025, transform=ccrs.PlateCarree())

        # Star at base point
        star_lat = lat[base_lat_spatial_idx]
        star_lon = lon[base_lon_spatial_idx]
        ax.plot(star_lon, star_lat, marker='*', color='black', markersize=12, transform=ccrs.PlateCarree())

        # Map formatting
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)

        # Gridlines + label logic
        gl = ax.gridlines(draw_labels=True, alpha=0)
        gl.top_labels = False
        gl.right_labels = False
        if ax != axes[-1]:
            gl.bottom_labels = False
        gl.left_labels = False

        ax.set_yticks([5.0, 2.5, 0, -2.5, -5.0], crs=ccrs.PlateCarree())
        ax.set_yticklabels(['5°N', '2.5°N', '0°', '2.5°S', '5°S'])

        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.set_title(f"Regressed Wind and {var_name} {lev} hPa – Time lag ({lag} hours)", fontsize=12)

    # Shared vertical colorbar
    # cbar = fig.colorbar(contour, ax=axes, orientation='vertical', pad=0.03, extend='both')
    # cbar.set_label(f"{var_name} ({var_units})", fontsize=13)
    # ticks = np.linspace(vmin, vmax, 9)
    # cbar.set_ticks(ticks)
    # cbar.set_ticklabels([f'{tick:.2f}' for tick in ticks])
    if round:
        cbar = fig.colorbar(contour, ax=axes, orientation='vertical', pad=0.03, extend='both')
        cbar.set_label(f"{var_name} ({var_units})", fontsize=13)
        ticks = np.linspace(vmin, vmax, 9)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f'{tick:.2f}' for tick in ticks])
    else:
        cbar = fig.colorbar(contour, ax=axes, orientation='vertical', pad=0.03, extend='both')
        cbar.set_label(f"{var_name} ({var_units})", fontsize=13)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


regression_5panel_plot(
    era_regress=lagRegSpatial_q * 1000,
    lat_min=lat_min_spatial,
    lat_max=lat_max_spatial,
    lon_min=lon_min_spatial,
    lon_max=lon_max_spatial,
    pval=pval_q,
    vmin=-0.12,
    vmax=0.12,
    lag_window=lags,
    indices_to_plot=[4, 8, 12, 16, 20],
    var_name="Specific humidity",
    var_units="g/kg",
    round=True,
    uwind=lagRegSpatial_uwind950,
    vwind=lagRegSpatial_vwind950,
    pval_u=pval_uwind950,
    pval_v=pval_vwind950,
    scale=8,  # You can tweak this based on wind magnitudes
    color='BrBG',
    lev=950,
    save_path="/Figs/Reg_5panel_q.png"
)

regression_5panel_plot(
    era_regress=lagRegSpatial_q500 * 1000,
    lat_min=lat_min_spatial,
    lat_max=lat_max_spatial,
    lon_min=lon_min_spatial,
    lon_max=lon_max_spatial,
    pval=pval_q500,
    vmin=-0.20,
    vmax=0.20,
    lag_window=lags,
    indices_to_plot=[4, 8, 12, 16, 20],
    var_name="Specific humidity",
    var_units="g/kg",
    round=True,
    uwind=lagRegSpatial_uwind500,
    vwind=lagRegSpatial_vwind500,
    pval_u=pval_uwind500,
    pval_v=pval_vwind500,
    scale=8,  # You can tweak this based on wind magnitudes
    color='BrBG',
    lev=500,
    save_path="/Figs/Reg_5panel_q500.png"
)

regression_5panel_plot(
    era_regress=lagRegSpatial_temp,
    lat_min=lat_min_spatial,
    lat_max=lat_max_spatial,
    lon_min=lon_min_spatial,
    lon_max=lon_max_spatial,
    pval=pval_temp,
    vmin=-0.5,
    vmax=0.5,
    lag_window=lags,
    indices_to_plot=[4, 8, 12, 16, 20],
    var_name="Temperature",
    var_units="K",
    round=True,
    uwind=lagRegSpatial_uwind950,
    vwind=lagRegSpatial_vwind950,
    pval_u=pval_uwind950,
    pval_v=pval_vwind950,
    scale=8,  # You can tweak this based on wind magnitudes
    color='RdBu_r',
    lev=950,
    save_path="/Figs/Reg_5panel_temp.png"
)

regression_5panel_plot(
    era_regress=lagRegSpatial_div,
    lat_min=lat_min_spatial,
    lat_max=lat_max_spatial,
    lon_min=lon_min_spatial,
    lon_max=lon_max_spatial,
    pval=pval_div,
    vmin=-abs(lagRegSpatial_div).max(),
    vmax=abs(lagRegSpatial_div).max(),
    lag_window=lags,
    indices_to_plot=[4, 8, 12, 16, 20],
    var_name="Divergence",
    var_units='s\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE}',
    round=False,
    uwind=lagRegSpatial_uwind200,
    vwind=lagRegSpatial_vwind200,
    pval_u=pval_uwind200,
    pval_v=pval_vwind200,
    scale=18,
    color='RdBu_r',
    lev=200,
    save_path="/Figs/Reg_5panel_div.png"
)
