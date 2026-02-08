from __future__ import annotations

from learning_driver_preferences.plot_style import set_plot_style, CUSTOM_COLORS
set_plot_style()

from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

from ipywidgets import interact, Dropdown, ToggleButtons

# ============================================================
# HELPERS
# ============================================================
def read_inputfile(
    inputfile: str | Path,
    depot: Optional[str] = None,
):
    # Read CSV or Excel, normalize headers, and (optionally) filter by depot.
    # Standardized columns (renamed when present):
    #   - route_id
    #   - request_time
    #   - depot (derived from route_id if available)
    #   - date
    #   - config_name
    #   - num_tasks
    #   - num_fixed

    # Excel-specific renames (case-insensitive):
    #   NumberOfTasks            -> num_tasks
    #   Date                     -> date
    #   TriggerType              -> config_name
    #   NumberOfTasksInInputPlan -> num_fixed
    #   RouteId                  -> route_id
    #   Time                     -> request_time

    # Returns:
    #   df_norm: DataFrame with normalized names and optional depot filtering
    #   info:    basic info dict with metadata (selected_depot, source, filename)

    path = Path(inputfile)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    is_csv = suffix == ".csv"
    is_excel = suffix in {".xlsx", ".xlsm", ".xls"}

    if not (is_csv or is_excel):
        raise ValueError("Unsupported file type. Provide .csv or .xlsx/.xlsm/.xls.")

    if is_csv:
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, engine="openpyxl")

    # Normalize header whitespace
    df = df.rename(columns={c: c.strip() if isinstance(c, str) else c for c in df.columns})

    # Case-insensitive mapping helper
    lower_map = {str(c).lower(): c for c in df.columns}

    # Shared rename map
    rename_pairs = {}

    # Route/time
    if "routeid" in lower_map:
        rename_pairs[lower_map["routeid"]] = "route_id"
    if "time" in lower_map:
        rename_pairs[lower_map["time"]] = "request_time"

    # Excel column renames (apply when present, CSV included too if headers match)
    if "numberoftasks" in lower_map:
        rename_pairs[lower_map["numberoftasks"]] = "num_tasks"
    if "date" in lower_map:
        rename_pairs[lower_map["date"]] = "date"
    if "configurationname" in lower_map:
        rename_pairs[lower_map["configurationname"]] = "config_name"
    if "numberoftasksininputplan" in lower_map:
        rename_pairs[lower_map["numberoftasksininputplan"]] = "num_fixed"

    # If CSV already has canonical names (e.g., num_tasks), this is a NOOP
    df = df.rename(columns=rename_pairs)

    # Derive depot (first 4 chars of route_id) if route_id exists
    if "route_id" in df.columns:
        df["depot"] = df["route_id"].astype(str).str.strip().str[:4]
    else:
        # If no route_id, keep depot as provided (filtering will be skipped)
        if "depot" not in df.columns:
            df["depot"] = np.nan

    # Optional depot filter (only if depot info is available)
    if depot is not None and "depot" in df.columns:
        df = df.loc[df["depot"].astype(str) == str(depot)].copy()
        selected_depot = str(depot)
    else:
        # Indicate selection in metadata
        selected_depot = "ALL" if df["depot"].notna().any() else "N/A"

    # Normalize 'date' if present: try to keep an ISO string (YYYY-MM-DD)
    if "date" in df.columns:
        # Accept 'YYYYMMDD', datetime, Excel serials, etc.
        d_raw = df["date"]
        if pd.api.types.is_integer_dtype(d_raw) or pd.api.types.is_float_dtype(d_raw):
            # Could be Excel serial or YYYYMMDD as number
            # Try Excel serial: heuristic — serials are typically < 60000 for 21st century
            s = pd.to_datetime(d_raw, errors="coerce", origin="1899-12-30", unit="D")
            # If that failed for many, try YYYYMMDD
            if s.isna().mean() > 0.5:
                s = pd.to_datetime(d_raw.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
        else:
            # Strings like '20240521' or already ISO
            s = pd.to_datetime(d_raw.astype(str), errors="coerce")
            # Secondary try for compact YYYYMMDD
            mask_bad = s.isna()
            if mask_bad.any():
                s2 = pd.to_datetime(d_raw[mask_bad].astype(str), format="%Y%m%d", errors="coerce")
                s = s.mask(mask_bad, s2)
        df["date"] = s.dt.date.astype(str)  # ISO-like "YYYY-MM-DD", 'NaT'->'NaT' string if NaN

    # Normalize "time" if present
    if "request_time" in df.columns:
        def normalize_time(x):
            if pd.isna(x):
                return None
            x = str(x).strip()
            # Try many formats (loose matching)
            fmts = [
                "%I:%M:%S.%f %p",  # 01:02:03.456 PM
                "%I:%M:%S %p",     # 01:02:03 PM
                "%H:%M:%S.%f",     # 13:02:03.456
                "%H:%M:%S",        # 13:02:03
                "%H:%M",           # 13:02
            ]
            for fmt in fmts:
                try:
                    t = datetime.datetime.strptime(x, fmt).time()
                    return t.strftime("%H:%M:%S")
                except:
                    pass

            # Sometimes Excel reads dates as timestamps:
            try:
                ts = pd.to_datetime(x, errors="coerce")
                if not pd.isna(ts):
                    return ts.strftime("%H:%M:%S")
            except:
                pass

            return None

        df["request_time"] = df["request_time"].apply(normalize_time)

    # build summary
    info = {
        "selected_depot": selected_depot,
        "source": "csv" if is_csv else "excel",
        "filename": path.name,
    }

    print(f"file information: {info}")

    return df, info

def parse_request_times(df: pd.DataFrame):
    # parses df['request_time'] if present. Returns a Series[datetime64[ns]]
    if "request_time" not in df.columns:
        return pd.Series([], dtype="datetime64[ns]")

    time_str = df["request_time"].astype(str).str.strip()
    parsed_time = pd.to_datetime(time_str, format="%H:%M:%S", errors="coerce")
    if parsed_time.isna().all():
        parsed_time = pd.to_datetime(time_str, format="%I:%M:%S.%f %p", errors="coerce")
    if parsed_time.isna().all():
        parsed_time = pd.to_datetime(time_str, format="%I:%M:%S %p", errors="coerce")
    if parsed_time.isna().all():
        parsed_time = pd.to_datetime(time_str, errors="coerce")
    return parsed_time


# ===============================================================================================================
# ANALYSIS HELPERS: MINUTE_OF_DAY, FILTER REQUESTS ON CUTOFF TIME AND TASKS, CREATE A TABLE WITH 1 ROW PER TRIP
# ===============================================================================================================
def ensure_minute_of_day(df: pd.DataFrame):
    """Return a Series with minutes after midnight for each row if possible.
    Tries, in order:
      - 'minute_of_day' column if present and numeric
      - parse_request_times and compute hour*60 + minute
    """
    if "minute_of_day" in df.columns:
        s = pd.to_numeric(df["minute_of_day"], errors="coerce")
        if s.notna().any():
            return s

    t = parse_request_times(df)
    if t.empty or t.isna().all():
        return pd.Series(np.nan, index=df.index)
    return (t.dt.hour * 60 + t.dt.minute).astype(int)

def filter_on_cutoff_time_and_tasks(
 # the combination of route_id and day/date = "a trip" (NL = "rit") = the execution of that route/route_id on a particular day
    df: pd.DataFrame,
    cutoff_time: datetime.time = datetime.time(11, 0, 0),  # 11:00
    cutoff_mean_tasks: float = 30.0    # keep only trips (route/day) with mean tasks > 30 (mean tasks = mean of num_tasks across the requests for that trip)
) -> pd.DataFrame:
    """Apply first-round filter (if column exists) and time cutoff; keep mornings with mean tasks > threshold."""
    # Standardize key columns (they may or may not be present depending on source)
    if "date" not in df.columns:
        raise ValueError("Input must include a 'date' column")
    if "request_time" not in df.columns:
        raise ValueError("Input must include a 'request_time' column")
    if "route_id" not in df.columns:
        raise ValueError("Input must include a 'route_id' column")
    if "config_name" not in df.columns:
        raise ValueError("Input must include a 'config_name' colum")
    if "num_tasks" not in df.columns:
        raise ValueError("Input must include a 'num_tasks' column")

    filter_df = df.copy()
    # # Filter on cutoff time (default 11h) using minute-of-day
    # minutes = ensure_minute_of_day(df_filter)
    # df_filter["_minute_of_day"] = minutes  # of: df_f = df_f.assign(_minutes=minutes)
    times = pd.to_datetime(filter_df["request_time"], format="%H:%M:%S", errors="coerce").dt.time

    mask = times <= cutoff_time
    filter_df = filter_df.loc[mask].copy()

    # Set num_tasks to numeric - defensive code to be sure num_tasks is numeric
    filter_df["num_tasks"] = pd.to_numeric(filter_df["num_tasks"], errors="coerce")

    # Keep only trips (route_id, date) with mean tasks > threshold
    key = ["route_id", "date"]
    trip_mean_tasks = (
        filter_df.groupby(key)["num_tasks"].mean().reset_index(name="mean_num_tasks_of_trip")
    )
    filter_tasks_df = trip_mean_tasks.loc[trip_mean_tasks["mean_num_tasks_of_trip"] > cutoff_mean_tasks, key] # only keep route_id and day, throw away the mean_num_tasks
    filtered_time_and_tasks_df = filter_df.merge(filter_tasks_df, on=key, how="inner")

    # Provide weekday
    # df_keep = df_keep.drop(columns=["_minute_of_day"], errors="ignore")
    filtered_time_and_tasks_df["date"] = pd.to_datetime(filtered_time_and_tasks_df["date"], errors="coerce")
    filtered_time_and_tasks_df["weekday"] = filtered_time_and_tasks_df["date"].dt.day_name()

    return filtered_time_and_tasks_df

# Annotate requests with sequence number: "run number" (first run, second run, third run = sequence number of requests to software) and "run label" (first run, intermediate run and last run) to be able to compare f.ex first and last request to software (run) of a trip
def annotate_requests_with_sequence_num(df: pd.DataFrame):
    # Ensure required columns exist
    required = {"route_id", "date", "request_time"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

   # Parse request_time and compute minute_of_day - to order the request_times in an easy way, minute_of_day gives an integer from 0 - 1439 (24x60min in a day), and you keep order across the day
    t = pd.to_datetime(df["request_time"], format="%H:%M:%S", errors="coerce")
    # df_files["minute_of_day"] = t.dt.hour*60 + t.dt.minute

    df["request_time"] = pd.to_datetime(
        df["request_time"],
        format="%H:%M:%S",
        errors="coerce"
    ).dt.time


    sorted_df= df.sort_values(
        by=["route_id", "date", "request_time"],
        ascending=[True, True, True]
    )

    sorted_df["run_number"] = (
        sorted_df.groupby(["route_id","date"]).cumcount() + 1)

    # Determine group sizes to identify 'last_run'
    group_sizes = (
        sorted_df.groupby(["route_id","date"])["request_time"].transform("size")
    )
    sorted_df["group_size"] = group_sizes

    # Assign run_label based on run_number and group_size
    sorted_df["run_label"] = "not_relevant"

    condition_only  = (sorted_df["run_number"].astype("Int64") == 1) & (sorted_df["group_size"] == 1)
    condition_first = (sorted_df["run_number"].astype("Int64") == 1) & (sorted_df["group_size"] > 1)
    condition_last  = (sorted_df["run_number"].astype("Int64") == sorted_df["group_size"]) & (sorted_df["group_size"] > 1)
    condition_mid   = ~(condition_only | condition_first | condition_last)

    sorted_df.loc[condition_only,  "run_label"] = "only_run"
    sorted_df.loc[condition_first, "run_label"] = "first_run"
    sorted_df.loc[condition_last,  "run_label"] = "last_run"
    sorted_df.loc[condition_mid,   "run_label"] = "intermediate_run"

    annotated_with_request_seq = sorted_df.copy()

    return annotated_with_request_seq

def aggregate_trips(filtered_time_and_tasks_df: pd.DataFrame) -> pd.DataFrame:

# Aggregate filtered rows from df_keep (= 1 row per kept request) to a table with trips ( = 1 row per trip = 1 row per route/day; requests are summed acroos the trip)
#     Returns columns:
#       - route_id, date, weekday
#       - requests_count
#       - day_mean_tasks
#       - estimate_pct, create_pct, add_pct (share of request_type within trip)

    df = filtered_time_and_tasks_df.copy()

    key = ["route_id", "date"] # trip

    # Request counts per trip
    request_counts = df.groupby(key).size().reset_index(name="requests_count")

    # Mean of num_tasks per trip
    trip_mean_tasks = (
        df.groupby(key)["num_tasks"].mean().reset_index(name="mean_tasks_trip")
    )

    # Config_name shares per trip
    # Computes the number of config_name (= type of request = EstimateTime, CreateSequence, AddToSequence) per trip (route, day)
    type_of_requests_count = (
        df.groupby(key + ["config_name"]).size()
        .unstack("config_name", fill_value=0) # unstack = set config_names to separate columns instead of 1 column config_name with types of requests as values in rows
    )

    type_of_requests_count = type_of_requests_count.rename(
        columns={
            "EstimateTime": "estimate_count",
            "CreateSequence": "create_count",
            "AddToSequence": "add_count"
        }
    )

    # Ensure consistent columns
    for config in ["estimate_count", "create_count", "add_count"]:
        if config not in type_of_requests_count.columns:
            type_of_requests_count[config] = 0

    type_of_requests_count = type_of_requests_count[["estimate_count", "create_count", "add_count"]]
    type_of_requests_count["total_count"] = type_of_requests_count.sum(axis=1).replace(0, np.nan)
    type_of_requests_portion= type_of_requests_count.div(type_of_requests_count["total_count"], axis=0) * 100.0
    type_of_requests_portion = type_of_requests_portion.rename(
        columns={
            "estimate_count": "estimate_pct",
             "create_count": "create_pct",
            "add_count": "add_pct",
            "total_count": "total_pct"
        }
    ).reset_index()

    # Merge
    aggretated_trips_df = (
        request_counts.merge(trip_mean_tasks, on=key, how="left")
        .merge(type_of_requests_count, on=key, how = "left")
        .merge(type_of_requests_portion, on=key, how="left")
        .merge(df[key + ["weekday"]].drop_duplicates(), on=key, how="left")
    )

    return aggretated_trips_df

# ==============================================================================
# (2) BASIC INFORMATION: NUMBER OF ROUTES, TRIPS AND REQUESTS TAKEN IN ACCOUNT
# ==============================================================================
def basic_information(df: pd.DataFrame, cutoff_mean_tasks: int = 30):
    df_keep_all_tasks = filter_on_cutoff_time_and_tasks(df, cutoff_mean_tasks=0)
    df_keep_with_min_tasks = filter_on_cutoff_time_and_tasks(df, cutoff_mean_tasks=cutoff_mean_tasks)

    number_of_routes_before_filter = df_keep_all_tasks["route_id"].nunique()
    number_of_routes_after_filter  = df_keep_with_min_tasks["route_id"].nunique()

    number_of_days_observed = df_keep_all_tasks["date"].nunique()

    number_of_trips_before_filter = (
        df_keep_all_tasks[["route_id", "date"]].drop_duplicates().shape[0]
    )
    number_of_trips_after_filter = (
        df_keep_with_min_tasks[["route_id", "date"]].drop_duplicates().shape[0]
    )

    number_requests_no_filter = df.drop_duplicates(subset=["route_id","date","request_time"]).shape[0]
    number_requests_after_time = df_keep_all_tasks.drop_duplicates(subset=["route_id","date","request_time"]).shape[0]
    number_requests_after_tasks = df_keep_with_min_tasks.drop_duplicates(subset=["route_id","date","request_time"]).shape[0]


    return pd.DataFrame([{
        "routes_all_tasks": number_of_routes_before_filter,
        "routes_with_task_filter": number_of_routes_after_filter,
        "days_observed": number_of_days_observed,
        "trips_all_tasks": number_of_trips_before_filter,
        "trips_with_task_filter": number_of_trips_after_filter,
        "requests_no_filter": number_requests_no_filter,
        "requests_with_time_filter": number_requests_after_time,
        "requests_with_task_filter": number_requests_after_tasks,
    }])

# ================================================================
# SUMMARIZE ROUTES AND TRIPS
# ================================================================
# ROUTES OR TRIPS DRIVEN PER DAY (basic statistics)
def trips_per_day(df_keep: pd.DataFrame, return_df: bool = False):
    # The number of trips (route_id) driven per day (date)
    # To give an idea of the number of trips to plan in a day

    # Ensure date is datetime
    dates = pd.to_datetime(df_keep["date"], errors="coerce")
    routes_per_day = (
        df_keep.assign(_date=dates)
               .groupby("_date")["route_id"]
               .nunique()
               .sort_index()
    )

    routes_per_day= pd.DataFrame([{
        "count_days": int(routes_per_day.shape[0]),
        "mean_routes_per_day": float(routes_per_day.mean()),
        "median_routes_per_day": float(routes_per_day.median()),
    }])

    return routes_per_day if return_df is True else None

# PLOT: TRIPS OR ROUTES DRIVEN PER DAY AND PER WEEKDAY (distribution/histogram and boxplot)
def trips_per_day_plot(
    df_keep: pd.DataFrame,
    save_plots: bool = True,
    show_plots: bool = True,
    output_dir: Optional[Union[Path, str]] = None,
    return_df: bool = False
    ):

    # Plot of the number of trips driven during a day:
        # a histogram/distribution of the number of trips during a day
        # a boxplot of the number of trips driven per weekday

    df = df_keep.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # One row per date: number of routes
    routes_by_day = (
        df.groupby("date")["route_id"]
        .nunique()
        .reset_index(name="routes_count")
    )
    routes_by_day["weekday"] = routes_by_day["date"].dt.day_name()

    # Order weekdays
    order = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]
    labels = [wd for wd in order if wd in routes_by_day["weekday"].unique()]

    # Determine output directory
    if save_plots:
        if output_dir is None:
            try:
                from learning_driver_preferences.paths import OUTPUT
                outdir = Path(OUTPUT) / "summarize_routes_and_trips"
            except Exception:
                outdir = Path("output") / "summarize_routes_and_trips"
        else:
            outdir = Path(output_dir)

        outdir.mkdir(parents=True, exist_ok=True)

    # DISTRIBUTION PLOT: routes per date
    fig1, ax1= plt.subplots()

    counts = routes_by_day["routes_count"]

    # Histogram
    counts.plot(kind="hist", bins=14, ax=ax1, label="Histogram")

    ax1.set_xticks(np.arange(counts.min(), counts.max() + 1, 10))
    ax1.set_title("Distribution of routes driven per day")
    ax1.set_xlabel("Routes per day")
    ax1.set_ylabel("Frequency")
    ax1.legend()
    fig1.tight_layout()

    if save_plots:
        fig1.savefig(outdir / "distribution_routes_per_day.png",
                     dpi=200, bbox_inches="tight", facecolor="white")

    if show_plots:
        plt.show()

    # BOX PLOT: routes per weekday
    fig2, ax2 = plt.subplots()

    data = [
        routes_by_day.loc[routes_by_day["weekday"] == wd, "routes_count"].values
        for wd in labels
    ]

    ax2.boxplot(data, labels=labels, vert=True, patch_artist=True,
                boxprops=dict(facecolor=CUSTOM_COLORS["dark_orange"], alpha=0.5))
    ax2.set_title("Routes driven per weekday (boxplot)")
    ax2.set_ylabel("Routes per day")
    fig2.tight_layout()

    if save_plots:
        fig2.savefig(outdir / "boxplot_routes_per_weekday.png",
                     dpi=200, bbox_inches="tight", facecolor="white")

    if show_plots:
        plt.show()

    return routes_by_day if return_df else None

def trips_per_route(trip_table: pd.DataFrame, show_plots: bool = True, return_df: bool = False):
    # How many times a route is driven during the observation period
    # Some routes are driven more often than others

    df= trip_table.copy()

    trips_per_route = df.groupby("route_id")["date"].nunique().rename("n_trips").sort_values(ascending=False)
    trips_per_route_df = trips_per_route.reset_index() # df from the trips_per_route, in case needed

    values = trips_per_route.to_numpy(dtype=int)

    fig, ax = plt.subplots()
    bins = np.arange(values.min(), values.max() + 2) - 0.5  # center bins on integers

    ax.hist(values, bins=bins, edgecolor="white")
    ax.set_xticks(np.arange(values.min(), values.max() + 1))
    ax.set_xlabel("Number of trips per route (distinct dates)")
    ax.set_ylabel("Number of routes")
    ax.set_title("Distribution of trips per route")
    ax.grid(axis="y", alpha=0.15)

    # --- 3) Percentile lines ---
    percentiles = [5, 10, 25, 50, 90]
    q = np.percentile(values, percentiles, method="linear")

    # Colors
    line_colors = {50: CUSTOM_COLORS["pline_median"]}
    default_color = CUSTOM_COLORS['pline_others']

    y_top = ax.get_ylim()[1]
    x_span = ax.get_xlim()[1] - ax.get_xlim()[0]

    # To avoid text collisions when several quantiles are equal, add tiny offsets
    used_positions = {}
    for p, x in zip(percentiles, q):
        color = line_colors.get(p, default_color)
        ax.axvline(x, color=color, linestyle="--", linewidth=1.5)

        # Small horizontal nudge if this x already has a label
        offset = used_positions.get(round(x, 2), 0) * (0.008 * x_span)
        used_positions[round(x, 2)] = used_positions.get(round(x, 2), 0) + 1

        ax.text(
            x + offset,
            y_top * 0.95,
            f"p{p}={int(round(x))}",
            rotation=90,
            va="top",
            ha="right",
            fontsize=9,
            backgroundcolor=(1, 1, 1, 0.5)
            )

    plt.tight_layout()

    if show_plots:
        plt.show()

    return trips_per_route_df if return_df else None

# ================================================================
# SUMMARIZE NUMBER OF TASKS
# ================================================================
# VARIABILITY IN NUMBER OF TASKS WITHIN TRIPS
# The planner makes several requests per trip (route+day). The number of tasks in a trip can change during that planning phase.
# How often does the number of tasks change during the planning of a trip? How big is the differcence?

def task_variability_within_trips(filtered_time_and_tasks_df: pd.DataFrame):
    if filtered_time_and_tasks_df.empty:
        return pd.DataFrame()

    # Ensure correct types
    df = filtered_time_and_tasks_df.copy()
    df["num_tasks"] = pd.to_numeric(df["num_tasks"], errors="coerce")
    df = df.sort_values(["route_id", "date", "request_time"], na_position="last")

    # Pre-calc: first vs last
    first_last = (
        df.groupby(["route_id", "date"])
        .apply(lambda g: pd.Series({
            "n_tasks_first_req": g["num_tasks"].iloc[0],
            "n_tasks_last_req":  g["num_tasks"].iloc[-1],
            "delta_last_first": g["num_tasks"].iloc[-1] - g["num_tasks"].iloc[0]
        }))
        .reset_index()
    )

    # Aggregate variability stats
    trip_task_var_df= (
        df.groupby(["route_id", "date"])["num_tasks"]
        .agg(
            min_tasks="min",
            max_tasks="max",
            mean_tasks="mean",
            requests_in_trip="count",
            nunique_tasks="nunique"
        )
        .reset_index()
    )

    trip_task_var_df["range_tasks"] = trip_task_var_df["max_tasks"] - trip_task_var_df["min_tasks"]
    trip_task_var_df["changed_within_trip"] = trip_task_var_df["nunique_tasks"] > 1

    # Merge first/last
    trip_task_var_df = trip_task_var_df.merge(first_last, on=["route_id", "date"], how="left")

    # Summarize trip variability statistics
    summary_trip_var = {
        "total_trips": trip_task_var_df.shape[0],
        "trips_with_changes": int(trip_task_var_df["changed_within_trip"].sum()),
        "first<last": (trip_task_var_df["delta_last_first"] > 0).sum(),
        "first>last":  (trip_task_var_df["delta_last_first"] < 0).sum(),
        "first=last":  (trip_task_var_df["delta_last_first"] == 0).sum(),
        "pct_changed": float(trip_task_var_df["changed_within_trip"].mean() * 100)
    }
    summary_trip_var_df = pd.DataFrame([summary_trip_var])


    top15_range_task_var_trip = (
        trip_task_var_df.sort_values("range_tasks", ascending=False)
                .head(15)
                [["route_id", "date", "n_tasks_first_req", "n_tasks_last_req", "range_tasks"]]
    )

    return trip_task_var_df, summary_trip_var_df, top15_range_task_var_trip

def plot_task_variability_in_trips(
    trip_task_var_df: pd.DataFrame,
    *,
    route_col: str = "route_id",
    date_col: str = "date",
    first_col: str = "n_tasks_first_req",
    last_col: str = "n_tasks_last_req",
    default_mode: str = "lollipop",   # "lollipop" or "bars"
    direction_lollipop = "vertical"   # or horizontal
    ):

    # @interactive visualize, per route_id, the number of tasks first vs last reques per trip (date), highlighting the range between them.

    # --- Validate columns
    for c in (route_col, date_col, first_col, last_col):
        if c not in trip_task_var_df.columns:
            raise ValueError(f"Column '{c}' not found in dataframe.")

    # --- Clean copy and types
    df = trip_task_var_df.copy()
    df[route_col] = df[route_col].astype(str)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    # If duplicates per route/date exist, keep a single row
    df = df.sort_values([route_col, date_col]).drop_duplicates([route_col, date_col], keep="last")

    routes = sorted(df[route_col].unique().tolist())
    if not routes:
        raise ValueError("No routes found in dataframe.")

    @interact(
        route_id=Dropdown(options=routes, description="Route:"),
        mode=ToggleButtons(options=["lollipop", "bars"], value=default_mode, description="Chart:")
    )
    def _show(route_id, mode):
        sub = (
            df.loc[df[route_col] == route_id, [date_col, first_col, last_col]]
              .sort_values(date_col)
              .reset_index(drop=True)
        )

        if sub.empty:
            print(f"No trips for route {route_id}.")
            return

        # Data arrays
        dates  = sub[date_col]
        firsts = sub[first_col].astype(float).to_numpy()
        lasts  = sub[last_col].astype(float).to_numpy()
        deltas = lasts - firsts

        # === LOLLIPOP / INTERVAL CHART (recommended) ===
        if mode == "lollipop":
            # Vertical plot
            if direction_lollipop == "vertical":
                x = np.arange(len(sub))
                fig, ax = plt.subplots(figsize=(max(10, len(sub) * 0.5), 6))

                for i in range(len(sub)):
                    y0, y1 = firsts[i], lasts[i]
                    # vertical segment at position x[i]
                    ax.plot([x[i], x[i]], [y0, y1], color=CUSTOM_COLORS["grey"], zorder=1)
                    ax.scatter([x[i]], [y0], color=CUSTOM_COLORS["dark_blue"], s=40, zorder=2, label="first" if i == 0 else "")
                    ax.scatter([x[i]], [y1], color=CUSTOM_COLORS["dark_orange"], s=40, zorder=2, label="last"  if i == 0 else "")
                    y_top = max(y0, y1)
                    color_delta = CUSTOM_COLORS["green"] if deltas[i] >= 0 else CUSTOM_COLORS["red"]
                    ax.text(x[i], y_top + 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0]), f"Δ {int(deltas[i])}", ha="center", va="bottom", color=color_delta, fontsize=9)

                ax.set_xticks(x)
                ax.set_xticklabels(dates.dt.strftime("%Y-%m-%d"), rotation=60, ha="right")
                ax.set_ylabel("Tasks")
                ax.set_title(f"First vs last tasks per trip — route {route_id}")
                ax.grid(axis="y", alpha=0.15)
                ax.legend(loc="upper left", frameon=False)

                ymin = min(firsts.min(), lasts.min())
                ymax = max(firsts.max(), lasts.max())
                ax.set_ylim(max(0, ymin - 1), ymax + max(3, 0.15 * (ymax - ymin + 1)))

            elif direction_lollipop == "horizontal":
                # Horizontal plot: y = dates, x = task counts
                y = np.arange(len(sub))
                fig, ax = plt.subplots(figsize=(max(8, len(sub) * 0.35), max(5, len(sub) * 0.45)))

                # Draw ranges as line segments and points
                for i in range(len(sub)):
                    x0, x1 = firsts[i], lasts[i]
                    ax.plot([x0, x1], [y[i], y[i]], color = CUSTOM_COLORS["grey"], zorder=1)
                    ax.scatter([x0], [y[i]],  color=CUSTOM_COLORS["dark_blue"],s=40, zorder=2, label="first" if i == 0 else "")
                    ax.scatter([x1], [y[i]],  color=CUSTOM_COLORS["dark_orange"], s=40, zorder=2, label="last"  if i == 0 else "")

                    # Annotate delta at the rightmost end
                    color_delta = CUSTOM_COLORS["green"] if deltas[i] >= 0 else CUSTOM_COLORS["red"]
                    x_txt = max(x0, x1)
                    ax.text(x_txt, y[i], f"  Δ {int(deltas[i])}", va="center", ha="left",
                            color=color_delta, fontsize=9)

                # Y axis labels = dates (string)
                ax.set_yticks(y)
                ax.set_yticklabels(dates.dt.strftime("%Y-%m-%d"))
                ax.invert_yaxis()  # latest at bottom
                ax.set_xlabel("Tasks")
                ax.set_title(f"First vs last tasks per trip — route {route_id}")
                ax.grid(axis="x", alpha=0.15)
                ax.legend(loc="lower right", frameon=False)

                # Give a little right margin for delta labels
                xmin = min(firsts.min(), lasts.min())
                xmax = max(firsts.max(), lasts.max())
                ax.set_xlim(xmin - 1, xmax + max(2, 0.06 * (xmax - xmin + 1)))

            plt.tight_layout()
            plt.show()

        # === SIDE-BY-SIDE BARS ===
        else:
            x = np.arange(len(sub))
            width = 0.42
            fig, ax = plt.subplots(figsize=(max(10, len(sub) * 0.5), 5))

            ax.bar(x - width/2, firsts, width, color= CUSTOM_COLORS["dark_blue"], label="first")
            ax.bar(x + width/2, lasts,  width, color= CUSTOM_COLORS["dark_orange"], label="last")

            # Range markers (optional): thin line between tops
            for i in range(len(sub)):
                y0 = firsts[i]
                y1 = lasts[i]
                ax.plot([x[i] - width/2, x[i] + width/2], [y0, y1], color=CUSTOM_COLORS["grey"], linewidth=1.5, zorder=3)
                y_top = max(y0, y1)
                color_delta = CUSTOM_COLORS["green"] if deltas[i] >= 0 else CUSTOM_COLORS["red"]
                ax.text(x[i], y_top + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]), f"Δ {int(deltas[i])}", ha="center", va="bottom", color=color_delta, fontsize=9)

            ax.set_xticks(x)
            ax.set_xticklabels(dates.dt.strftime("%Y-%m-%d"), rotation=60, ha="right")
            ax.set_ylabel("Tasks")
            ax.set_title(f"First vs last tasks per trip — route {route_id}")
            ax.legend(loc="upper left", frameon=False, ncol=2)
            ax.grid(axis="y", alpha=0.15)

            # Some headroom
            ymin = min(firsts.min(), lasts.min())
            ymax = max(firsts.max(), lasts.max())
            ax.set_ylim(max(0, ymin - 1), ymax + max(2, 0.06 * (ymax - ymin + 1)))

            plt.tight_layout()
            plt.show()

def delta_n_tasks_first_last_req_per_day(trip_var_df: pd.DataFrame, save_plots: bool = True, show_plots: bool = True, output_dir: Optional[Path | str] = None, ):
    df = trip_var_df.copy()
    delta_df = df.groupby("date")["delta_last_first"].sum().reset_index()

    delta_df["date"] = pd.to_datetime(delta_df["date"], errors="coerce")
    delta_df = delta_df.sort_values("date")

    if save_plots:
        if output_dir is None:
            try:
                from learning_driver_preferences.paths import OUTPUT
                outdir = Path(OUTPUT) / "task_variability"
            except Exception:
                outdir = Path("output") / "task_variability"
        else:
            outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)


    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(delta_df["date"].dt.strftime("%Y-%m-%d"), delta_df["delta_last_first"], color = CUSTOM_COLORS["dark_blue"])

    ax.set_xlabel("Date")
    ax.set_ylabel("Delta n tasks last - first request")
    ax.set_title("Daily sum of task change between last and first request per trip")

    plt.xticks(rotation=60, ha="right")

    fig.tight_layout()

    if show_plots:
        plt.show()
    else:
        plt.close()

    return delta_df

# ================================================================
# SUMMARIZE REQUESTS
# ================================================================
def summarize_requests(
    trip_table: pd.DataFrame,
    save_plots: bool = True,
    show_plots: bool = True,
    output_dir: Optional[Path | str] = None
):

    df = trip_table.copy()

    # Ensure types
    df["requests_count"] = pd.to_numeric(df["requests_count"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Derive weekday if missing
    if "weekday" not in df.columns:
        df["weekday"] = df["date"].dt.day_name()

    # REQUESTS PER ROUTE (ROUTE_ID)
    # Number of requests per route (route_id)
    route_summary = (
        df.groupby("route_id")["requests_count"]
          .agg(mean_requests_per_route="mean",
               median_requests_per_route="median",
               days_observed="count")
          .reset_index()
          .sort_values("mean_requests_per_route", ascending=False)
    )

    # Routes with the highest mean number of requests across all days observed
    top15_requests_routes = route_summary.head(15).copy()[["route_id", "mean_requests_per_route"]]

    # REQUESTS PER TRIP (ROUTE/DAY)
    # Number of requests per trip (route/day)
    mean_requests_per_trip = df["requests_count"].mean()
    median_requests_per_trip = df["requests_count"].median()

    trip_summary = pd.DataFrame([{
        "count_trips": int(df.shape[0]),
        "mean": float(mean_requests_per_trip),
        "median": float(median_requests_per_trip),
    }])

    top15_requests_trips = df.sort_values("requests_count", ascending=False).head(15).copy()[["route_id", "date", "requests_count"]]

    # Plot: distribution of requests trips
    if save_plots:
        if output_dir is None:
            try:
                from learning_driver_preferences.paths import OUTPUT
                outdir = Path(OUTPUT) / "summarize_requests"
            except Exception:
                outdir = Path("output") / "summarize_requests"
        else:
            outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        saved_path = outdir / "distribution_of_requests_per_trip.png"

        fig1, ax1 = plt.subplots()
        ax1.hist(df["requests_count"].dropna().values, bins=20, edgecolor="white")
        ax1.axvline(mean_requests_per_trip, color=CUSTOM_COLORS["pline_mean"], linestyle="--", linewidth=2, label=f"mean = {mean_requests_per_trip:.1f}")
        ax1.axvline(median_requests_per_trip, color=CUSTOM_COLORS["pline_median"], linestyle="--", linewidth=2, label=f"median = {median_requests_per_trip:.1f}")
        ax1.set_title("Requests per trip (distributie)")
        ax1.set_xlabel("Requests per trip")
        ax1.set_ylabel("Frequency")
        ax1.legend()
        fig1.tight_layout()
        fig1.savefig(saved_path, dpi=200, bbox_inches="tight", facecolor="white")

    if show_plots:
        plt.show()

    # Boxplot: number of requests per weekday
    if save_plots:
        if output_dir is None:
            try:
                from learning_driver_preferences.paths import OUTPUT
                outdir = Path(OUTPUT) / "summarize_requests"
            except Exception:
                outdir = Path("output") / "summarize_requests"
        else:
            outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        saved_path = outdir / "boxplot_of_requests_per_weekday.png"
        order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        labels = [wd for wd in order if wd in df["weekday"].unique()]
        data = [df.loc[df["weekday"] == wd, "requests_count"].dropna().values for wd in labels]

        fig2, ax2  = plt.subplots()
        ax2.boxplot(data, labels=labels, vert=True, patch_artist=True,
                boxprops=dict(facecolor=CUSTOM_COLORS["dark_orange"], alpha=0.5))
        ax2.set_title("Requests per weekday (boxplot)")
        ax2.set_ylabel("Requests per trip")
        fig2.tight_layout()
        fig2.savefig(saved_path, dpi=200, bbox_inches="tight", facecolor="white")

    if show_plots:
        plt.show()

    return route_summary, trip_summary, top15_requests_routes, top15_requests_trips

def plot_requests_and_tasks_per_trip(
    trip_table: pd.DataFrame,
    *,
    route_col: str = "route_id",
    date_col: str = "date",
    req_col: str = "requests_count",
    tasks_col: str = "mean_tasks_trip",
    default_view: str = "both",   # "requests", "tasks", "both"
    sort_by_date: bool = True
):
    # Validate columns
    for c in (route_col, date_col, req_col, tasks_col):
        if c not in trip_table.columns:
            raise ValueError(f"Column '{c}' not found in dataframe.")

    # Clean copy & dtypes
    df = trip_table.copy()
    df[route_col] = df[route_col].astype(str)
    df[date_col]  = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    # If duplicates per route/date exist, aggregate (sum requests, mean tasks)
    df = (
        df.groupby([route_col, date_col], as_index=False)
           .agg({req_col: "sum", tasks_col: "mean"})
    )

    routes = sorted(df[route_col].unique().tolist())
    if not routes:
        raise ValueError("No routes found in dataframe.")

    @interact(
        route_id=Dropdown(options=routes, description="Route:"),
        view=ToggleButtons(options=["requests", "tasks", "both"], value=default_view, description="Show:")
    )
    def _show(route_id, view):
        sub = df.loc[df[route_col] == route_id, [date_col, req_col, tasks_col]].copy()
        if sub.empty:
            print(f"No trips for route {route_id}")
            return

        if sort_by_date:
            sub = sub.sort_values(date_col)

        # Data arrays
        x = np.arange(len(sub))
        dates  = sub[date_col].dt.strftime("%Y-%m-%d").tolist()
        reqs   = sub[req_col].astype(float).to_numpy()
        tasks  = sub[tasks_col].astype(float).to_numpy()

        # Colors
        color_tasks = CUSTOM_COLORS["dark_blue"]
        color_requests = CUSTOM_COLORS["dark_orange"]

        if view in ("requests", "tasks"):
            # Single-axis bar chart
            fig, ax = plt.subplots(figsize=(max(10, len(sub) * 0.5), 5))
            if view == "requests":
                ax.bar(x, reqs, color=color_requests, edgecolor="white")
                ax.set_ylabel("Requests (count)")
                ymax = max(reqs.max(), 1)
                title_suffix = " — requests"
            else:
                ax.bar(x, tasks,color=color_tasks, edgecolor="white")
                ax.set_ylabel("Mean tasks per trip")
                ymax = max(tasks.max(), 1)
                title_suffix = " — mean tasks"

            ax.set_title(f"{route_id}{title_suffix}")
            ax.set_xticks(x)
            ax.set_xticklabels(dates, rotation=60, ha="right")
            ax.set_ylim(0, ymax * 1.15)
            ax.grid(axis="y")
            plt.tight_layout()
            plt.show()

        else:
            fig, ax_tasks = plt.subplots(figsize=(max(12, len(sub) * 0.55), 5))

            # Bars = tasks (primary y-axis)
            ax_tasks.bar(x, tasks, color=color_tasks, edgecolor="white", label="Mean tasks / trip")
            ax_tasks.set_ylabel("Mean tasks / trip")
            ax_tasks.tick_params(axis='y')
            ax_tasks.grid(axis="y")
            ax_tasks.set_ylim(0, max(tasks.max(), 1) * 1.15)

            # Line = requests (secondary y-axis)
            ax_reqs = ax_tasks.twinx()
            ax_reqs.plot(x, reqs, color=color_requests, label="Requests (count)")
            ax_reqs.set_ylabel("Requests (count)")
            ax_reqs.tick_params(axis='y')
            ax_reqs.set_ylim(0, max(reqs.max(), 1) * 1.15)

            # X axis
            ax_tasks.set_xticks(x)
            ax_tasks.set_xticklabels(dates, rotation=60, ha="right")

            # Title & combined legend
            ax_tasks.set_title(f"{route_id} — tasks (bars) + requests (line)")
            h1, l1 = ax_tasks.get_legend_handles_labels()
            h2, l2 = ax_reqs.get_legend_handles_labels()
            ax_tasks.legend(h1 + h2, l1 + l2, loc="upper right", frameon=False)


            fig.tight_layout()
            plt.show()

# # ============================================================
# # SUMMARIZE TYPE OF REQUESTS (config_name)
# # ============================================================
def summarize_type_of_requests(
    filtered_time_and_tasks_df: pd.DataFrame,
    aggregated_trips_df: pd.DataFrame,
    save_plots: bool = True,
    show_plots: bool = True,
    show_plots_route_level: bool = False,
    show_plots_weekday: bool = False,
    output_dir: Optional[Path | str] = None,
):
    # df[route_col] = df[route_col].astype(str)
    # df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    # df = df.dropna(subset=[date_col])

    # PLOT number of request type in bar chart per route_id and per trip (on trip-level) - show with interact
    route_col = "route_id"
    date_col = "date"

    df = aggregated_trips_df.copy()

    trips = sorted(df["route_id"].astype(str).unique().tolist())

    cols = ["create_count", "estimate_count", "add_count"]

    @interact(route_id=Dropdown(options=trips, description="Route:"))
    def _show(route_id):
        # Filter this route
        sub = (
            df[df[route_col] == route_id]
            .groupby(date_col, as_index=False)[list(cols)]
            .sum()
            .sort_values(date_col)
        )

        if sub.empty:
            print(f"No trips found for route {route_id}")
            return

        # Extract stacks
        create  = sub[cols[0]].astype(float).to_numpy()
        estimate  = sub[cols[1]].astype(float).to_numpy()
        add = sub[cols[2]].astype(float).to_numpy()

        x = np.arange(len(sub))
        labels = sub["date"].dt.strftime("%Y-%m-%d")

        # --- Plot ---
        fig1, ax1 = plt.subplots(figsize=(max(10, len(sub) * 0.5), 5))

        ax1.bar(x, create, label="CreateSequence")
        ax1.bar(x, estimate, bottom=create, label="EstimateTime")
        ax1.bar(x, add, bottom=create+estimate, label="AddToSequence")

        # Y-label and title
        total_max = (create + estimate + add).max()
        ax1.set_title(f"Requests per trip for route {route_id}")
        ax1.set_ylabel("Count")
        ax1.set_ylim(0, max(total_max, 1) * 1.15)

        # X axis
        ax1.set_xticks(x, labels, rotation=60, ha="right")

        # Annotation (only if not too many bars)
        if len(sub) <= 25:
            for i in range(len(sub)):
                if create[i] > 0:
                    ax1.text(x[i], create[i]/2, f"{int(create[i])}", ha="center", va="center", color="white", fontsize=9)
                if estimate[i] > 0:
                    ax1.text(x[i], create[i] + estimate[i]/2, f"{int(estimate[i])}", ha="center", va="center", color="white", fontsize=9)
                if add[i] > 0:
                    ax1.text(x[i], create[i] + estimate[i] + add[i]/2, f"{int(add[i])}", ha="center", va="center", color="white", fontsize=9)

        # Add legend
        ax1.legend(loc="upper left", frameon=False, ncol=3)

        plt.tight_layout()
        plt.show()

    # Aggregate per route (per route_id, across dates) on type of requests, start with dataframe
    aggregate_route_df = (
       aggregated_trips_df.groupby("route_id")[["estimate_count", "create_count", "add_count"]]
                  .sum()
                  .rename_axis("route_id")
    )

    aggregate_route_df["total"] = aggregate_route_df .sum(axis=1).replace(0, np.nan)
    aggregate_route_pct_df = aggregate_route_df.div(aggregate_route_df["total"], axis=0) * 100.0
    aggregate_route_pct_df = aggregate_route_pct_df.rename(columns={
        "estimate_count": "estimate_pct",
        "create_count": "create_pct",
        "add_count": "add_pct"
    })
    aggregate_route_pct_df  = aggregate_route_pct_df.reset_index()

    top15_routes_estimate = aggregate_route_pct_df.head(15)[["route_id", "estimate_pct"]]

    # PLOT number of request type in bar chart per route_id; on route-level (across days) - show with interact
    df2 = aggregate_route_df.copy()

    # Ensure route_id is index
    if "route_id" in df2.columns:
       df2 = df2.set_index("route_id")

    cols = ["create_count", "estimate_count", "add_count"]

    # Dropdown options
    routes = df2.index.tolist()

    @interact(route_id=Dropdown(options=routes, description="Route:"))
    def _show(route_id):
        row = df2.loc[route_id, cols]
        create, estimate, add = row.tolist()

        fig2, ax2 = plt.subplots()

        # Stacked bar
        ax2.bar([0], [create], label="CreateSequence")
        ax2.bar([0], [estimate], bottom=create, label="EstimateTime")
        ax2.bar([0], [add], bottom=create+estimate,  label="AddToSequence")

        # Annotation
        def annotate(y0, h, txt):
            if h > 0:
                ax2.text(0, y0 + h/2, txt, ha="center", va="center", color="white", fontsize=10)

        annotate(0, create, f"{int(create)}")
        annotate(create, estimate, f"{int(estimate)}")
        annotate(estimate+create, add, f"{int(add)}")


        ax2.set_xticks([0])
        ax2.set_xticklabels(["Requests"])
        ax2.set_title(f"Requests for route {route_id}")
        ax2.set_ylabel("Count")
        ax2.set_ylim(0, max(estimate + create + add, 1) * 1.25)

        ax2.legend(loc="upper right")

        if show_plots_route_level:
            plt.show()
        else:
            plt.close()

    # Aggregate all requests to be able to make 1 overall plot of request types
    aggregate_all_df = (
        aggregate_route_df[["create_count", "estimate_count", "add_count"]]
        .sum()
        .to_frame(name="total_across_all_requests")
    )

    aggregate_all_plot = aggregate_all_df.rename(index={
        "create_count":   "CreateSequence",
        "estimate_count": "EstimateTime",
        "add_count":      "AddSequence"
    })

    # PLOT share of config_name across all requests
    if save_plots:
        if output_dir is None:
            try:
                from learning_driver_preferences.paths import OUTPUT
                outdir = Path(OUTPUT) / "summarize_requests"
            except Exception:
                outdir = Path("output") / "summarize_requests"
        else:
            outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        saved_path = outdir / "bar_plot_type_of_requests.png"

        colors = [CUSTOM_COLORS["dark_blue"], CUSTOM_COLORS["dark_orange"], CUSTOM_COLORS["dark_green"]]

    fig3, ax3 = plt.subplots()

    bars = ax3.bar(
        aggregate_all_plot.index,
        aggregate_all_plot["total_across_all_requests"],
        color=colors
    )

    ax3.bar_label(
        bars,
        labels=[f"{int(v):.0f}" for v in aggregate_all_plot["total_across_all_requests"]],
        padding=3,                # distance in points above the bar
        fontsize=10

    )

    ax3.set_xlabel("Request type")
    ax3.set_ylabel("Number of requests")
    ax3.set_title("Total requests across all trips")

    fig3.tight_layout()
    fig3.savefig(saved_path, dpi=200, bbox_inches="tight", facecolor="white")

    if show_plots:
        plt.show()

    # BOX PLOT: EstimateTime requests per weekday
    if save_plots:
        if output_dir is None:
            try:
                from learning_driver_preferences.paths import OUTPUT
                outdir = Path(OUTPUT) / "summarize_requests"
            except Exception:
                outdir = Path("output") / "summarize_requests"
        else:
            outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        # Ensure weekday exists
        df3 = aggregated_trips_df.copy()
        if "weekday" not in df3.columns:
            if "date" in df3.columns:
                df3["date"] = pd.to_datetime(df3["date"], errors="coerce")
                df3["weekday"] = df3["date"].dt.day_name()
            else:
                raise ValueError("trip_table must include 'weekday' or 'date' to derive 'weekday'.")

        # Safety: make sure EstimateTime is numeric (counts)
        df3["estimate_count"] = pd.to_numeric(df3["estimate_count"], errors="coerce").fillna(0)

        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        labels = [wd for wd in order if wd in df3["weekday"].unique()]

        # include zeros (distribution across all trips)
        saved_path_all = outdir / "boxplot_estimatetime_per_weekday_all_trips.png"
        data_all = [df3.loc[df3["weekday"] == wd, "estimate_count"].values for wd in labels]

    fig4, ax4 = plt.subplots()
    ax4.boxplot(data_all, labels=labels, vert=True, patch_artist=True,
                boxprops=dict(facecolor=CUSTOM_COLORS["dark_orange"], alpha=0.5))
    ax4.set_title("EstimateTime requests per weekday (all trips)")
    ax4.set_ylabel("EstimateTime-requests per trip")

    fig4.tight_layout()
    fig4.savefig(saved_path_all, dpi=200, bbox_inches="tight", facecolor="white")

    if show_plots_weekday:
        plt.show()
    else:
        plt.close()

    return aggregate_route_df, aggregate_route_pct_df, top15_routes_estimate

# # # ============================================================
# # # (4) RELATIONSHIPS (#tasks vs #requests; #tasks vs Estimate/Create shares)
# # # ============================================================
# def relationship_analysis(trip_table: pd.DataFrame):
#     # Correlate workload and behavior
#     # Returns Pearson and Spearman correlations, plus simple slopes (np.polyfit).

#     out = {}

#     def _corr_and_slope(x: pd.Series, y: pd.Series) -> Dict[str, float]:
#         x = pd.to_numeric(x, errors="coerce")
#         y = pd.to_numeric(y, errors="coerce")
#         m = pd.DataFrame({"x": x, "y": y}).dropna()
#         if m.shape[0] < 3:
#             return {"pearson": np.nan, "spearman": np.nan, "slope": np.nan, "n": m.shape[0]}
#         pear = m["x"].corr(m["y"], method="pearson")
#         spear = m["x"].corr(m["y"], method="spearman")
#         slope = float(np.polyfit(m["x"], m["y"], 1)[0])  # y ~ a*x + b, return a
#         return {"pearson": float(pear), "spearman": float(spear), "slope": slope, "n": int(m.shape[0])}

#     # A) Are more tasks -> more requests?
#     out["tasks_vs_requests"] = _corr_and_slope(trip_table["mean_tasks_trip"], trip_table["requests_count"])

#     # B) Are more tasks -> higher EstimateTime share?
#     out["tasks_vs_estimate_pct"] = _corr_and_slope(trip_table["mean_tasks_trip"], trip_table["estimate_pct"])

#     # C) Are more tasks -> higher CreateSequence share?
#     out["tasks_vs_create_pct"] = _corr_and_slope(trip_table["mean_tasks_trip"], trip_table["create_pct"])

#     # D) (Optional) Are more requests -> higher EstimateTime share?
#     out["requests_vs_estimate_pct"] = _corr_and_slope(trip_table["requests_count"], trip_table["estimate_pct"])

#     return out
