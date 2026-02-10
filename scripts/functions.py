from __future__ import annotations

import json
import os
import pandas as pd

from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Iterable, List, Union

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator, FixedLocator
import datetime as dt
import re

from ipywidgets import interact, Dropdown, ToggleButtons

from config import SCRIPTS_DIR
plt.style.use(SCRIPTS_DIR / "plot_style.mplstyle")

CUSTOM_COLORS = {
    "pline_mean": "#ff0000", # red
    "pline_median":  "#ee6363", # rose
    "pline_others": "#fc9208", # orange
    "red":  "#ca0a0a", # red
    "green":  "#05C505", # green
    "grey":  "#8C8C8C", # grey
    "light_orange":'#D4A373',   # light orange
    "light_green": '#6B8E7F',   # light green
    }

#=============================================================
# CREATE DF OF DATA INPUT FILES
#============================================================
def create_df_of_requests(path_to_parent_folder):
    dirs = os.listdir(path_to_parent_folder)
    if "README.txt" in dirs:
        dirs.pop(dirs.index("README.txt")) # Remove README.txt folder

    rows = []

    for dir in dirs:
        path = os.path.join(path_to_parent_folder, dir)
        if os.path.isdir(path):
            files = os.listdir(path)
            for file in files:
                route_id = file.split('-')[0].split('_')[1]
                date = pd.to_datetime(file.split('-')[1], format="%Y%m%d")
                time = pd.to_datetime(file.split('-')[2], format="%H%M%S").time()
                with open(os.path.join(path, file), "r") as f:
                    data = json.load(f)
                    request_type = data["configurationName"]
                    request_id = data["id"]
                    tasks = data["tasks"]
                    fixed_tasks = [task["taskId"] for task in data["fixedTasks"]]
                    for task in tasks:
                        row = {}
                        row_id = file.split('.')[-2] + '-' + task["id"]
                        row["row_id"] = row_id
                        row["route_id"] = route_id
                        row["date"] = date
                        row["time"] = time
                        row["request_id"] = request_id
                        row["request_type"] = request_type
                        row["task_id"] = task["id"]
                        loc = task["address"]
                        lat = loc["latitude"]
                        lon = loc["longitude"]
                        row["lat"] = lat
                        row["lon"] = lon
                        row["location_id"] = str(int(lat * 10 ** 8)) + str(int(lon * 10 ** 8))
                        row["fixed"] = True if row["task_id"] in fixed_tasks else False
                        row["position_fixed"] = (
                            fixed_tasks.index(row["task_id"]) if row["fixed"] else None
                        )
                        rows.append(row)

    return pd.DataFrame(rows)


def create_df_of_responses(path_to_parent_folder):
    dirs = os.listdir(path_to_parent_folder)

    rows = []

    for dir in dirs:
        path = os.path.join(path_to_parent_folder, dir)
        if os.path.isdir(path):
            files = os.listdir(path)
            for file in files:
                route_id = file.split('-')[0].split('_')[1]
                date = pd.to_datetime(file.split('-')[1], format="%Y%m%d")
                time = pd.to_datetime(file.split('-')[2], format="%H%M%S").time()
                with open(os.path.join(path, file), "r") as f:
                       tasks = f.readlines()
                       for task in tasks:
                        row = {}
                        row_id = file.split('.')[-2] + '-' + task.strip()
                        row["row_id"] = row_id
                        row['route_id'] = route_id
                        row['date'] = date
                        row['time'] = time
                        row['task_id'] = task.strip()
                        row['task_sequence_number'] = tasks.index(task) + 1
                        rows.append(row)

    return pd.DataFrame(rows)


def join_requests_and_responses(requests_df, responses_df):
    return requests_df.merge(
        responses_df[["row_id", "task_sequence_number"]],
        on='row_id',
        how="inner"
    )

#=========================================================
# COUNT NUMBER OF TASKS AND COMPARE START AND END REQUEST
#=========================================================
def count_tasks_per_sequence(df):
    counted_df = (df.groupby(["route_id", "date", "time", "request_type"])["location_id"]
        .nunique()
        .reset_index(name="count")
        )
    counted_df["date"] = counted_df["date"].dt.strftime("%Y-%m-%d")
    return counted_df


def select_compare_start_and_end(df):
    rows = []

    for (r, d), g in df.groupby(["route_id", "date"]):
        g_sorted = g.sort_values(by="time", ascending=True)
        types = g_sorted["request_type"].unique().tolist()
        create_sequence = types[::-1].index("CreateSequence") if "CreateSequence" in types else (len(types) - 1)
        idx = len(types) - create_sequence - 1
        counts = g_sorted["count"].to_list()
        row = {
            "route_id": r,
            "date": d,
            "count_start": counts[idx],
            "count_end": counts[-1],
            "abs_diff_end_start": counts[-1] - counts[idx],
            "pct_diff_end_start": (counts[-1] - counts[idx]) / counts[idx] * 100
        }
        rows.append(row)

    return pd.DataFrame(rows)


#=======================================
# READ FROM INPUTFILE IN CSV OR EXCEL
#=======================================
def read_inputfile(
    inputfile: str | Path,
    depot: Optional[str] = "0521",
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
                    t = dt.datetime.strptime(x, fmt).time()
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

#========================================
# DEFINE CUTOFF TIME AND TASKS
#========================================
# Helper function to parse time
def parse_request_times(df: pd.DataFrame):
    # parses ddf['request_time'] if present. Returns a Series[datetime64[ns]]

    if "request_time" not in df.columns:
        return pd.Series([], dtype="datetime64[ns]")

    time_str = df["request_time"].astype(str).str.strip()
    parsed_time= pd.to_datetime(time_str, format="%H:%M:%S", errors="coerce")
    if parsed_time.isna().all():
        parsed_time = pd.to_datetime(time_str, format="%I:%M:%S.%f %p", errors="coerce")
    if parsed_time.isna().all():
        parsed_time = pd.to_datetime(time_str, format="%I:%M:%S %p", errors="coerce")
    if parsed_time.isna().all():
        parsed_time = pd.to_datetime(time_str, errors="coerce")
    return parsed_time

# =========================================================================
# FUNCTION TO DEFINE CUTOFF TIME — DISTRIBUTION OF REQUESTS MADE OVER TIME
# =========================================================================
# Vraag = welk tijdsegment nemen we in acht bij het analyseren van de requests? Methode: het aantal requests in de tijd uitzetten, nl. distributie van aantal requests per minuut over alle routes en dagen heen. Zo kan je zien op welke momenten requests gemaakt worden en kan je bepalen op welk tijdstip 85%, 90%, 95%... van de requests gemaakt zijn.

def define_cutoff_time(
    # inputfile: Path | str,
    df: pd.DataFrame,
    nonrelevant_hour_from: Optional[int] = 19,  # None => include whole day
    depot: Optional[str] = None, *,
    line_width: float = 1.2,
    percentiles: Iterable[int] = (85, 90, 95, 98),
    output_dir: Optional[Path | str] = None,  # if None, tries OUTPUT, else ./output
    filename: Optional[str] = None,  # auto-generated if None
    save: bool = False,
    save_dpi: int = 200,
    show: bool = True,
):
    # Plot distribution of requests per minute-of-day and draw percentile cut lines.
    # Returns: fig, ax, info (dict with 'selected_depot', 'n_requests_kept')

    # df, info_in = read_inputfile(inputfile, depot=depot)

    # path = Path(inputfile)
    if "request_time" not in df.columns:
         raise KeyError("Input must contain a parsable 'request_time' column.")

    times = parse_request_times(df)

    # DEFINE REQUEST TIMES TO INCLUDE AND PREPARE HOURS FOR X-AXIS OF THE PLOT
    if nonrelevant_hour_from is None:
        cutoff_hours = 24
        keep_mask = times.notna()
        cutoff_label_short = "full-day"
        subtitle = "full day"
    else:
        cutoff_hours = int(nonrelevant_hour_from)
        if cutoff_hours < 1 or cutoff_hours > 24:
            raise ValueError("nonrelevant_hour_from must be in [1, 24] or None.")
        keep_mask = times.dt.hour < cutoff_hours
        cutoff_label_short = f"before{cutoff_hours:02d}h"
        subtitle = f"< {cutoff_hours:02d}:00"

    cutoff_minutes = cutoff_hours * 60
    max_minute = cutoff_minutes - 1

    times_kept = times[keep_mask].dropna()

    minute_of_day = (times_kept.dt.hour * 60 + times_kept.dt.minute).astype(int)
    counts = (
        minute_of_day.value_counts()
        .sort_index()
        .reindex(range(max_minute + 1), fill_value=0)
    )

    # CREATE CUMULATIVE PERCENTILES
    total_requests = int(counts.sum())
    p_targets = tuple(int(p) for p in percentiles)
    p_minutes: Dict[int, int] = {}
    p_labels: Dict[int, str] = {}
    if total_requests > 0:
        cum_counts = counts.cumsum().values.astype(np.int64)
        cum_frac = cum_counts / total_requests

        def minute_for_percent(p: float) -> int:
            idx = int(np.searchsorted(cum_frac, p, side="left"))
            return min(idx, max_minute)

        for p in sorted(set(k for k in p_targets if 1 <= k <= 99)):
            m = minute_for_percent(p / 100.0)
            p_minutes[p] = m
            hh, mm = divmod(m, 60)
            p_labels[p] = f"{hh:02d}:{mm:02d}"

    # PLOT
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(counts.index, counts.values, linewidth=line_width, marker=None, markersize=0, linestyle = "-", label="Requests/min")

    hours_ticks = list(range(0, cutoff_minutes + 1, 60))
    ax.set_xlim(0, cutoff_minutes)
    ax.xaxis.set_major_locator(FixedLocator(hours_ticks))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, cutoff_hours + 1)])
    ax.xaxis.set_minor_locator(MultipleLocator(15))
    ax.tick_params(axis="x", which="major", length=7, width=1.0, color="#333")
    ax.tick_params(axis="x", which="minor", length=4, width=0.8, color="#888", labelbottom=False)
    ax.grid(which="minor", axis="x", linestyle=":", linewidth=0.6, color="#bbb", alpha=0.5)
    ax.grid(which="major", axis="x", linestyle="--", linewidth=0.8, color="#888", alpha=0.7)

    ax.set_xlabel("Time of day")
    ax.set_ylabel("Number of requests")
    ax.set_title(f"Requests per minute from depot {depot}")

    if total_requests > 0 and p_minutes:
        base_colors = {85: "#d67d30", 90: "#f16101", 95: "#e64e3d", 98: "#ad4444"}
        ymax = max(counts.max(), 1)
        for k in sorted(p_minutes.keys()):
            color = base_colors.get(k, "#6c5ce7")
            xm = p_minutes[k]
            ax.axvline(x=xm, color=color, linestyle="--", linewidth=1.2, label=f"p{k} @ {p_labels[k]}")
            ax.text(
                xm + 2, ymax * 0.95, f"p{k}\n{p_labels[k]}",
                color=color, fontsize=9, va="top",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2),
            )
    else:
        ax.text(0.5, 0.5, "No requests in selection", transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color="#555")

    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()

    # SAVE PLOT TO FILE
    saved_path = None
    if save:
        if output_dir is None:
            outdir = Path("output") / "distribution_req_per_minute"
        else:
            outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            depot_sfx = f"{depot} if {depot} else "
            filename = f"cutoff_time_{cutoff_label_short}{depot_sfx}.png"
        saved_path = outdir / filename
        fig.savefig(saved_path, dpi=save_dpi, bbox_inches="tight", facecolor="white")

    if show:
        plt.show()
    else:
        plt.close(fig)

    info = {
        "selected_depot": depot,
        "n_requests_kept": total_requests,
    }

    print(f"information: {info}")
    return fig, ax, info

# ============================================================
# 3) DEFINE CUTOFF STOPS — DISTRIBUTION OF num_tasks
# ============================================================
# Vraag = hoeveel tasks moeten er minimaal in een route zitten opdat we de route_id in acht nemen het analyseren van de requests? Methode:

def define_cutoff_stops(
    # inputfile: Path | str,
    df: pd.DataFrame,
    depot: Optional[str] = None,
    *,
    column: str = "num_tasks",
    percentiles: Iterable[float] | None = None,  # 0–1 scale (e.g., [0.05,0.10,0.25,0.5])
    bins: int | str = 40,            # or 'auto'/'fd'
    output_dir: Optional[Path | str] = None,
    filename_prefix: Optional[str] = None,   # base name; auto if None
    save: bool = False,
    save_dpi: int = 200,
    show: bool = True,
):
    # Compute distribution stats for 'num_tasks' and derive cutoff stops.
    # Returns: dict(summary=DataFrame, cutoffs=list[float], info=dict)

    # READ INPUTFILE CSV OR EXCEL
    # df, info_in = read_inputfile(inputfile, depot=depot)

    # FILTER OUT TIMES AFTER 19h
    times = parse_request_times(df)

    if not times.empty:
        keep_mask = times.dt.hour < 19
        df = df.loc[keep_mask].copy()


    # COLUMN "num_tasks" HAS TO EXIST OR TO BE RENAMED
    if column not in df.columns:
        # Try common alternates (safety)
        alternates = ["NumberOfTasks", "numberOfTasks", "NumberofTasks"]
        for alt in alternates:
            if alt in df.columns:
                df = df.rename(columns={alt: "num_tasks"})
                break

    if column not in df.columns:
        raise ValueError(f"No column '{column}' found after normalization.")

    s = pd.to_numeric(df[column], errors="coerce").dropna()
    if s.empty:
        raise ValueError(f"Column '{column}' contains no valid numeric values after filtering.")

    # # Prepare a prettier series for plotting (optional clipping)
    # s_for_plot = s.copy()

    # CALCULATE STATS
    perc_list = percentiles or [0.05, 0.10, 0.25, 0.50]
    desc = s.describe(percentiles=perc_list)
    summary = desc.to_frame(name="value").reset_index().rename(columns={"index": "stat"})

    # CALCULATE CUTOFFS
    cutoffs = [float(np.quantile(s, p)) for p in perc_list]

    # CREATE OUTPUT DIR IF NOT EXISTS
    if output_dir is None:
             outdir = Path("output") / "distribution_num_tasks"
    else:
        outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # CREATE FILENAME
    base = filename_prefix
    if not base:
        base = f"{column}_from_depot: {depot}"

    # STAVE STATS IN FILE
    if save:
        summary.to_csv(outdir / f"{base}_distribution_stats.csv", index=False)

    # PLOT HISTOGRAM
    # Basic histogram
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.histplot(s, bins=bins, ax=ax, stat="count")
    ax.set_title(f"Distribution of {column} - depot: {depot}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")

    # Set lines for each cutoff
    for x in cutoffs:
        ax.axvline(x, color="red", ls="--", lw=1)

    # Label cutoff lines
    ymax = ax.get_ylim()[1]
    label_map = {0.05: "P5", 0.10: "P10", 0.25: "P25", 0.50: "Median"}
    for p in perc_list:
        p_rounded = round(float(p), 2)
        lab = label_map.get(p_rounded)
        if lab:
            x = float(np.quantile(s, p))
            ax.text(x + 0.05, ymax * 0.95, lab, color="red", ha="left", va="top", fontsize=9, rotation=90)

    fig.tight_layout()
    if save:
        fig.savefig(outdir / f"{base}_distribution.png", dpi=save_dpi, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    else:
        plt.close(fig)

    info = {
        "selected_depot": {depot},
        "n_rows": int(df.shape[0]),
    }

    print(f"information: {info}")
    return summary, info

# ======================================================================================================================================
# ANALYSIS HELPERS: FILTER REQUESTS ON CUTOFF TIME AND TASKS, ANNOTATE SEQUENCE NUMBER OF REQUESTS, CREATE A TABLE WITH 1 ROW PER TRIP
# ======================================================================================================================================
def filter_on_cutoff_time_and_tasks(
 # the combination of route_id and day/date = "a trip" (NL = "rit") = the execution of that route/route_id on a particular day
    df: pd.DataFrame,
    cutoff_time: dt.time = dt.time(11, 0, 0),  # 11:00
    cutoff_mean_tasks: float = 30.0    # keep only trips (route/day) with mean tasks > 30 (mean tasks = mean of num_tasks across the requests for that trip)
    ):
#     """Apply first-round filter (if column exists) and time cutoff; keep mornings with mean tasks > threshold."""
#     # Standardize key columns (they may or may not be present depending on source)
#     if "date" not in df.columns:
#         raise ValueError("Input must include a 'date' column")
#     if "request_time" not in df.columns:
#         raise ValueError("Input must include a 'request_time' column")
#     if "route_id" not in df.columns:
#         raise ValueError("Input must include a 'route_id' column")
#     if "config_name" not in df.columns:
#         raise ValueError("Input must include a 'config_name' colum")
#     if "num_tasks" not in df.columns:
#         raise ValueError("Input must include a 'num_tasks' column")

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

def trips_per_day_plot(
    filtered_df: pd.DataFrame,
    save_plots: bool = False,
    show_plots: bool = False,
    show_weekday_plot: bool = True,
    output_dir: Optional[Union[Path, str]] = None,
    return_df: bool = False
    ):

    # Plot of the number of trips driven during a day:
        # a histogram/distribution of the number of trips during a day
        # a boxplot of the number of trips driven per weekday

    df = filtered_df.copy()
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
    else:
        plt.close()

    # BOX PLOT: routes per weekday
    fig2, ax2 = plt.subplots()

    data = [
        routes_by_day.loc[routes_by_day["weekday"] == wd, "routes_count"].values
        for wd in labels
    ]

    ax2.boxplot(data, labels=labels, vert=True, patch_artist=True,
                boxprops=dict(facecolor="white"))
    ax2.set_title("Routes driven per weekday (boxplot)")
    ax2.set_ylabel("Routes per day")
    fig2.tight_layout()

    if save_plots:
        fig2.savefig(outdir / "boxplot_routes_per_weekday.png",
                     dpi=200, bbox_inches="tight", facecolor="white")

    if show_weekday_plot:
        plt.show()
    else:
        plt.close()

    return routes_by_day if return_df else None

# ================================================================
# SUMMARIZE REQUESTS
# ================================================================
def summarize_requests(
    trip_df: pd.DataFrame,
    save_plots: bool = False,
    show_plots: bool = True,
    output_dir: Optional[Path | str] = None
):

    df = trip_df.copy()

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

    # REQUESTS PER TRIP (ROUTE/DAY)
    # Number of requests per trip (route/day)
    mean_requests_per_trip = df["requests_count"].mean()
    median_requests_per_trip = df["requests_count"].median()

    # Plot: distribution of requests trips
    if save_plots:
        if output_dir is None:
            outdir = Path("output") / "summarize_requests"
        else:
            outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        saved_path = outdir / "distribution_of_requests_per_trip.png"

    fig1, ax1 = plt.subplots()
    ax1.hist(df["requests_count"].dropna().values, bins=20, edgecolor="white")
    ax1.axvline(mean_requests_per_trip, color=CUSTOM_COLORS["pline_mean"], linestyle="--", linewidth=2, label=f"mean = {mean_requests_per_trip:.1f}")
    ax1.axvline(median_requests_per_trip, color=CUSTOM_COLORS["pline_median"],  linestyle="--", linewidth=2, label=f"median = {median_requests_per_trip:.1f}")
    ax1.set_title("Requests per trip (distributie)")
    ax1.set_xlabel("Requests per trip")
    ax1.set_ylabel("Frequency")
    ax1.legend()
    fig1.tight_layout()

    if save_plots:
        fig1.savefig(saved_path, dpi=200, bbox_inches="tight", facecolor="white")

    if show_plots:
        plt.show()

    return route_summary

#============================================================
# SUMMARIZE TYPE OF REQUESTS (config_name)
#============================================================
def summarize_type_of_requests(
    trip_df: pd.DataFrame,
    save_plots: bool = False,
    show_plots: bool = True,
    output_dir: Optional[Path | str] = None,
):
    # PLOT number of request type in bar chart per route_id and per trip (on trip-level) - show with interact
    route_col = "route_id"
    date_col = "date"

    df = trip_df.copy()

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
       trip_df.groupby("route_id")[["estimate_count", "create_count", "add_count"]]
                  .sum()
                  .rename_axis("route_id")
    )

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
            outdir = Path("output") / "summarize_requests"
        else:
            outdir = Path(output_dir)

        outdir.mkdir(parents=True, exist_ok=True)

        saved_path = outdir / "bar_plot_type_of_requests.png"

    fig3, ax3 = plt.subplots(figsize=(7,4))

    colors = [CUSTOM_COLORS[k] for k in ["grey", "light_orange", "light_green"]]

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
    if save_plots:
        fig3.savefig(saved_path, dpi=200, bbox_inches="tight", facecolor="white")

    if show_plots:
        plt.show()

    return aggregate_route_df

# ================================================================
# SUMMARIZE NUMBER OF TASKS
# ================================================================
# VARIABILITY IN NUMBER OF TASKS WITHIN TRIPS
# The planner makes several requests per trip (route+day). The number of tasks in a trip can change during that planning phase.
# How often does the number of tasks change during the planning of a trip? How big is the differcence?

def task_variability_within_trips(filtered_time_and_tasks_df: pd.DataFrame, show_plots: bool = True):
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

    df = summary_trip_var_df.copy()

    columns = ["first=last", "first<last", "first>last"]  # x-axis order

    # Handle single-row summary DataFrame
    row = df.iloc[0]

    # Build the data in the specified order
    values = [int(row[w]) for w in columns]

    # Compute percentages if total_trips is present
    total = int(row["total_trips"]) if "total_trips" in df.columns else None
    percentages = [(v / total * 100) if total and total > 0 else None for v in values]

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(7, 4))

    colors = [CUSTOM_COLORS[k] for k in ["grey", "light_orange", "light_green"]]

    bars = ax.bar(columns, values, color=colors)

    ax.set_title("Number of tasks first vs last request", pad=12)
    ax.set_xlabel("Category")
    ax.set_ylabel("Frequence")

    # Add value labels on top of bars (and % if available)
    for bar, v, p in zip(bars, values, percentages):
        height = bar.get_height()
        if p is not None:
            ax.annotate(f"{v:,}\n({p:.1f}%)",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom")
        else:
            ax.annotate(f"{v:,}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom")

    # Tidy up y-axis to start at 0
    ax.set_ylim(0, max(values) * 1.15 if values else 1)

    # Optional: light grid
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    if show_plots:
        plt.show()
    else:
        plt.close()

    return trip_task_var_df, summary_trip_var_df

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
                    ax.scatter([x[i]], [y0], color=CUSTOM_COLORS["light_orange"], s=40, zorder=2, label="first" if i == 0 else "")
                    ax.scatter([x[i]], [y1], color=CUSTOM_COLORS["light_green"], s=40, zorder=2, label="last"  if i == 0 else "")
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
                    ax.scatter([x0], [y[i]],  color=CUSTOM_COLORS["light_orange"],s=40, zorder=2, label="first" if i == 0 else "")
                    ax.scatter([x1], [y[i]],  color=CUSTOM_COLORS["light_green"], s=40, zorder=2, label="last"  if i == 0 else "")

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

            ax.bar(x - width/2, firsts, width, color=CUSTOM_COLORS["light_orange"], label="first")
            ax.bar(x + width/2, lasts,  width, color= CUSTOM_COLORS["light_green"], label="last")

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


# To test hypothese that the number of tasks stays stable during a working day - so the sum of the delta first - last request should be 0
# Want: we gaan ervan uit dat de taken die uit de ene route gehaald worden, bij een andere route toegevoegd worden
# Test over depot 0521, per dag en test over alle depots heen
def delta_n_tasks_first_last_req_per_day(trip_var_df: pd.DataFrame, save_plots: bool = False, show_plots: bool = True, output_dir: Optional[Path | str] = None, ):
    df = trip_var_df.copy()
    delta_df = df.groupby("date")["delta_last_first"].sum().reset_index()

    delta_df["date"] = pd.to_datetime(delta_df["date"], errors="coerce")
    delta_df = delta_df.sort_values("date")

    if save_plots:
        if output_dir is None:
            outdir = Path("output") / "task_variability"
        else:
            outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()

    ax.bar(delta_df["date"].dt.strftime("%Y-%m-%d"), delta_df["delta_last_first"])

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


#========================================================================
# CALCULATE DIFFERENCE BETWEEN RESPONSE FILES WITH LEVENSTHEIN DISTANCE
#========================================================================
# Question: how many changes are there in the outcome of the requests? Compare last create sequence and last estimate time-request
# Remark: Levensthein distance measures every change between two strings, so what is needed to go from str1 to str2 (added, deleted, changed order) - so this does not measure only the order change

# Config
ROUTE_COL   = "route_id"
DATE_COL    = "date"           # dtype: date
TIME_COL    = "request_time"   # dtype: datetime
CONFIG_COL  = "config_name"    # contains 'CreateSequence' and 'EstimateTime' (case sensitive)

CREATE_VALUE   = "CreateSequence"
ESTIMATE_VALUE = "EstimateTime"

# If filenames time don’t exactly match request_time, pick the nearest file within tolerance:
NEAREST_MATCH_TOLERANCE_SECONDS = 60  # set 0 to require exact hhmmss match

# =========================
# Helpers
# =========================
def _to_yyyymmdd(d):
    """Accepts date or datetime-like -> 'YYYYMMDD'."""
    return pd.to_datetime(d).strftime("%Y%m%d")

def _to_hhmmss(t):
    if isinstance(t, dt.time):
        return t.strftime("%H%M%S")
    ts = pd.to_datetime(t, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.strftime("%H%M%S")


def _read_sequence_txt(file_path: Path):
    """Read one task_id per line. Ignore blanks. Keep as strings."""
    seq = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # If somebody exported comma-separated on one line; be lenient:
            parts = re.split(r"[,\s]+", s)
            for p in parts:
                p = p.strip()
                if p:
                    seq.append(p)
    return seq

def _levenshtein_list(a, b):
    """Levenshtein distance between two lists."""
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    if m > n:
        a, b = b, a
        n, m = m, n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,       # delete
                curr[j - 1] + 1,   # insert
                prev[j - 1] + cost # substitute
            )
        prev = curr
    return prev[m]

def _extract_hhmmss_from_filename(name: str, route_id: str, yyyymmdd: str):
    """
    Returns 'HHMMSS' if filename looks like: <route>-<yyyymmdd>-<hhmmss>...
    """
    rid = re.escape(route_id)
    ymd = re.escape(yyyymmdd)
    m = re.match(rf"^{rid}-{ymd}-(\d{{6}})", name)
    return m.group(1) if m else None

def _list_candidate_files(folder: Path, route_id: str, yyyymmdd: str):
    """All .txt files in folder that start with <route>-<yyyymmdd>-..."""
    if not folder.is_dir():
        return []
    prefix = f"{route_id}-{yyyymmdd}-"
    return sorted([p for p in folder.glob("*.txt") if p.name.startswith(prefix)])

def _pick_exact_or_nearest(folder: Path, route_id: str, yyyymmdd: str,
                           hhmmss_target: str, tolerance_sec: int):
    """
    Choose a file with exact HHMMSS; if none and tolerance>0, pick the nearest within tolerance.
    Returns Path or None.
    """
    candidates = _list_candidate_files(folder, route_id, yyyymmdd)
    if not candidates:
        return None

    # 1) Exact match by prefix
    exact_prefix = f"{route_id}-{yyyymmdd}-{hhmmss_target}"
    exact = [p for p in candidates if p.name.startswith(exact_prefix)]
    if exact:
        return sorted(exact)[-1]  # If multiple, pick lexicographically last

    if tolerance_sec <= 0:
        return None

    # 2) Nearest time within tolerance
    def to_dt(hhmmss):
        return dt.strptime(hhmmss, "%H%M%S")

    t_target = to_dt(hhmmss_target)
    nearest = None
    best_delta = None
    for p in candidates:
        hh = _extract_hhmmss_from_filename(p.name, route_id, yyyymmdd)
        if not hh:
            continue
        delta = abs((to_dt(hh) - t_target).total_seconds())
        if best_delta is None or delta < best_delta:
            nearest = p
            best_delta = delta

    if nearest is not None and best_delta is not None and best_delta <= tolerance_sec:
        return nearest

    return None

def _last_of_type(g: pd.DataFrame, config_value: str):
    """Row of the last request of given type within the group (by TIME_COL)."""
    sub = g[g[CONFIG_COL] == config_value]
    if sub.empty:
        return None
    sub = sub.sort_values(by=[TIME_COL])
    return sub.iloc[-1]

# =========================
# Main compute Levenshtein
# =========================
# Calculate the differences between the response files corresponding to the last create_sequence-request and the last estimate_time-request
# Last Create_Sequence and Last Estimate_time omdat bij de laatste create_sequence het aantal taken grotendeels vastligt (is niet altijd zo, maar meestal wel)

def compute_levenshtein_from_df(
    df: pd.DataFrame,
    responses_dir: Path,
    output_dir: Optional[Path | str] = None,
    save_csv: bool = True,
    tolerance_seconds: int = NEAREST_MATCH_TOLERANCE_SECONDS
):
    """
    Compute Levenshtein-afstanden per (route_id, date) tussen de sequences
    van het laatste CreateSequence- en EstimateTime-verzoek.

    Parameters
    ----------
    df : DataFrame
        Must contain columns: ROUTE_COL, DATE_COL, TIME_COL, CONFIG_COL
        where date is date dtype & request_time is datetime dtype.
    responses_dir : Path
        Root folder with <route>-<yyyymmdd> subfolders.
    output_dir : Path
        Where outputs are written.
    tolerance_seconds : int
        If >0: allow nearest file within this tolerance if exact hhmmss not found.

    Returns
    -------
    result_df : DataFrame
        One row per trip (route_id, date) with distances and metadata.
    """
    if save_csv:
        if output_dir is None:
            outdir = Path("output") / "task_variability"
        else:
            outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)

    req_cols = {ROUTE_COL, DATE_COL, TIME_COL, CONFIG_COL}
    missing = req_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input df is missing required columns: {missing}")

    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL]).dt.date

    rows = []
    not_found = []

    for (route_id, date_val), grp in df.groupby([ROUTE_COL, DATE_COL], as_index=False):
        last_cs = _last_of_type(grp, CREATE_VALUE)
        last_et = _last_of_type(grp, ESTIMATE_VALUE)
        if last_cs is None or last_et is None:
            continue

        yyyymmdd = _to_yyyymmdd(date_val)
        hhmmss_cs = _to_hhmmss(last_cs[TIME_COL])
        hhmmss_et = _to_hhmmss(last_et[TIME_COL])
        folder = Path(responses_dir) / f"{route_id}-{yyyymmdd}"

        file_cs = _pick_exact_or_nearest(folder, route_id, yyyymmdd, hhmmss_cs, tolerance_seconds)
        file_et = _pick_exact_or_nearest(folder, route_id, yyyymmdd, hhmmss_et, tolerance_seconds)

        if not file_cs or not file_et:
            not_found.append({
                "route_id": route_id,
                "date": str(date_val),
                "missing": "CS" if not file_cs else ("ET" if not file_et else "both"),
                "expected_folder": str(folder),
                "requested_time_cs": hhmmss_cs,
                "requested_time_et": hhmmss_et
            })
            continue

        seq_cs = _read_sequence_txt(file_cs)
        seq_et = _read_sequence_txt(file_et)

        dist = _levenshtein_list(seq_cs, seq_et)
        denom = max(len(seq_cs), len(seq_et))
        dist_norm = (dist / denom) if denom else 0.0

        rows.append({
            "route_id": route_id,
            "date": pd.to_datetime(date_val),
            "distance": dist,
            "distance_norm": dist_norm,
            "len_cs": len(seq_cs),
            "len_et": len(seq_et),
            "file_cs": str(file_cs.relative_to(responses_dir)),
            "file_et": str(file_et.relative_to(responses_dir)),
            "time_cs": hhmmss_cs,
            "time_et": hhmmss_et
        })

    result_df = pd.DataFrame(rows).sort_values(["route_id", "date"]).reset_index(drop=True)

    # Save outputs
    csv_path = outdir / "levenshtein_by_trip.csv"
    result_df.to_csv(csv_path, index=False)

    # Log missing, if any
    if not_found:
        pd.DataFrame(not_found).to_csv(outdir / "missing_files_log.csv", index=False)

    return result_df

# ====================================================================================================================
# Plot bar chart with Levenshtein differences, per route (interact) and per date - both normalized and not normalized
# ====================================================================================================================

def plot_route_grouped_bars(
    result_df: pd.DataFrame,
    route_id: str,
    metrics: tuple[str, str] = ("distance", "distance_norm"),
    date_format: str = "%Y-%m-%d",
    rotate_xticks: int = 45,
    annotate_values: bool = True,
    width: float = 0.36,
    metrics_on_axes: tuple[str, str] | None = None,  # e.g. ("left","right") or ("right","left")
    right_ylim: tuple[float, float] | None = (0.0, 1.0),  # normalized is typically 0..1
    left_label: str | None = None,
    right_label: str | None = None,
):
    """
    Grouped bars with two y-axes for a single route_id across exact dates.
    - Left axis plots metrics[0]
    - Right axis plots metrics[1] (default) so normalized stays visible

    Parameters
    ----------
    result_df : DataFrame
        Output of compute_levenshtein_from_df. Must contain 'route_id', 'date', and the metric columns.
    route_id : str
        Route to plot.
    metrics : (str, str)
        (left_metric, right_metric) by default. If metrics_on_axes is provided,
        you can flip axes assignment.
    colors : (str, str)
        Colors for the left and right bars.
    right_ylim : (float, float) | None
        y-limits for the right axis. Default (0,1) suits 'distance_norm'.
    metrics_on_axes : ("left","right") or ("right","left") or None
        If None, defaults to ("left","right") -> metrics[0]=left, metrics[1]=right.
    """

    # Basic checks
    if metrics_on_axes is None:
        metrics_on_axes = ("left", "right")
    if set(metrics_on_axes) != {"left", "right"}:
        raise ValueError("metrics_on_axes must be a permutation of ('left','right').")

    left_metric = metrics[0] if metrics_on_axes[0] == "left" else metrics[1]
    right_metric = metrics[1] if metrics_on_axes[1] == "right" else metrics[0]

    req_cols = {"route_id", "date", left_metric, right_metric}
    missing = req_cols - set(result_df.columns)
    if missing:
        raise KeyError(f"Missing required columns for plotting: {sorted(missing)}")

    df = result_df.loc[result_df["route_id"] == route_id, ["route_id", "date", left_metric, right_metric]].copy()
    if df.empty:
        print(f"No rows found for route_id={route_id}")
        return

    # Normalize dtypes
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date")

    # X positioning
    x_labels = [pd.to_datetime(d).strftime(date_format) for d in df["date"]]
    x = np.arange(len(df))

    # Prepare figure and twin axes
    fig, ax = plt.subplots(figsize=(12,5))
    ax2 = ax.twinx()

    # Bar positions: left and right metric offset around x
    b1 = ax.bar(x - width/2, df[left_metric].values, width=width, color=CUSTOM_COLORS["grey"], label=left_metric)
    b2 = ax2.bar(x + width/2, df[right_metric].values, width=width, color=CUSTOM_COLORS["light_orange"], label=right_metric)

    # Titles and labels
    title = f"{route_id} — {left_metric.replace('_',' ').title()} (left) vs {right_metric.replace('_',' ').title()} (right)"
    ax.set_title(title)
    ax.set_xlabel("date")
    ax.set_ylabel(left_label if left_label else left_metric.replace("_", " ").title())
    ax2.set_ylabel(right_label if right_label else right_metric.replace("_", " ").title())
    ax.set_xticks(x, x_labels, rotation=rotate_xticks, ha="right")

    # Optional: fix right axis limits for normalized data
    if right_ylim is not None and right_ylim[0] < right_ylim[1]:
        ax2.set_ylim(*right_ylim)

    # Grid on left axis to help reading bars
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Merge legends (handles artists from both axes)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper left", frameon=False)

    # Annotate values (formatted per axis)
    if annotate_values:
        # Left bars annotation
        for b in b1:
            y = b.get_height()
            if pd.isna(y):
                continue
            ax.text(
                b.get_x() + b.get_width()/2, y,
                f"{y:.0f}",
                ha="center", va="bottom", fontsize=9, color=CUSTOM_COLORS["grey"]
            )
        # Right bars annotation
        for b in b2:
            y = b.get_height()
            if pd.isna(y):
                continue
            ax2.text(
                b.get_x() + b.get_width()/2, y,
                f"{y:.2f}",
                ha="center", va="bottom", fontsize=9, color=CUSTOM_COLORS["light_orange"]
            )

    plt.tight_layout()
    plt.show()

# ==================================================
# Distribution of normalized Levenstheid distances
# ==================================================
def plot_distribution_levensthein(result_df):
    # df = pd.read_csv(inputfile)

    df = result_df.copy()
    # -- Pick the normalized distance column --
    col = "distance_norm" if "distance_norm" in df.columns else None
    if col is None:
        # Try to auto-detect a plausible column name
        candidates = [c for c in df.columns if "norm" in c.lower() and "dist" in c.lower()]
        if not candidates:
            raise ValueError("Could not find a normalized distance column. Expected 'distance_norm'.")
        col = candidates[0]

    # -- Clean/clip the data to [0, 1] --
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    s = s[(s >= 0) & (s <= 1)]
    if s.empty:
        raise ValueError(f"No valid values in '{col}' after cleaning.")

    # -- Colors (use your palette if present) --
    try:
        bar_color    = CUSTOM_COLORS.get("grey", "#8C8C8C")
        mean_color   = CUSTOM_COLORS.get("pline_mean",  "#d62728")  # red
        median_color = CUSTOM_COLORS.get("pline_median","#ff7f0e")  # orange
    except NameError:
        bar_color, mean_color, median_color = "#8C8C8C", "#d62728", "#ff7f0e"

    # -- Histogram --
    fig, ax = plt.subplots(figsize=(8,4))

    # Fixed bins from 0.0 to 1.0 inclusive; change the number of bins if you want finer/coarser detail
    bins = np.linspace(0, 1, 31)  # 30 bins
    n, b, patches = ax.hist(s, bins=bins, color=bar_color, edgecolor="white", alpha=0.95)

    # Mean & median
    m, med = s.mean(), s.median()
    ax.axvline(m,   color=mean_color,   linestyle="--", linewidth=1.6, label=f"Mean = {m:.3f}")
    ax.axvline(med, color=median_color, linestyle=":",  linewidth=1.6, label=f"Median = {med:.3f}")

    # Labels & styling
    ax.set_xlim(0, 1)
    ax.set_xlabel("Normalized Levenshtein distance")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of normalized Levenshtein distance")
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.6)
    ax.legend()
    plt.tight_layout()
    plt.show()
