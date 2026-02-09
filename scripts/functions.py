from __future__ import annotations

import json
import os
import pandas as pd

from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Iterable, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator, FixedLocator

from config import SCRIPTS_DIR
plt.style.use(SCRIPTS_DIR / "plot_style.mplstyle")

#=============================================================
# CREATE DF OF OUTPUT FILES
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





#========================================
# DEFINE CUTOFF TIME AND TASKS
#========================================
# ============================================================
# FUNCTION TO PARSE TIME
# ============================================================
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
    save: bool = True,
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
    save: bool = True,
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
    perc_list = percentiles or [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
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
    label_map = {0.05: "P5", 0.10: "P10", 0.25: "P25", 0.50: "Median", 0.75: "P75", 0.90: "P90", 0.95: "P95"}
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
