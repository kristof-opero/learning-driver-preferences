from __future__ import annotations

from learning_driver_preferences.plot_style import set_plot_style, CUSTOM_COLORS
set_plot_style()

from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Iterable, List
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator, FixedLocator

# ============================================================
# FUNCTION TO READ CSV OR EXCEL AND NORMALIZE COLUMNS
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
    if "triggertype" in lower_map:
        rename_pairs[lower_map["triggertype"]] = "config_name"
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

    info = {
        "selected_depot": selected_depot,
        "source": "csv" if is_csv else "excel",
        "filename": path.name,
    }

    print(f"file information: {info}")

    return df, info

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
    ax.plot(counts.index, counts.values, color=CUSTOM_COLORS["dark_blue"], linewidth=line_width, marker=None, markersize=0, linestyle = "-", label="Requests/min")

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
            try:
                from learning_driver_preferences.paths import OUTPUT
                outdir = Path(OUTPUT) / "distribution_req_per_minute"
            except Exception:
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
        try:
            from learning_driver_preferences.paths import OUTPUT
            outdir = Path(OUTPUT) / "distribution_num_tasks"
        except Exception:
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
    sns.histplot(s, bins=bins, color=CUSTOM_COLORS["dark_blue"], ax=ax, stat="count")
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
