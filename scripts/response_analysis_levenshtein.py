from __future__ import annotations

from learning_driver_preferences.plot_style import set_plot_style, CUSTOM_COLORS
set_plot_style()

from pathlib import Path
import re
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from typing import Optional, Tuple, Dict, Any, List, Union

# =========================
# Config
# =========================
ROUTE_COL   = "route_id"
DATE_COL    = "date"           # dtype: date
TIME_COL    = "request_time"   # dtype: datetime
CONFIG_COL  = "config_name"    # contains 'CreateSequence' and 'EstimateTime' (case sensitive)

CREATE_VALUE   = "CreateSequence"
ESTIMATE_VALUE = "EstimateTime"

# Which metric to plot on Y (raw or normalized)
PLOT_NORMALIZED = True   # set True to plot normalized distance

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
            try:
                from learning_driver_preferences.paths import OUTPUT
                outdir = Path(OUTPUT) / "task_variability"
            except Exception:
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

# ==============================================================================================================
# Plot bar chart with Levenshtein differences, per route (interact) and per date - normalized or not normalized
# ==============================================================================================================

def plot_route_bar(
    result_df: pd.DataFrame,
    route_id: str,
    metric: str = "distance",     # or "distance_norm"
    title: str | None = None,
    annotate_values: bool = True,
    date_format: str = "%Y-%m-%d",
    sort_dates: bool = True,
    rotate_xticks: int = 45
):
    """
    Bar plot per exact date for one route_id.

    Parameters
    ----------
    result_df : DataFrame
        Output from compute_levenshtein_from_df; must contain 'route_id', 'date', and metric column.
    route_id : str
        Route to plot.
    metric : str
        'distance' or 'distance_norm'.
    """

    req_cols = {"route_id", "date", metric}
    missing = req_cols - set(result_df.columns)
    if missing:
        raise KeyError(f"Missing required columns for plotting: {sorted(missing)}")

    df = result_df.loc[result_df["route_id"] == route_id, ["route_id", "date", metric]].copy()
    if df.empty:
        print(f"No rows found for route_id={route_id}")
        return

    # Ensure date is datetime (in case it's dtype 'object' or 'date')
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # Optional: sort by date
    if sort_dates:
        df = df.sort_values("date")

    # Prepare x labels (exact dates)
    x_labels = [pd.to_datetime(d).strftime(date_format) for d in df["date"]]
    x_pos = range(len(df))

    # Plot
    fig, ax = plt.subplots()
    bars = ax.bar(x_pos, df[metric].values, width=0.8)

    # Titles & labels
    ttl = title if title is not None else f"{route_id} — {metric.replace('_', ' ').title()} per date"
    ax.set_title(ttl)
    ax.set_xlabel("date")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xticks(ticks=x_pos, labels=x_labels, rotation=rotate_xticks, ha="right")

    # Annotate values on bars (optional)
    if annotate_values:
        for b, val in zip(bars, df[metric].values):
            y = b.get_height()
            ax.text(
                b.get_x() + b.get_width()/2, y,
                f"{val:.2f}" if isinstance(val, float) else str(val),
                ha="center", va="bottom", fontsize=9
            )

    plt.tight_layout()
    plt.show()

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
    b1 = ax.bar(x - width/2, df[left_metric].values, width=width, color=CUSTOM_COLORS["dark_blue"], label=left_metric)
    b2 = ax2.bar(x + width/2, df[right_metric].values, width=width, color=CUSTOM_COLORS["dark_orange"], label=right_metric)

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
                ha="center", va="bottom", fontsize=9, color=CUSTOM_COLORS["dark_blue"]
            )
        # Right bars annotation
        for b in b2:
            y = b.get_height()
            if pd.isna(y):
                continue
            ax2.text(
                b.get_x() + b.get_width()/2, y,
                f"{y:.2f}",
                ha="center", va="bottom", fontsize=9, color=CUSTOM_COLORS["dark_orange"]
            )

    plt.tight_layout()
    plt.show()
