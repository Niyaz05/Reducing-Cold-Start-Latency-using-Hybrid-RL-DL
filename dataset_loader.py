"""
dataset_loader.py
-----------------
Loads the Azure Functions 2019 dataset from a folder of Excel (.xlsx) or
CSV (.csv) files.  The Azure dataset has two main file types:

  invocations_per_function_md.anon.d01.csv  (or .xlsx)
    Columns: HashOwner, HashApp, HashFunction, 1, 2, ..., 1440
    Each numbered column = invocations in that minute of the day.

  function_durations_percentiles.anon.d01.csv  (or .xlsx)
    Columns: HashOwner, HashApp, HashFunction, Average, 1, 25, 50, 75, 95, 99, 100
    We use 'Average' as a proxy for cold-start overhead duration.

This loader:
  1. Scans `data_dir` for all .xlsx / .csv files.
  2. Separates invocation files from duration files by column names.
  3. Melts invocation files from wide (1440 columns) to long format.
  4. Merges duration data as cold_start_ms.  If no duration file exists,
     cold_start_ms is simulated from a log-normal distribution so the
     three CASR overhead buckets (0-1s, 1-60s, >60s) are proportionally
     represented, matching the Azure dataset statistics cited in the paper.
  5. Returns a clean DataFrame ready for CASREnv.

Output DataFrame columns:
    func_id        str   Anonymised function hash
    day            int   Day number (1-14)
    arrival_minute int   Minute of the day (1-1440)
    count          int   Invocations in that minute
    cold_start_ms  float Cold-start duration in milliseconds
    memory_mb      float Simulated memory footprint (MB)
"""

import os
import re
import numpy as np
import pandas as pd
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _is_invocation_file(df: pd.DataFrame) -> bool:
    """Return True if the DataFrame looks like an invocation trace file."""
    # Invocation files have integer column names 1..1440
    numeric_cols = [c for c in df.columns if str(c).isdigit()]
    return len(numeric_cols) >= 100


def _is_duration_file(df: pd.DataFrame) -> bool:
    """Return True if the DataFrame looks like a duration percentile file."""
    return "Average" in df.columns or "average" in df.columns


def _read_file(path: str) -> pd.DataFrame:
    """Read either a .csv or .xlsx file into a DataFrame."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path, low_memory=False)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(path, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _extract_day(filename: str) -> int:
    """
    Try to parse the day number from the filename.
    Recognises patterns like: d01, day01, day_01, _1_, etc.
    Falls back to 1 if not found.
    """
    match = re.search(r"[dD](?:ay[_-]?)?(\d{1,2})", os.path.basename(filename))
    if match:
        return int(match.group(1))
    # fallback: look for any 1-2 digit number in the filename
    digits = re.findall(r"\d{1,2}", os.path.basename(filename))
    return int(digits[0]) if digits else 1


def _simulate_cold_start_ms(func_ids: pd.Series, rng: np.random.Generator) -> pd.Series:
    """
    Simulate cold_start_ms for each unique function when no duration file
    is available.  We sample from a log-normal mixture to reproduce the
    overhead distribution reported in the CASR paper:
        ~55%  functions:  0 – 1 000 ms   (Q0 bucket)
        ~30%  functions:  1 000 – 60 000 ms (Q1 bucket)
        ~15%  functions:  > 60 000 ms    (Q2 bucket)
    """
    unique_funcs = func_ids.unique()
    n = len(unique_funcs)
    n_q0 = int(n * 0.55)
    n_q1 = int(n * 0.30)
    n_q2 = n - n_q0 - n_q1

    # Q0: log-normal centred around 500 ms
    q0 = rng.lognormal(mean=6.2, sigma=0.6, size=n_q0)       # ~500ms
    # Q1: log-normal centred around 10 000 ms
    q1 = rng.lognormal(mean=9.2, sigma=0.8, size=n_q1)       # ~10s
    # Q2: log-normal centred around 120 000 ms
    q2 = rng.lognormal(mean=11.7, sigma=0.5, size=n_q2)      # ~120s

    durations = np.concatenate([q0, q1, q2])
    rng.shuffle(durations)

    func_duration_map = dict(zip(unique_funcs, durations))
    return func_ids.map(func_duration_map)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def load_azure_dataset(
    data_dir: str,
    sample_rows: Optional[int] = 300_000,
    min_count: int = 1,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Load and preprocess the Azure Functions 2019 dataset.

    Parameters
    ----------
    data_dir : str
        Folder containing the dataset files (.xlsx or .csv).
        Put ALL your Azure dataset files in this folder.
    sample_rows : int or None
        Maximum number of rows to keep after preprocessing.
        Set to None to load everything (may be very large).
    min_count : int
        Drop rows where invocation count < min_count.
    random_seed : int
        Seed for reproducibility of sampling and simulation.

    Returns
    -------
    pd.DataFrame with columns:
        func_id, day, arrival_minute, count, cold_start_ms, memory_mb
    """
    rng = np.random.default_rng(random_seed)

    invocation_frames = []
    duration_frames = []

    files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith((".csv", ".xlsx", ".xls"))
    ])

    if not files:
        raise FileNotFoundError(
            f"No CSV or Excel files found in '{data_dir}'.\n"
            "Please put your Azure Functions dataset files there."
        )

    print(f"[DatasetLoader] Found {len(files)} file(s) in '{data_dir}'")

    for fpath in files:
        try:
            df = _read_file(fpath)
        except Exception as e:
            print(f"  [WARN] Could not read {os.path.basename(fpath)}: {e}")
            continue

        day = _extract_day(fpath)

        if _is_invocation_file(df):
            print(f"  [OK] Invocation file  → {os.path.basename(fpath)}  (day={day})")
            # Identify the HashFunction column (case-insensitive search)
            hash_col = next(
                (c for c in df.columns if "hashfunction" in str(c).lower()),
                df.columns[2]   # fallback: assume 3rd column
            )
            # Keep only numeric minute columns
            minute_cols = [c for c in df.columns if str(c).isdigit()]
            subset = df[[hash_col] + minute_cols].copy()
            subset = subset.rename(columns={hash_col: "func_id"})
            # Melt wide → long
            melted = subset.melt(
                id_vars="func_id",
                var_name="arrival_minute",
                value_name="count",
            )
            melted["arrival_minute"] = melted["arrival_minute"].astype(int)
            melted["count"] = pd.to_numeric(melted["count"], errors="coerce").fillna(0).astype(int)
            melted["day"] = day
            # Drop zero-invocation rows early to save memory
            melted = melted[melted["count"] >= min_count]
            invocation_frames.append(melted)

        elif _is_duration_file(df):
            print(f"  [OK] Duration file    → {os.path.basename(fpath)}  (day={day})")
            hash_col = next(
                (c for c in df.columns if "hashfunction" in str(c).lower()),
                df.columns[2]
            )
            avg_col = next(
                (c for c in df.columns if str(c).lower() == "average"),
                None
            )
            if avg_col is None:
                print(f"  [WARN] No 'Average' column found in {os.path.basename(fpath)}, skipping.")
                continue
            dur_df = df[[hash_col, avg_col]].copy()
            dur_df.columns = ["func_id", "cold_start_ms"]
            dur_df["cold_start_ms"] = pd.to_numeric(dur_df["cold_start_ms"], errors="coerce")
            dur_df = dur_df.dropna(subset=["cold_start_ms"])
            duration_frames.append(dur_df)
        else:
            print(f"  [SKIP] Unrecognised format → {os.path.basename(fpath)}")

    if not invocation_frames:
        raise ValueError(
            "No invocation files were found/loaded.\n"
            "The loader expects files with 1440 numeric minute columns\n"
            "(matching the Azure Functions dataset format)."
        )

    # ── Combine all invocation data ──────────────────────────────────────────
    print("[DatasetLoader] Combining invocation frames...")
    combined = pd.concat(invocation_frames, ignore_index=True)

    # ── Attach cold-start duration ───────────────────────────────────────────
    if duration_frames:
        print("[DatasetLoader] Merging duration data...")
        all_durations = pd.concat(duration_frames, ignore_index=True)
        # Average across duplicate entries
        all_durations = all_durations.groupby("func_id", as_index=False)["cold_start_ms"].mean()
        combined = combined.merge(all_durations, on="func_id", how="left")
        # Fill missing durations with simulation
        missing_mask = combined["cold_start_ms"].isna()
        if missing_mask.any():
            combined.loc[missing_mask, "cold_start_ms"] = _simulate_cold_start_ms(
                combined.loc[missing_mask, "func_id"], rng
            ).values
    else:
        print("[DatasetLoader] No duration files found — simulating cold_start_ms...")
        combined["cold_start_ms"] = _simulate_cold_start_ms(combined["func_id"], rng).values

    # ── Simulate memory footprint ────────────────────────────────────────────
    # Memory is correlated with cold-start overhead (heavier deps = more memory)
    # We model it as: memory_mb = base + fraction_of_cold_start_time
    # This matches the Azure dataset distribution where avg memory ~ 170 MB
    base_mem = rng.lognormal(mean=4.8, sigma=0.9, size=len(combined))  # ~170 MB median
    overhead_factor = (combined["cold_start_ms"].values / 1000.0) * 5.0
    combined["memory_mb"] = np.clip(base_mem + overhead_factor, 64, 3000)

    # ── Final cleanup ─────────────────────────────────────────────────────────
    combined = combined[["func_id", "day", "arrival_minute", "count", "cold_start_ms", "memory_mb"]]
    combined = combined.dropna()
    combined["cold_start_ms"] = combined["cold_start_ms"].clip(lower=10.0)
    combined = combined.reset_index(drop=True)

    print(f"[DatasetLoader] Total rows before sampling: {len(combined):,}")

    # ── Sample ────────────────────────────────────────────────────────────────
    if sample_rows is not None and len(combined) > sample_rows:
        combined = combined.sample(n=sample_rows, random_state=random_seed).reset_index(drop=True)
        print(f"[DatasetLoader] Sampled to {len(combined):,} rows")

    print(f"[DatasetLoader] Done. {len(combined):,} rows, "
          f"{combined['func_id'].nunique():,} unique functions.\n"
          f"  cold_start_ms stats: min={combined['cold_start_ms'].min():.0f}  "
          f"mean={combined['cold_start_ms'].mean():.0f}  "
          f"max={combined['cold_start_ms'].max():.0f}")

    return combined


def compute_workload_features(df: pd.DataFrame) -> dict:
    """
    Extract summary statistics from a DataFrame slice that characterise the
    workload type.  These features are the INPUT to the Meta-model.

    Returns a dict of scalar features.
    """
    cs = df["cold_start_ms"].values
    counts = df["count"].values

    # Overhead bucket fractions
    frac_q0 = float(np.mean(cs < 1_000))
    frac_q1 = float(np.mean((cs >= 1_000) & (cs < 60_000)))
    frac_q2 = float(np.mean(cs >= 60_000))

    return {
        "mean_invocations_per_minute": float(counts.mean()),
        "std_invocations_per_minute": float(counts.std() + 1e-8),
        "mean_cold_start_ms": float(cs.mean()),
        "std_cold_start_ms": float(cs.std() + 1e-8),
        "frac_low_overhead": frac_q0,
        "frac_medium_overhead": frac_q1,
        "frac_high_overhead": frac_q2,
        "mean_memory_mb": float(df["memory_mb"].mean()),
        "n_unique_functions": float(df["func_id"].nunique()),
        "total_invocations": float(counts.sum()),
    }
