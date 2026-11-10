"""Data loading module for FAQ datasets."""

from __future__ import annotations

import os
from typing import Set

import pandas as pd


def load_faqs(filepath: str) -> pd.DataFrame:
    """Load FAQ dataset from CSV or JSON into a validated Pandas DataFrame.

    Args:
        filepath: Path to the dataset (.csv or .json).

    Returns:
        pd.DataFrame containing the validated FAQ entries.
    """
    if not os.path.exists(filepath):
        # Auto-detect relative paths from project root or src
        candidates = [
            filepath,
            os.path.join("data", os.path.basename(filepath)),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", os.path.basename(filepath)),
            os.path.join(os.path.dirname(__file__), "..", "data", os.path.basename(filepath)),
        ]
        found = None
        for cand in candidates:
            if os.path.exists(cand):
                found = cand
                break
        if found:
            filepath = found
        else:
            # Remote fallback if running in Google Colab or detached directory
            remote_url = f"https://raw.githubusercontent.com/eyosialex/AWS-Exercies/main/AWS_week_2/data/{os.path.basename(filepath)}"
            try:
                import urllib.request
                target_dir = os.path.dirname(filepath) or "data"
                os.makedirs(target_dir, exist_ok=True)
                download_path = os.path.join(target_dir, os.path.basename(filepath))
                urllib.request.urlretrieve(remote_url, download_path)
                print(f"📥 Downloaded dataset from remote fallback: {remote_url}")
                filepath = download_path
            except Exception:
                raise FileNotFoundError(f"FAQ dataset file not found at: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(filepath, quotechar='"', skipinitialspace=True)
    elif ext == ".json":
        df = pd.read_json(filepath)
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Must be .csv or .json")

    # Schema validation
    required_cols: Set[str] = {"question", "answer"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Dataset missing required columns: {required_cols - set(df.columns)}. "
            f"Available columns: {list(df.columns)}"
        )

    # Defaults for optional metadata
    if "id" not in df.columns:
        df["id"] = range(1, len(df) + 1)

    if "category" not in df.columns:
        df["category"] = "General"

    return df
