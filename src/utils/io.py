"""
I/O utilities for persisting datasets, results, and models.

All save helpers embed lightweight metadata (format version, timestamp,
row/column counts, etc.) so that files are self-describing and downstream
loaders can sanity-check them before use.
"""

from __future__ import annotations

import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    """Create *path* and all intermediate directories if they do not exist.

    Equivalent to ``mkdir -p``.

    Parameters
    ----------
    path:
        Target directory path.

    Returns
    -------
    Path
        The resolved ``Path`` object (guaranteed to exist on return).
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# DataFrame I/O
# ---------------------------------------------------------------------------

def save_dataframe(
    df: pd.DataFrame,
    path: str | Path,
    format: Literal["parquet", "csv"] = "parquet",
) -> Path:
    """Persist a DataFrame to disk with light metadata.

    For Parquet, schema and row-count metadata are stored in the file itself.
    For CSV, a companion ``<name>.meta.json`` file is written alongside.

    Parameters
    ----------
    df:
        The DataFrame to save.
    path:
        Full output path (including filename).  Parent directories are created
        automatically.
    format:
        ``"parquet"`` (default, strongly preferred) or ``"csv"``.

    Returns
    -------
    Path
        Resolved path to the written file.
    """
    p = Path(path)
    ensure_dir(p.parent)

    if format == "parquet":
        df.to_parquet(p, index=True, engine="pyarrow")
    elif format == "csv":
        df.to_csv(p, index=True)
        meta = {
            "saved_at": _now_iso(),
            "rows": len(df),
            "columns": list(df.columns),
            "format": "csv",
        }
        with open(p.with_suffix(".meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format!r}.  Use 'parquet' or 'csv'.")

    return p


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a DataFrame, auto-detecting format from the file extension.

    Parameters
    ----------
    path:
        Path to a ``.parquet`` or ``.csv`` file.

    Returns
    -------
    pd.DataFrame
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p}")

    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(p, engine="pyarrow")
    elif suffix == ".csv":
        return pd.read_csv(p, index_col=0)
    else:
        raise ValueError(f"Cannot infer format from extension {suffix!r}.")


# ---------------------------------------------------------------------------
# Results JSON I/O
# ---------------------------------------------------------------------------

def save_results(results_dict: Dict[str, Any], path: str | Path) -> Path:
    """Persist a results dictionary to a timestamped JSON file.

    Non-JSON-serialisable values (``numpy`` scalars, ``Path`` objects, etc.)
    are coerced to Python built-ins automatically.

    Parameters
    ----------
    results_dict:
        Flat or nested mapping of metric names → values.
    path:
        Output file path.

    Returns
    -------
    Path
        Resolved path to the written file.
    """
    p = Path(path)
    ensure_dir(p.parent)

    payload = {
        "_meta": {"saved_at": _now_iso(), "format_version": "1.0"},
        "results": results_dict,
    }

    def _default(obj: Any) -> Any:
        """JSON encoder fallback for numpy scalars and paths."""
        import numpy as np  # noqa: WPS433

        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")

    with open(p, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_default)

    return p


def load_results(path: str | Path) -> Dict[str, Any]:
    """Load a results JSON file saved by :func:`save_results`.

    Parameters
    ----------
    path:
        Path to the JSON file.

    Returns
    -------
    dict
        The ``results`` sub-dictionary (metadata stripped).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Results file not found: {p}")

    with open(p, encoding="utf-8") as fh:
        payload = json.load(fh)

    return payload.get("results", payload)


# ---------------------------------------------------------------------------
# Model serialisation
# ---------------------------------------------------------------------------

def save_model(
    model: Any,
    path: str | Path,
    model_name: str,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """Pickle a trained model alongside a metadata sidecar.

    Parameters
    ----------
    model:
        Any pickle-able model object (sklearn, XGBoost, LightGBM, etc.).
    path:
        Output path for the ``.pkl`` file.
    model_name:
        Human-readable model identifier written to the metadata.
    extra_meta:
        Optional dict of additional metadata fields to include.

    Returns
    -------
    Path
        Resolved path of the saved pickle file.
    """
    p = Path(path)
    if p.suffix != ".pkl":
        p = p.with_suffix(".pkl")
    ensure_dir(p.parent)

    with open(p, "wb") as fh:
        pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)

    meta: Dict[str, Any] = {
        "model_name": model_name,
        "saved_at": _now_iso(),
        "pickle_protocol": pickle.HIGHEST_PROTOCOL,
        "model_class": type(model).__name__,
    }
    if extra_meta:
        meta.update(extra_meta)

    meta_path = p.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    return p


def load_model(path: str | Path) -> Any:
    """Load a pickled model from *path*.

    Parameters
    ----------
    path:
        Path to the ``.pkl`` file.

    Returns
    -------
    Any
        The deserialised model object.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")

    with open(p, "rb") as fh:
        return pickle.load(fh)
