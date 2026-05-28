"""
Logging utilities built on loguru.

loguru's design is a single global logger; we simulate named loggers by
binding `name` into the context, which shows up in every log record.
Rotation is size-based so long experiments don't blow out a single file.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from loguru import logger
from omegaconf import DictConfig, OmegaConf


def setup_logger(
    log_level: str = "INFO",
    log_file: str | Path | None = None,
    rotation: str = "100 MB",
    retention: str = "30 days",
    enqueue: bool = True,
) -> None:
    """Configure loguru sinks for the experiment run.

    Parameters
    ----------
    log_level:
        Minimum severity forwarded to both sinks. Accepts loguru level names.
    log_file:
        Path to the rotating log file. If ``None``, only stdout is used.
    rotation:
        loguru rotation trigger — size string, time string, or callable.
    retention:
        How long completed log files are kept before deletion.
    enqueue:
        Thread-safe async queueing for file sink; keeps hot-path fast.
    """
    # Remove the default stderr sink so we control the format fully.
    logger.remove()

    fmt_stdout = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{extra[name]}</cyan> | "
        "<level>{message}</level>"
    )
    fmt_file = (
        "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
        "{extra[name]} | {function}:{line} | {message}"
    )

    logger.configure(extra={"name": "root"})  # default binding for bare logger calls

    logger.add(
        sys.stdout,
        level=log_level.upper(),
        format=fmt_stdout,
        colorize=True,
    )

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_path),
            level=log_level.upper(),
            format=fmt_file,
            rotation=rotation,
            retention=retention,
            enqueue=enqueue,
            encoding="utf-8",
        )


def get_logger(name: str) -> "logger":  # type: ignore[type-arg]
    """Return a context-bound logger that tags every record with *name*.

    Parameters
    ----------
    name:
        Logical component name, e.g. ``"preprocessing"`` or ``"xgboost_trainer"``.
    """
    return logger.bind(name=name)


def log_experiment_start(cfg: DictConfig) -> None:
    """Emit a structured block at experiment launch so runs are self-documenting.

    Parameters
    ----------
    cfg:
        The fully resolved Hydra config for this run.
    """
    _log = get_logger("experiment")
    _log.info("=" * 72)
    _log.info("Experiment starting")
    _log.info("Full resolved config:\n{}", OmegaConf.to_yaml(cfg, resolve=True))
    _log.info("=" * 72)


def log_metrics(
    metrics_dict: dict[str, Any],
    step: int | None = None,
    prefix: str = "",
) -> None:
    """Log a flat metrics dictionary in a consistent, parseable format.

    Each key-value pair is emitted on its own line with an optional *prefix*
    and *step* so that log-scraping pipelines (e.g. grep + awk) can extract
    numbers without parsing YAML or JSON.

    Parameters
    ----------
    metrics_dict:
        Mapping of metric name → scalar value.
    step:
        Training step / epoch / fold index. Omitted from output when ``None``.
    prefix:
        String prepended to every metric key, e.g. ``"val/"`` or ``"test/"``.
    """
    _log = get_logger("metrics")
    step_tag = f"step={step} | " if step is not None else ""
    for key, value in metrics_dict.items():
        metric_name = f"{prefix}{key}" if prefix else key
        if isinstance(value, float):
            _log.info("{}metric={} value={:.6f}", step_tag, metric_name, value)
        else:
            _log.info("{}metric={} value={}", step_tag, metric_name, value)
