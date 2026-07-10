"""Shared, side-effect-light helpers used across pipeline stages.

Keeps I/O boilerplate (logging setup, directory creation, figure/table saving)
out of the analysis modules so each of those can focus on a single responsibility.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    Deliberately does not attach a handler or set a level: per Python logging
    convention, library code should only name its loggers and let the
    application decide how (or whether) to display them. CLI entry points
    (``main.main``, ``inference.main``) call :func:`configure_logging` once to
    actually turn output on.
    """
    return logging.getLogger(name)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging output. Call once, from a CLI entry point only."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )


def ensure_dir(path: Path) -> Path:
    """Create `path` (and parents) if it doesn't exist yet; return it for chaining."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig: plt.Figure, path: Path, *, dpi: int = 150) -> None:
    """Save a matplotlib figure to `path`, creating parent dirs, then close it."""
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_table(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    """Save a DataFrame to CSV, creating parent dirs first."""
    ensure_dir(path.parent)
    df.to_csv(path, index=index)


def save_json(data: dict[str, Any], path: Path) -> None:
    """Save a dict as pretty-printed JSON, creating parent dirs first."""
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
