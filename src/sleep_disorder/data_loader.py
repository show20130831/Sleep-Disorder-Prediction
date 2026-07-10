"""Load the raw Sleep Health and Lifestyle CSV into a DataFrame."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sleep_disorder import config
from sleep_disorder.utils import get_logger

logger = get_logger(__name__)


def load_raw_data(path: Path | None = None) -> pd.DataFrame:
    """Read the raw CSV.

    ``keep_default_na=False`` is intentional: the "Sleep Disorder" column uses the
    literal string "None" as a valid category (no disorder), not a missing value.
    Letting pandas' default NA sniffing run would silently turn that column into NaN.
    """
    path = path if path is not None else config.RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {path}")

    df = pd.read_csv(path, keep_default_na=False)
    logger.info("Loaded raw data: %d rows, %d columns from %s", *df.shape, path)
    return df
