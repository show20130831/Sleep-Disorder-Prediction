"""Score a single new record with a trained, persisted pipeline.

Loads a joblib-serialized pipeline from ``outputs/models/`` and applies the
same deterministic cleaning used at training time (blood pressure split, BMI
category merge) before predicting — everything from ``preprocessing.py``
except target encoding, since there's no label at inference time. One-hot
encoding/imputation/SMOTE are already baked into the loaded pipeline.

CLI usage:
    python -m sleep_disorder.inference --model random_forest --input record.json

``record.json`` holds one record with the same raw column names as the
source CSV (see ``examples/sample_record.json``), minus ``Person ID`` and
``Sleep Disorder``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline

from sleep_disorder import config
from sleep_disorder.models import MODEL_KEYS, RANDOM_FOREST_KEY
from sleep_disorder.preprocessing import merge_bmi_categories, split_blood_pressure
from sleep_disorder.utils import configure_logging, get_logger

logger = get_logger(__name__)

MODEL_CHOICES = MODEL_KEYS


def load_model(model_name: str) -> ImbPipeline:
    """Load a trained pipeline saved by ``train.train_all_models``."""
    model_path = config.MODELS_DIR / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run `python -m sleep_disorder.main` first."
        )
    return joblib.load(model_path)


def prepare_record(record: dict[str, Any]) -> pd.DataFrame:
    """Apply the training-time deterministic cleaning steps to one raw record."""
    df = pd.DataFrame([record])
    df = split_blood_pressure(df)
    df = merge_bmi_categories(df)
    return df


def predict_one(pipeline: ImbPipeline, record: dict[str, Any]) -> dict[str, Any]:
    """Predict the sleep disorder class and per-class probabilities for one record."""
    X = prepare_record(record)
    predicted_class = int(pipeline.predict(X)[0])
    probabilities = pipeline.predict_proba(X)[0]
    return {
        "prediction": config.CLASS_NAMES[predicted_class],
        "probabilities": dict(zip(config.CLASS_NAMES, (float(p) for p in probabilities))),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score one record with a trained pipeline.")
    parser.add_argument("--model", choices=MODEL_CHOICES, default=RANDOM_FOREST_KEY)
    parser.add_argument(
        "--input", type=Path, required=True, help="Path to a JSON file holding one record."
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = _parse_args()

    try:
        record = json.loads(args.input.read_text(encoding="utf-8"))
        logger.debug("Loaded record: %s", record)

        pipeline = load_model(args.model)
        result = predict_one(pipeline, record)
    except Exception:
        logger.exception("Prediction failed for model=%s input=%s", args.model, args.input)
        raise

    logger.info("Model: %s | Prediction: %s", args.model, result["prediction"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
