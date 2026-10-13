#!/usr/bin/env python3
"""Train the supervised ML predictor (3 HistGradientBoosting models)."""
import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from src.models.ml_predictor import MLPredictor, TARGET_NAMES
from src.utils import io as uio

log = logging.getLogger("train_ml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/ml_data/ml_dataset.npz")
    ap.add_argument("--out", default="models/ml_predictor.joblib")
    ap.add_argument("--metrics-out", default="results/ml_data/ml_metrics.json")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    data = np.load(args.data, allow_pickle=False)
    X, Y, split = data["X"], data["Y"], data["split"]
    tr = split == 0
    va = split == 1
    log.info("rows: %d train / %d val (scenario-disjoint)", tr.sum(), va.sum())

    pred = MLPredictor()
    pred.fit(X[tr], {name: Y[tr, i] for i, name in enumerate(TARGET_NAMES)})
    val_metrics = pred.evaluate(X[va], {name: Y[va, i] for i, name in enumerate(TARGET_NAMES)})
    train_metrics = pred.evaluate(X[tr], {name: Y[tr, i] for i, name in enumerate(TARGET_NAMES)})

    log.info("validation metrics:")
    for name, m in val_metrics.items():
        log.info("  %-10s MAE=%8.4f RMSE=%8.4f R2=%6.3f (mean target %.4f, mean pred %.4f)",
                 name, m["mae"], m["rmse"], m["r2"], m["mean_target"], m["mean_pred"])

    pred.save(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(args.metrics_out)), exist_ok=True)
    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": os.path.abspath(args.data),
            "n_train": int(tr.sum()), "n_val": int(va.sum()),
            "validation": val_metrics, "train": train_metrics,
            "model_size_bytes": pred.model_size_bytes(),
            "model_path": os.path.abspath(args.out),
        }, f, indent=1)
    log.info("saved model %s (%d bytes) and metrics %s",
             args.out, pred.model_size_bytes(), args.metrics_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
