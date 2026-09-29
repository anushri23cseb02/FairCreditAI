"""
FairCredit AI - Phase 3/4 CLI entry point.

Thin wrapper around backend/ml/trainer.py so there is exactly one
training implementation, shared with the POST /train API endpoint
(backend/api/routes_train.py).

Run from the project root (after scripts/prepare_data.py):

    python scripts/train_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.trainer import train_and_persist  # noqa: E402


def main() -> None:
    metadata = train_and_persist(version="v1")
    metrics = metadata["held_out_test_metrics_final_model"]
    print("Training complete.")
    print(f"  Test ROC-AUC (final model): {metrics['roc_auc']}")
    print(f"  Test F1 (final model):      {metrics['f1']}")
    print(f"  Artifact: models/artifacts/model_{metadata['version']}.joblib")


if __name__ == "__main__":
    main()
