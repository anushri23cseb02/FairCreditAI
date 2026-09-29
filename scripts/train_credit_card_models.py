"""
Track A (credit-card default) CLI training entry point. Thin wrapper
around backend/ml/credit_card/trainer.py -- same implementation used by
POST /credit-card/train.

Run from the project root (after scripts/prepare_credit_card_data.py):
    python scripts/train_credit_card_models.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.credit_card.trainer import train_and_compare  # noqa: E402


def main() -> None:
    metadata = train_and_compare()
    print("Training complete for all 3 models.")
    print(f"  Selected model: {metadata['selected_model']}")
    print(f"  Rationale: {metadata['selection_rationale']}")
    for name, s in metadata["selection_scores"].items():
        print(f"  {name}: composite={s['composite_score']} roc_auc={s['roc_auc']} f1={s['f1']}")


if __name__ == "__main__":
    main()
