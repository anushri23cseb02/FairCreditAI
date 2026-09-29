"""
Regression test: retraining ONE track's model must never wipe the
OTHER track's registry entry. This session found a real bug where the
HMDA trainer overwrote the whole registry.json file instead of merging
into it, silently deleting the credit_card entry the first time
/train (HMDA) was called after /credit-card/train had run.
"""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.credit_card_service import get_credit_card_model_service
from backend.services.model_service import get_model_service

client = TestClient(app)
REGISTRY_PATH = Path(__file__).resolve().parent.parent / "models" / "registry" / "registry.json"
HMDA_PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


@pytest.fixture(scope="module", autouse=True)
def skip_if_either_track_not_trained():
    if not get_model_service().is_ready or not get_credit_card_model_service().is_ready:
        pytest.skip("Both tracks must be trained in this environment for this regression test.")


def _hmda_processed_data_available() -> bool:
    return all(
        (HMDA_PROCESSED_DIR / name).exists()
        for name in ("train.csv", "val.csv", "test.csv")
    )


def test_retraining_hmda_preserves_credit_card_registry_entry():
    if not _hmda_processed_data_available():
        pytest.skip("HMDA processed data (data/processed/{train,val,test}.csv) not available in this environment.")
    before = json.load(open(REGISTRY_PATH))
    assert "credit_card" in before, "credit_card entry missing before retrain -- test setup problem, not the bug under test"

    r = client.post("/train")  # HMDA retrain
    assert r.status_code == 200

    after = json.load(open(REGISTRY_PATH))
    assert "credit_card" in after, "HMDA retrain wiped the credit_card registry entry"
    assert after["credit_card"] == before["credit_card"]


def test_retraining_credit_card_preserves_hmda_registry_entry():
    before = json.load(open(REGISTRY_PATH))
    assert "active_version" in before

    r = client.post("/credit-card/train")
    assert r.status_code == 200

    after = json.load(open(REGISTRY_PATH))
    assert "active_version" in after, "credit-card retrain wiped the HMDA registry entry"
    assert after["versions"] == before["versions"]
