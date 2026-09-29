"""
Builds the sklearn preprocessing pipeline used for BOTH training and
inference. Persisting this object together with the trained model
(see train_model.py) guarantees the API never re-derives transformations
from scratch and can never silently drift from what the model saw during
training.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from backend.ml.feature_config import ALL_CATEGORICAL_FEATURES, ALL_NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # handle_unknown="ignore" so a category never seen during
            # training (e.g. a new loan_type at inference time) does not
            # crash the API — it is encoded as all-zeros instead.
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, ALL_NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, ALL_CATEGORICAL_FEATURES),
        ]
    )
