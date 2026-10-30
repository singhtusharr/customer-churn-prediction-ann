"""Reusable preprocessing: a single fitted ColumnTransformer saved to disk and
reused at inference, instead of re-deriving scaling statistics from whatever
data happens to be on hand at prediction time.

Why this matters (the resume's "prevent data leakage between training and
inference"): StandardScaler's mean/std and OneHotEncoder's category list must
come ONLY from the training set. If you fit a new scaler on the single row a
user submits in the Streamlit app, that row trivially scales to itself (mean
subtracted from itself = 0) — nonsense, and a textbook example of leakage.
Fitting once on training data, saving that exact fitted object, and only ever
calling .transform() (never .fit() again) on new data is what prevents this.
"""
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, PASSTHROUGH_FEATURES


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("passthrough", "passthrough", PASSTHROUGH_FEATURES),
        ]
    )


def save_preprocessor(preprocessor: ColumnTransformer, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(preprocessor, f)


def load_preprocessor(path: str) -> ColumnTransformer:
    with open(path, "rb") as f:
        return pickle.load(f)


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Human-readable names for the transformed columns — used by the
    Streamlit app's feature-contribution chart."""
    return list(preprocessor.get_feature_names_out())
