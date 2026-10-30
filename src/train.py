"""End-to-end training: load -> split -> preprocess (fit on train only) ->
train ANN with EarlyStopping -> evaluate -> save model + preprocessor.

Backpropagation isn't a separate step we call explicitly — it's what
model.fit() does internally every batch: compute the loss, propagate
gradients backward through every layer via the chain rule, and update each
layer's weights via the optimizer. It's named here because understanding
*why* training works (not just calling .fit()) is exactly what an
interviewer probing this bullet point will ask about.
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras

from src.config import (
    DATA_PATH,
    DROP_COLUMNS,
    EARLY_STOPPING_PATIENCE,
    BATCH_SIZE,
    MAX_EPOCHS,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    VALIDATION_SPLIT,
)
from src.model import build_ann
from src.preprocessing import build_preprocessor, save_preprocessor


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df.drop(columns=DROP_COLUMNS)


def train() -> dict:
    df = load_data()
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor()
    # fit_transform on TRAIN ONLY — this is the data-leakage boundary.
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    model = build_ann(input_dim=X_train_processed.shape[1])

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train_processed,
        y_train,
        validation_split=VALIDATION_SPLIT,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=2,
    )

    test_loss, test_accuracy = model.evaluate(X_test_processed, y_test, verbose=0)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    save_preprocessor(preprocessor, PREPROCESSOR_PATH)

    val_losses = history.history["val_loss"]
    first_val_loss = val_losses[0]
    best_val_loss = min(val_losses)
    val_loss_reduction_pct = (first_val_loss - best_val_loss) / first_val_loss * 100

    return {
        "epochs_trained": len(val_losses),
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "first_val_loss": first_val_loss,
        "best_val_loss": best_val_loss,
        "val_loss_reduction_pct": val_loss_reduction_pct,
    }


if __name__ == "__main__":
    metrics = train()
    print(f"Trained {metrics['epochs_trained']} epochs (EarlyStopping-controlled)")
    print(f"Test accuracy: {metrics['test_accuracy']*100:.2f}%")
    print(f"Test loss: {metrics['test_loss']:.4f}")
    print(
        f"Validation loss reduced {metrics['val_loss_reduction_pct']:.1f}% "
        f"({metrics['first_val_loss']:.4f} -> {metrics['best_val_loss']:.4f})"
    )
