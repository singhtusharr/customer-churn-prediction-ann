"""The ANN architecture: a multi-layer (2 hidden layers) feed-forward
network for binary classification (churn / no churn).

Architecture choices:
- Hidden layer sizes (16, 8) roughly follow the common rule-of-thumb of
  shrinking layer width as you go deeper, sized relative to the input
  dimensionality (~11 features after preprocessing) rather than being
  arbitrary — too wide risks overfitting on a 10k-row dataset, too narrow
  underfits.
- ReLU for hidden layers: avoids the vanishing-gradient problem that
  sigmoid/tanh have in deeper networks, and is cheap to compute.
- Sigmoid output + 1 unit: squashes the final linear output into a (0, 1)
  probability for binary classification — this IS the "churn probability"
  the Streamlit app displays.
- Adam optimizer: adapts the learning rate per-parameter using running
  estimates of gradient mean/variance (combines momentum + RMSProp ideas) —
  converges faster and more reliably than plain SGD without hand-tuning a
  learning rate schedule.
- binary_crossentropy loss: the standard loss for binary classification with
  a sigmoid output — penalizes confident-wrong predictions heavily.
"""
from tensorflow import keras
from tensorflow.keras import layers


def build_ann(input_dim: int) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(16, activation="relu"),
            layers.Dense(8, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
