"""Central constants: column roles, file paths, and ANN hyperparameters,
so the training script and the Streamlit app never disagree about them.
"""
import os

DATA_PATH = os.path.join("data", "Churn_Modelling.csv")

# RowNumber/CustomerId/Surname are identifiers, not predictive features —
# dropped before preprocessing. Exited is the target.
DROP_COLUMNS = ["RowNumber", "CustomerId", "Surname"]
TARGET_COLUMN = "Exited"

NUMERIC_FEATURES = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
CATEGORICAL_FEATURES = ["Geography", "Gender"]
# Already binary 0/1 in the source data — passed through unscaled, no encoding needed.
PASSTHROUGH_FEATURES = ["HasCrCard", "IsActiveMember"]

MODEL_PATH = os.path.join("saved_model", "churn_ann.keras")
PREPROCESSOR_PATH = os.path.join("saved_model", "preprocessor.pkl")

RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 32
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
