# Customer Churn Prediction — ANN

Predicts whether a bank customer will churn, using a multi-layer Artificial
Neural Network trained on the "Churn Modelling" dataset (10,000 customers,
demographic + transactional features).

## Architecture

```
Churn_Modelling.csv (10,000 rows)
    │  drop RowNumber/CustomerId/Surname (identifiers, not predictive)
    ▼
train/test split (80/20, stratified on Exited)
    │
    ▼
ColumnTransformer — FIT ON TRAIN ONLY
    ├── StandardScaler:   CreditScore, Age, Tenure, Balance, NumOfProducts, EstimatedSalary
    ├── OneHotEncoder:    Geography, Gender
    └── passthrough:      HasCrCard, IsActiveMember (already 0/1)
    │
    ▼
ANN: Dense(16, relu) -> Dense(8, relu) -> Dense(1, sigmoid)
    │  Adam optimizer, binary_crossentropy loss
    │  EarlyStopping(monitor=val_loss, patience=10, restore_best_weights=True)
    ▼
saved_model/churn_ann.keras + saved_model/preprocessor.pkl
    │
    ▼
Streamlit app (app.py) — real-time churn probability + visual analytics
```

## Why each piece is there

- **Dropping RowNumber/CustomerId/Surname**: these are identifiers with no
  predictive signal — including them would let the model "memorize" rows
  instead of learning the actual churn pattern.
- **Fit the preprocessor on the training split only**: StandardScaler's
  mean/std and OneHotEncoder's category list must come from training data
  only. Fitting on the full dataset (or refitting per prediction) leaks
  information from data the model shouldn't have access to at train time —
  this is the resume's "prevent data leakage between training and inference."
  The fitted preprocessor is saved once and only ever `.transform()`-ed
  against new data afterward, including every prediction the Streamlit app
  makes.
- **ReLU hidden layers, sigmoid output**: ReLU avoids vanishing gradients in
  hidden layers; sigmoid squashes the final layer into a (0, 1) probability
  for binary classification — this probability is literally what the app
  displays as "churn probability."
- **EarlyStopping**: stops training once validation loss stops improving for
  `patience` epochs and restores the weights from the best epoch — prevents
  overfitting to the training set past the point where it generalizes well,
  without having to guess a fixed epoch count upfront.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# train the model (writes saved_model/churn_ann.keras + preprocessor.pkl)
python -m src.train

# launch the app
streamlit run app.py
```

## Study notebook

`notebook/churn_ann_study.ipynb` walks through EDA, preprocessing, and
training interactively, importing the same `src/` code the app uses.
