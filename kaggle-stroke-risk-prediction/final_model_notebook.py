# %%
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

# %%
script_dir = Path(__file__).resolve().parent
data_dir = script_dir.parent / "data" / "kaggle-stroke-risk-prediction"

train_df = pd.read_csv(data_dir / "train.csv")

train_df['gender'] = train_df['gender'].map({'Male': 0, 'Female': 1})
train_df['ever_married'] = train_df['ever_married'].map({'No': 0, 'Yes': 1})
train_df['Residence_type'] = train_df['Residence_type'].map({'Rural': 0, 'Urban': 1})
train_df['age_x_hypertension'] = train_df['age'] * train_df['hypertension']
train_df['age_x_heart_disease'] = train_df['age'] * train_df['heart_disease']
train_df['age_x_bmi'] = train_df['age'] * train_df['bmi']
train_df['glucose_x_bmi'] = train_df['avg_glucose_level'] * train_df['bmi']

# %%
X_train, X_val, y_train, y_val = train_test_split(
    train_df.drop(columns=["id", "stroke"]),
    train_df["stroke"],
    test_size=0.2,
    random_state=42,
    stratify=train_df["stroke"],
)
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)

# %%
cat_features = train_df.select_dtypes(include='str').columns.tolist()
print("Categorical features:", cat_features)

# %%
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    eval_metric="AUC",
    random_seed=42,
    verbose=100,
    cat_features=cat_features,
    class_weights=[1, 45.5]
)
model.fit(
    X=X_train,
    y=y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=50
)

# %%
y_pred_proba = model.predict_proba(X_train)[:, 1]
roc_auc = roc_auc_score(y_train, y_pred_proba)
print(f"Training ROC AUC: {roc_auc:.4f}")

y_pred_proba = model.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, y_pred_proba)
print(f"Validation ROC AUC: {roc_auc:.4f}")

# Training ROC AUC: 0.9267
# Validation ROC AUC: 0.8811

# %%
