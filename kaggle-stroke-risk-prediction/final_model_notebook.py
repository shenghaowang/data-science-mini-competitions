# %%
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

# %%
script_dir = Path(__file__).resolve().parent
data_dir = script_dir.parent / "data" / "kaggle-stroke-risk-prediction"

train_df = pd.read_csv(data_dir / "train.csv")
smoking_order = {
    'never smoked': 0,
    'Unknown': 1,
    'formerly smoked': 2,
    'smokes': 3,
}

# train_df['smoking_status'] = train_df['smoking_status'].map(smoking_order)
train_df['gender'] = train_df['gender'].map({'Male': 0, 'Female': 1})
train_df['ever_married'] = train_df['ever_married'].map({'No': 0, 'Yes': 1})
train_df['Residence_type'] = train_df['Residence_type'].map({'Rural': 0, 'Urban': 1})
train_df['age_x_hypertension'] = train_df['age'] * train_df['hypertension']
train_df['age_x_heart_disease'] = train_df['age'] * train_df['heart_disease']
train_df['age_x_bmi'] = train_df['age'] * train_df['bmi']
train_df['glucose_x_bmi'] = train_df['avg_glucose_level'] * train_df['bmi']

# %%
# Train CatBoost model with 5-fold cross-validation and early stopping

X = train_df.drop(columns=["id", "stroke"])
y = train_df["stroke"]
cat_features = X.select_dtypes(include='str').columns.tolist()
print("Categorical features:", cat_features)

test_df = pd.read_csv(data_dir / "test.csv")
# test_df['smoking_status'] = test_df['smoking_status'].map(smoking_order)
test_df['gender'] = test_df['gender'].map({'Male': 0, 'Female': 1})
test_df['ever_married'] = test_df['ever_married'].map({'No': 0, 'Yes': 1})
test_df['Residence_type'] = test_df['Residence_type'].map({'Rural': 0, 'Urban': 1})
test_df['age_x_hypertension'] = test_df['age'] * test_df['hypertension']
test_df['age_x_heart_disease'] = test_df['age'] * test_df['heart_disease']
test_df['age_x_bmi'] = test_df['age'] * test_df['bmi']
test_df['glucose_x_bmi'] = test_df['avg_glucose_level'] * test_df['bmi']
test_X = test_df.drop(columns=["id"])

# %%
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test_X))
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n--- Fold {fold + 1} ---")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

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

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(test_X)[:, 1] / skf.n_splits

    fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
    fold_scores.append(fold_auc)
    print(f"Fold {fold + 1} AUC: {fold_auc:.4f}")

oof_auc = roc_auc_score(y, oof_preds)
print(f"\nOOF AUC: {oof_auc:.4f}")
print(f"Mean Fold AUC: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")

# %%
# Create submission file
submission_df = pd.DataFrame({
    "id": test_df["id"],
    "stroke": test_preds
})

submission_df.head()

# %%
submission_df.to_csv(data_dir / "submission.csv", index=False)

# %%
