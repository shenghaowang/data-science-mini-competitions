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
print(train_df.shape)

train_df.head()
# %%
train_df.info()
# %%
train_df['stroke'].value_counts()
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
)
model.fit(
    X=X_train,
    y=y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=50
)
# %%
y_pred_proba = model.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, y_pred_proba)
print(f"Validation ROC AUC: {roc_auc:.4f}")

# Validation ROC AUC: 0.8555
# %%
# Create submission file
test_df = pd.read_csv(data_dir / "test.csv")
test_X = test_df.drop(columns=["id"])

submission_df = pd.DataFrame({
    "id": test_df["id"],
    "stroke": model.predict_proba(test_X)[:, 1]
})
submission_df.to_csv(data_dir / "submission.csv", index=False)
