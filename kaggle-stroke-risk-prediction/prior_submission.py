from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

script_dir = Path(__file__).resolve().parent
data_dir = script_dir.parent / "data" / "kaggle-stroke-risk-prediction"


CV_SPLITS = 5
RANDOM_STATE = 42

def main():
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    submission_df = pd.read_csv(data_dir / "sample_submission.csv")

    train_df['ever_married'] = train_df['ever_married'].map({'Yes': 1, 'No': 0})
    train_df['Residence_type'] = train_df['Residence_type'].map({'Urban': 1, 'Rural': 0})

    feature_cols = [col for col in train_df.columns if col not in ['id', 'stroke']]
    target_col = 'stroke'

    X, y = train_df[feature_cols], train_df[target_col]

    object_cols = X.select_dtypes(include='object').columns
    for col in object_cols:
        X[col] = X[col].astype('category')

    # List of categorical feature column names
    categorical_cols = X.select_dtypes(include='category').columns.tolist()

    auc_list = []
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        print(f"\nFold {fold}")

        # Split data
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        # Create CatBoost Pools (handles categorical features)
        train_pool = Pool(X_train, label=y_train, cat_features=categorical_cols)
        val_pool = Pool(X_val, label=y_val, cat_features=categorical_cols)

        # Define and train model
        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.15,
            depth=3,
            subsample=0.8,
            loss_function='Logloss',
            eval_metric='AUC',
            early_stopping_rounds=50,
            verbose=False,
            random_seed=RANDOM_STATE,
            use_best_model=True,
            class_weights=[1, 3.2]
        )

        model.fit(train_pool, eval_set=val_pool)

        # Predict probabilities
        y_prob = model.predict_proba(X_val)[:, 1]

        # AUC score
        auc = roc_auc_score(y_val, y_prob)
        auc_list.append(auc)

        print(f"AUC: {auc:.4f} | Best iteration: {model.get_best_iteration() + 1}")

    print(f"\nAverage AUC across folds: {np.mean(auc_list):.4f}")

    # Create submission
    test_df['ever_married'] = test_df['ever_married'].map({'Yes': 1, 'No': 0})
    test_df['Residence_type'] = test_df['Residence_type'].map({'Urban': 1, 'Rural': 0})

    submission_df['stroke'] = model.predict_proba(test_df[feature_cols])[:, 1]
    submission_df.to_csv(data_dir / "submission.csv", index=False)

    # Public Score: 0.84966
    # Private Score: 0.83405

if __name__ == "__main__":
    main()
