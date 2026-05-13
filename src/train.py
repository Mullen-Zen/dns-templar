import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import json

def train(features_path="../data/processed/features.csv"):
    print("Loading features...")
    df = pd.read_csv(features_path)

    feature_cols = [c for c in df.columns if c not in ["label", "family"]]
    X = df[feature_cols]
    y = df["label"]

    print(f"Features: {feature_cols}")
    print(f"Dataset size: {len(df):,} rows")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    print("\nTraining XGBoost...")
    model = XGBClassifier(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        early_stopping_rounds=50,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )

    print("\nEvaluating...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["legit","dga"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== Threshold Analysis ===")
    y_proba = model.predict_proba(X_test)[:, 1]

    for threshold in [0.5, 0.4, 0.35, 0.3, 0.25]:
        y_thresh = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_thresh).ravel()
        dga_recall = tp / (tp + fn)
        dga_precision = tp / (tp + fp)
        print(f"threshold={threshold:.2f} | "
              f"DGA recall={dga_recall:.3f} | "
              f"DGA precision={dga_precision:.3f} | "
              f"false negatives={fn:,} | "
              f"false positives={fp:,}")

    print("\nFeature importances:")
    importances = dict(zip(feature_cols, model.feature_importances_))
    for feat, score in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"    {feat:<25} {score:.4f}")

    model.save_model("../models/classifier.json")
    print("\nModel saved to models/classifier.json")

    # Threshold of confidence for classification as DGA.
    # Refer to threshold analysis above to select as appropriate
    THRESHOLD = 0.35

    with open("../models/threshold.json", "w") as f:
        json.dump({"threshold": THRESHOLD}, f)
    print(f"\nThreshold {THRESHOLD} saved to models/threshold.json")

if __name__ == "__main__":
    train()