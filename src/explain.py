import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

from features import load_ngram_table

import json
import os

def load_model_and_data(
        features_path="../data/processed/features.csv",
        model_path="../models/classifier.json"
):
    df = pd.read_csv(features_path)
    feature_cols = [c for c in df.columns if c not in ["label", "family"]]
    X = df[feature_cols]
    y = df["label"]

    model = XGBClassifier()
    model.load_model(model_path)

    return model, X, y, df, feature_cols

def compute_shap_values(model, X, sample_size=2000):
    print(f"Computing SHAP values on {sample_size} samples...")
    X_sample = X.sample(sample_size, random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    return shap_values, X_sample

def plot_global(shap_values, output_dir="../outputs"):
    os.makedirs(output_dir, exist_ok=True)

    # Beeswarm
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved shap_beeswarm.png")

    # Bar (mean absolute SHAP)
    plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved shap_bar.png")

# Explain one prediction with SHAP
def explain_domain(domain, model, ngram_table, feature_cols, tld_freq_path="../models/tld_freq.json", threshold=0.35):
    import sys
    sys.path.insert(0, ".")
    from features import extract_features

    with open(tld_freq_path) as f:
        tld_counts = json.load(f)

    feats = extract_features(domain, ngram_table)
    domain_tld = feats.pop("tld")
    feats["tld_freq"] = tld_counts.get(domain_tld, 0)

    X_single = pd.DataFrame([feats])[feature_cols]
    proba = model.predict_proba(X_single)[0][1]
    verdict = "DGA" if proba >= threshold else "LEGIT"

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer(X_single)

    print(f"\nDomain:   {domain}")
    print(f"Verdict: {verdict} ({proba:.1%} confidence)")
    print(f"\nFeature contributions (+ = toward DGA, - = toward legit):")
    contributions = list(zip(feature_cols, shap_vals.values[0]))
    for feat, val in sorted(contributions, key=lambda x: -abs(x[1])):
        direction = "-> DGA " if val > 0 else "-> legit "
        print(f"    {feat:<25} {direction}  {val:+.4f}")

if __name__ == "__main__":
    model, X, y, df, feature_cols = load_model_and_data()

    shap_values, X_sample = compute_shap_values(model, X)
    plot_global(shap_values)

    ngram_table = load_ngram_table("../models/ngram_table.json")

    test_domains = [
        "ymvpdgmplptkaommi.tw",           # gibberish
        "horsepressurecalculatedeliver.com", # matsnu-style
        "cmfggfkte.com.sv",                # conficker
        "fmspuxrs7i8sfi8osd.biz",         # digit-heavy
        "google.com",                      # clearly legit
        "wideworld-sports.me",             # legit with hyphen
    ]

    for domain in test_domains:
        explain_domain(domain, model, ngram_table, feature_cols)
        print("-" * 55)