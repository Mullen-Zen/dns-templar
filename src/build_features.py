import pandas as pd
import json
import os
from features import extract_features, load_ngram_table

def build_feature_matrix(
    data_path="../data/raw/domains.csv",
    ngram_path="../models/ngram_table.json",
    output_path="../data/processed/features.csv"
):
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    print("Loading n-gram table...")
    table = load_ngram_table(ngram_path)

    print("Extracting features...")
    feature_rows = []
    for i, row in enumerate(df.itertuples(), 1):
        features = extract_features(str(row.domain), table)
        features["family"] = row.family
        features["label"] = 1 if row.type == "dga" else 0
        feature_rows.append(features)
        if i % 50_000 == 0:
            print(f"    {i:,} / {len(df):,}")
    
    print("Building dataframe")
    fdf = pd.DataFrame(feature_rows)

    print("Encoding TLD...")
    tld_counts = fdf["tld"].value_counts().to_dict()
    fdf["tld_freq"] = fdf["tld"].map(tld_counts)
    fdf = fdf.drop(columns=["tld"])

    tld_freq_path = "../models/tld_freq.json"
    with open(tld_freq_path, "w") as f:
        json.dump(tld_counts, f)
    print(f"Saved TLD frequency map ({len(tld_counts)} TLDs)")

    os.makedirs("data/processed", exist_ok=True)
    fdf.to_csv(output_path, index=False)
    print(f"Saved {len(fdf):,} rows to {output_path}")
    print(f"\nFeature columns: {[c for c in fdf.columns if c not in ['label','family']]}")
    print(f"\nLabel distribution:\n{fdf['label'].value_counts()}")

if __name__ == "__main__":
    build_feature_matrix()