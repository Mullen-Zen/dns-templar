import requests
import pandas as pd

def fetch_dataset(output_path="data/raw/domains.csv"):
    print("Fetching dataset...")
    url = "https://raw.githubusercontent.com/chrmor/DGA_domains_dataset/master/dga_domains_full.csv"
    r = requests.get(url)
    
    rows = []
    for line in r.text.strip().split("\n"):
        parts = line.strip().split(",")
        if len(parts) == 3:
            rows.append({
                "type": parts[0],
                "family": parts[1],
                "domain": parts[2]
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    
    print(f"Total rows: {len(df)}")
    print(f"Legit domains: {len(df[df['type'] == 'legit'])}")
    print(f"DGA domains:   {len(df[df['type'] == 'dga'])}")
    print(f"\nDGA families:\n{df[df['type'] == 'dga']['family'].value_counts()}")

if __name__ == "__main__":
    fetch_dataset()