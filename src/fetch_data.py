import requests
import pandas as pd
from io import BytesIO
from zipfile import ZipFile

def fetch_dataset(output_path="../data/raw/domains.csv"):
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

def fetch_whitelist(output_path="../models/whitelist.txt", limit=50_000):
    print("Fetching Tranco whitelist...")
    url = "https://tranco-list.eu/download_daily/6G89X"
    r = requests.get(url)
    z = ZipFile(BytesIO(r.content))
    filename = z.namelist()[0]

    domains = []
    with z.open(filename) as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            parts = line.decode().strip().split(",")
            if len(parts) == 2:
                domains.append(parts[1].lower())
    
    with open(output_path, "w") as out:
        out.write("\n".join(domains))
    print(f"Saved {len(domains)} whitelisted domains to {output_path}")

if __name__ == "__main__":
    fetch_dataset()
    fetch_whitelist()