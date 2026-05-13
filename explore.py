import pandas as pd

df = pd.read_csv("data/raw/domains.csv")

# Strip TLD so we're analyzing just the hostname part
df["hostname"] = df["domain"].apply(lambda x: x.rsplit(".", 1)[0])
df["length"] = df["hostname"].apply(len)

print("=== LEGIT EXAMPLES ===")
print(df[df["type"] == "legit"]["domain"].sample(10, random_state=42).tolist())

print("\n=== DGA EXAMPLES (mixed families) ===")
print(df[df["type"] == "dga"]["domain"].sample(10, random_state=42).tolist())

print("\n=== AVERAGE HOSTNAME LENGTH ===")
print(df.groupby("type")["length"].mean().round(2))

print("\n=== LENGTH DISTRIBUTION (percentiles) ===")
print(df.groupby("type")["length"].describe().round(2))

print("\n=== ONE FAMILY UP CLOSE: conficker ===")
print(df[df["family"] == "conficker"]["domain"].sample(10, random_state=42).tolist())

print("\n=== ONE FAMILY UP CLOSE: matsnu ===")
print(df[df["family"] == "matsnu"]["domain"].sample(10, random_state=42).tolist())