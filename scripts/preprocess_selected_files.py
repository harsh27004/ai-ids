import pandas as pd
import numpy as np
import os

RAW_DATA_DIR = "data/raw/cicids2018/"
FILES_TO_USE = [
    "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv"
]

OUTPUT_FILE = "data/processed/cicids2018_selected_binary.csv"

dfs = []
all_columns = set()

print("🔄 Reading files...")

# ---- Load & clean ----
for file in FILES_TO_USE:
    path = os.path.join(RAW_DATA_DIR, file)
    print(f"📥 Loading {file}")

    df = pd.read_csv(
        path,
        low_memory=False,
        on_bad_lines="skip"
    )

    # Clean columns
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Replace inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Binary label
    df.dropna(subset=["Label"], inplace=True)
    df["Label"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)

    # Keep numeric features + Label
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "Label" not in numeric_cols:
        numeric_cols.append("Label")

    df = df[numeric_cols]

    all_columns.update(df.columns)
    dfs.append(df)

# ---- Align using UNION (CRITICAL FIX) ----
all_columns = sorted(all_columns)

aligned_dfs = []
for df in dfs:
    aligned = df.reindex(columns=all_columns, fill_value=0)
    aligned_dfs.append(aligned)

# ---- Merge ----
final_df = pd.concat(aligned_dfs, ignore_index=True)
final_df.drop_duplicates(inplace=True)

# ---- Save ----
os.makedirs("data/processed", exist_ok=True)
final_df.to_csv(OUTPUT_FILE, index=False)

print("✅ Dataset READY")
print("📁 Saved:", OUTPUT_FILE)
print("📊 Shape:", final_df.shape)
print("📊 Label distribution:")
print(final_df["Label"].value_counts())
