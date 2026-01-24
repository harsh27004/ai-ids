import pandas as pd
import numpy as np
import os
from tqdm import tqdm

RAW_DATA_DIR = "data/raw/cicids2018/"
OUTPUT_FILE = "data/processed/cicids_2018_binary_clean.csv"

os.makedirs("data/processed", exist_ok=True)

csv_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".csv")]

first_write = True   # controls CSV header writing

for file in tqdm(csv_files, desc="Processing CSV files"):
    file_path = os.path.join(RAW_DATA_DIR, file)

    # Read CSV safely
    df = pd.read_csv(file_path, low_memory=False)

    # Fix column name spaces
    df.columns = df.columns.str.strip()

    # Replace infinity values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with missing labels
    df.dropna(subset=["Label"], inplace=True)

    # Binary label encoding
    df["Label"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)

    # Drop remaining NaNs
    df.dropna(inplace=True)

    # Append to output CSV (incremental write)
    df.to_csv(
        OUTPUT_FILE,
        mode="a",
        header=first_write,
        index=False
    )

    first_write = False  # header only once

print("✅ Preprocessing complete!")
print(f"📁 Final dataset saved as: {OUTPUT_FILE}")
