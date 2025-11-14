#!/usr/bin/env python3
"""
Script per verificare il formato dei manifest TSV
"""

import pandas as pd
from pathlib import Path


def check_manifest(manifest_path):
    """Verifica struttura manifest"""
    print(f"\n{'='*60}")
    print(f"Checking: {manifest_path}")
    print(f"{'='*60}")

    if not Path(manifest_path).exists():
        print(f"❌ File not found!")
        return

    # Carica
    df = pd.read_csv(manifest_path, sep="\t")

    print(f"✅ Loaded: {len(df)} rows")
    print(f"\n📊 Columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i}. '{col}'")

    print(f"\n📝 First 3 rows:")
    print(df.head(3).to_string())

    print(f"\n🔍 Sample values:")
    for col in df.columns:
        sample = df[col].iloc[0] if len(df) > 0 else "N/A"
        print(f"   {col}: {sample}")


if __name__ == "__main__":
    # Check train manifest
    check_manifest("manifests/train.tsv")

    # Check val manifest
    check_manifest("manifests/val.tsv")

    # Check test manifest
    check_manifest("manifests/test.tsv")
