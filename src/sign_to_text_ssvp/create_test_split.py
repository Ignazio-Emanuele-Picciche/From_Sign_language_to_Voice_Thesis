"""
Create test_split.csv from how2sign_test.csv
"""

import pandas as pd
from pathlib import Path

# Read test CSV
test_csv = Path("../../data/raw/test/how2sign_test.csv")
df = pd.read_csv(test_csv, sep="\t")

# Rename columns to match expected format
df_output = pd.DataFrame(
    {
        "video_name": df["SENTENCE_NAME"],
        "caption": df["SENTENCE"],
        "Source collection": "How2Sign",
        "video_id": df["VIDEO_ID"],
        "sentence_id": df["SENTENCE_ID"],
        "start_time": df["START"],
        "end_time": df["END"],
    }
)

# Save to results/how2sign_splits/
output_dir = Path("../../results/how2sign_splits")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "test_split.csv"
df_output.to_csv(output_file, index=False)

print(f"✅ Created {output_file}")
print(f"   Total samples: {len(df_output)}")
