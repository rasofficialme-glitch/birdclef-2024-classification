import pandas as pd
from sklearn.model_selection import train_test_split

# Load metadata
df = pd.read_csv("birdclef-2024/train_metadata.csv")

# Split 80% train, 20% validation
train_df, val_df = train_test_split(
    df,
    test_size=0.3,
    stratify=df["primary_label"],
    random_state=42
)

# Save the splits
train_df.to_csv("train_split.csv", index=False)
val_df.to_csv("val_split.csv", index=False)

print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))
print("Training species:", train_df["primary_label"].nunique())
print("Validation species:", val_df["primary_label"].nunique())