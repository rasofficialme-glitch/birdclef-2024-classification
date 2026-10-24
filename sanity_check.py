import pandas as pd
import os

# Load metadata
df = pd.read_csv("birdclef-2024/train_metadata.csv")

audio_folder = "birdclef-2024/train_audio"

# 1. Check missing audio files
missing_files = []

for filename in df["filename"]:
    path = os.path.join(audio_folder, filename)

    if not os.path.exists(path):
        missing_files.append(filename)

print("Total recordings:", len(df))
print("Missing audio files:", len(missing_files))

# 2. Check missing labels
missing_labels = df["primary_label"].isna().sum()

print("Missing labels:", missing_labels)

# 3. Check duplicate filenames
duplicates = df["filename"].duplicated().sum()

print("Duplicate filenames:", duplicates)

# 4. Check species
species_count = df["primary_label"].nunique()

print("Number of species:", species_count)

# 5. Check recordings per species
species_counts = df["primary_label"].value_counts()

print("\nSpecies with fewest recordings:")
print(species_counts.tail(10))

print("\nSpecies with most recordings:")
print(species_counts.head(10))