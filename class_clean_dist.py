import pandas as pd
import matplotlib.pyplot as plt

# Load metadata
df = pd.read_csv("birdclef-2024/train_metadata.csv")

# Count recordings per species
counts = df["primary_label"].value_counts()

# Get top 20 and bottom 20
top_20 = counts.head(20)
bottom_20 = counts.tail(20)

# Combine them
selected = pd.concat([top_20, bottom_20])

# Sort for a clean horizontal chart
selected = selected.sort_values()

# Plot
plt.figure(figsize=(12, 10))

selected.plot(kind="barh")

plt.xlabel("Number of Recordings")
plt.ylabel("Bird Species")
plt.title("BirdCLEF 2024: Top 20 and Bottom 20 Species by Recording Count")

plt.tight_layout()

# Save the figure
plt.savefig("class_distribution_top_bottom_20.png", dpi=300)

plt.show()