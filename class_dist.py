import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("birdclef-2024/train_metadata.csv")

counts = df["primary_label"].value_counts()

plt.figure(figsize=(15, 5))
counts.plot(kind="bar")

plt.xlabel("Bird Species")
plt.ylabel("Number of Recordings")
plt.title("BirdCLEF 2024 Class Distribution")

plt.tight_layout()
plt.show()