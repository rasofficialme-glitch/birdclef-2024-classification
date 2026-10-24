import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

# Load metadata
df = pd.read_csv("birdclef-2024/train_metadata.csv")

# Select 6 different species
samples = df.groupby("primary_label").first().head(6)

fig, axes = plt.subplots(3, 2, figsize=(14, 10))
axes = axes.flatten()

for i, (_, row) in enumerate(samples.iterrows()):

    # Audio path
    audio_path = os.path.join(
        "birdclef-2024",
        "train_audio",
        row["filename"]
    )

    # Load 5 seconds of audio
    y, sr = librosa.load(audio_path, sr=32000, duration=5)

    # Create Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=1024,
        hop_length=512,
        n_mels=128
    )

    # Convert to dB
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Display
    librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=512,
        x_axis="time",
        y_axis="mel",
        ax=axes[i]
    )

    axes[i].set_title(row.name)

fig.suptitle(
    "Log-Mel Spectrograms of Sample Bird Recordings",
    fontsize=16
)

plt.tight_layout()

# Save for presentation
plt.savefig(
    "spectrogram_sanity_check.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()