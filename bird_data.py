import pandas as pd
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

SAMPLE_RATE = 32000
DURATION = 5
N_MELS = 128
IMAGE_SIZE = 224


class BirdDataset(Dataset):

    def __init__(self, csv_file, audio_dir, label_to_idx):
        self.df = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        # Audio path
        audio_path = f"{self.audio_dir}/{row['filename']}"

        # Load audio
        audio, sr = librosa.load(
            audio_path,
            sr=SAMPLE_RATE,
            duration=DURATION
        )

        # Make audio exactly 5 seconds
        target_length = SAMPLE_RATE * DURATION

        if len(audio) < target_length:
            audio = np.pad(
                audio,
                (0, target_length - len(audio))
            )
        else:
            audio = audio[:target_length]

        # Mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_mels=N_MELS,
            n_fft=1024,
            hop_length=512
        )

        # Convert to dB
        mel = librosa.power_to_db(
            mel,
            ref=np.max
        )

        # Convert to tensor
        mel = torch.tensor(
            mel,
            dtype=torch.float32
        )

        # Add channel dimension
        mel = mel.unsqueeze(0)

        # Convert 1 channel → 3 channels for EfficientNet
        mel = mel.repeat(3, 1, 1)

        # Resize to 224 x 224
        mel = F.interpolate(
            mel.unsqueeze(0),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        # Normalize
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        # Label
        label = self.label_to_idx[row["primary_label"]]

        return mel, torch.tensor(label, dtype=torch.long)


# Load training metadata
train_df = pd.read_csv("train_split.csv")

# Create species → number mapping
species = sorted(train_df["primary_label"].unique())

label_to_idx = {
    label: idx
    for idx, label in enumerate(species)
}

# Create dataset
dataset = BirdDataset(
    "train_split.csv",
    "birdclef-2024/train_audio",
    label_to_idx
)

# Create DataLoader
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

# Test one batch
images, labels = next(iter(loader))

print("Batch shape:", images.shape)
print("Labels shape:", labels.shape)
print("Number of classes:", len(label_to_idx))
print("Labels:", labels)    