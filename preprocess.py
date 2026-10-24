import librosa
import numpy as np
import matplotlib.pyplot as plt

# Audio settings
SAMPLE_RATE = 32000
DURATION = 5
N_MELS = 128

# Test audio file
audio_path = "birdclef-2024/train_audio/asbfly/XC134896.ogg"

# Load 5 seconds of audio
audio, sr = librosa.load(
    audio_path,
    sr=SAMPLE_RATE,
    duration=DURATION
)

# Make sure every clip is exactly 5 seconds
target_length = SAMPLE_RATE * DURATION

if len(audio) < target_length:
    audio = np.pad(audio, (0, target_length - len(audio)))
else:
    audio = audio[:target_length]

# Create Mel spectrogram
mel = librosa.feature.melspectrogram(
    y=audio,
    sr=SAMPLE_RATE,
    n_mels=N_MELS,
    n_fft=1024,
    hop_length=512
)

# Convert to decibels
mel_db = librosa.power_to_db(mel, ref=np.max)

print("Audio shape:", audio.shape)
print("Mel spectrogram shape:", mel_db.shape)
print("Sample rate:", sr)

# Display spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(
    mel_db,
    sr=sr,
    hop_length=512,
    x_axis="time",
    y_axis="mel"
)

plt.colorbar(format="%+2.0f dB")
plt.title("BirdCLEF 2024 - Mel Spectrogram")
plt.tight_layout()
plt.show()