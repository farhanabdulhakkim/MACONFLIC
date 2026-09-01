"""
preprocess.py - Audio loading & Mel-Spectrogram generator for Elephant Audio Classification project.
Loads dataset samples directly from manifest.csv using recording-level splits.
"""

import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Optional imports with fallbacks
try:
    import torchaudio
    import torchaudio.transforms as T
    HAS_TORCHAUDIO = True
except (ImportError, OSError):
    HAS_TORCHAUDIO = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


class AudioPreprocessor:
    """Handles audio loading, resampling, padding/cropping, and Mel-Spectrogram conversion."""
    def __init__(self, sample_rate=22050, n_mels=64, n_fft=1024, hop_length=512, target_duration=3.0):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_length = int(sample_rate * target_duration)

    def load_audio(self, file_path):
        """Loads audio file and returns tensor of shape (1, num_samples) at target sample rate."""
        if HAS_TORCHAUDIO:
            waveform, sr = torchaudio.load(file_path)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            return waveform
        elif HAS_LIBROSA:
            y, _ = librosa.load(file_path, sr=self.sample_rate, mono=True)
            return torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        else:
            raise RuntimeError("Neither torchaudio nor librosa is available to load audio files.")

    def pad_crop(self, waveform):
        """Pads or crops waveform to fixed target duration."""
        num_samples = waveform.shape[-1]
        if num_samples < self.target_length:
            pad_len = self.target_length - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        elif num_samples > self.target_length:
            waveform = waveform[:, :self.target_length]
        return waveform

    def compute_mel_spectrogram(self, waveform):
        """Converts raw waveform into log-mel spectrogram tensor (1, n_mels, time_steps)."""
        if HAS_TORCHAUDIO:
            mel_transform = T.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels
            )
            mel_spec = mel_transform(waveform)
            log_mel = torch.log(mel_spec + 1e-6)
            return log_mel
        elif HAS_LIBROSA:
            y = waveform.squeeze().numpy()
            mel = librosa.feature.melspectrogram(
                y=y, sr=self.sample_rate, n_fft=self.n_fft,
                hop_length=self.hop_length, n_mels=self.n_mels
            )
            log_mel = np.log(mel + 1e-6)
            return torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)
        else:
            stft = torch.stft(waveform.squeeze(0), n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)
            spectrogram = torch.abs(stft)[:self.n_mels, :]
            return torch.log(spectrogram + 1e-6).unsqueeze(0)

    def process(self, file_path):
        """Full pipeline: load -> pad/crop -> compute mel spectrogram."""
        waveform = self.load_audio(file_path)
        waveform = self.pad_crop(waveform)
        mel_spec = self.compute_mel_spectrogram(waveform)
        return mel_spec


class ElephantDataset(Dataset):
    """PyTorch Dataset loading from manifest.csv with split support."""
    def __init__(self, manifest_path="dataset/manifest.csv", split="train", mode="detector", preprocessor=None):
        """
        manifest_path: path to manifest.csv
        split: 'train', 'val', or 'test'
        mode: 'detector' (binary: 0=non_elephant, 1=elephant)
              'classifier' (multi-class: 0=trumpet, 1=roar, 2=rumble)
        """
        self.manifest_path = manifest_path
        self.split = split
        self.mode = mode
        self.preprocessor = preprocessor or AudioPreprocessor()
        self.samples = []
        self.classifier_label_map = {"trumpet": 0, "roar": 1, "rumble": 2}

        self._load_samples()

    def _load_samples(self):
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Manifest not found at {self.manifest_path}. Run prepare_dataset.py first.")

        with open(self.manifest_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] != self.split:
                    continue

                filepath = row["filepath"]
                if self.mode == "detector":
                    label = int(row["label"])
                    self.samples.append((filepath, label))
                elif self.mode == "classifier":
                    # Derive specific call type from filepath or category
                    if row["category"] == "elephant":
                        for call in self.classifier_label_map:
                            if call in filepath.lower():
                                self.samples.append((filepath, self.classifier_label_map[call]))
                                break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        spectrogram = self.preprocessor.process(file_path)
        return spectrogram, torch.tensor(label, dtype=torch.long)


if __name__ == "__main__":
    print("Testing ElephantDataset with manifest.csv...")
    try:
        train_ds = ElephantDataset(manifest_path="dataset/manifest.csv", split="train", mode="detector")
        val_ds = ElephantDataset(manifest_path="dataset/manifest.csv", split="val", mode="detector")
        test_ds = ElephantDataset(manifest_path="dataset/manifest.csv", split="test", mode="detector")

        print(f"Dataset successfully initialized:")
        print(f"  Train samples: {len(train_ds)}")
        print(f"  Val samples:   {len(val_ds)}")
        print(f"  Test samples:  {len(test_ds)}")

        # Test loading a single sample
        if len(train_ds) > 0:
            spec, lbl = train_ds[0]
            print(f"  Sample 0 Spectrogram Shape: {spec.shape}, Label: {lbl.item()}")
    except Exception as e:
        print(f"Error during verification: {e}")