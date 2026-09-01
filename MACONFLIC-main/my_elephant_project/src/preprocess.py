"""
preprocess.py - Audio loading & Mel-Spectrogram generator for Elephant Audio Classification project.
"""

import os
import glob
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Optional imports with fallbacks
try:
    import torchaudio
    import torchaudio.transforms as T
    HAS_TORCHAUDIO = True
except ImportError:
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
            raise RuntimeError("Neither torchaudio nor librosa is installed. Please install one to load audio files.")

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
            # Fallback simple STFT calculation if librosa/torchaudio are absent
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
    """PyTorch Dataset for elephant binary or call-type multi-class audio data."""
    def __init__(self, data_dir, mode="detector", preprocessor=None):
        """
        mode: 'detector' (binary: elephant vs non_elephant)
              'classifier' (multi-class: trumpet vs roar vs rumble)
        """
        self.data_dir = data_dir
        self.mode = mode
        self.preprocessor = preprocessor or AudioPreprocessor()
        self.samples = []
        self.label_map = {}

        self._load_samples()

    def _load_samples(self):
        if self.mode == "detector":
            self.label_map = {"non_elephant": 0, "elephant": 1}
            for label_name, label_idx in self.label_map.items():
                category_dir = os.path.join(self.data_dir, label_name)
                if not os.path.exists(category_dir):
                    continue
                for root, _, files in os.walk(category_dir):
                    for file in files:
                        if file.endswith((".wav", ".mp3", ".flac", ".ogg")):
                            self.samples.append((os.path.join(root, file), label_idx))

        elif self.mode == "classifier":
            elephant_dir = os.path.join(self.data_dir, "elephant")
            call_types = ["trumpet", "roar", "rumble"]
            self.label_map = {call: i for i, call in enumerate(call_types)}
            for call in call_types:
                call_dir = os.path.join(elephant_dir, call)
                if not os.path.exists(call_dir):
                    continue
                for file in os.listdir(call_dir):
                    if file.endswith((".wav", ".mp3", ".flac", ".ogg")):
                        self.samples.append((os.path.join(call_dir, file), self.label_map[call]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        spectrogram = self.preprocessor.process(file_path)
        return spectrogram, torch.tensor(label, dtype=torch.long)


if __name__ == "__main__":
    print("Preprocess module loaded. Ready to process audio files.")
