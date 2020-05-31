import json
import torchaudio
from pathlib import Path
import pandas as pd

from .utils import get_onset_list

import torch

from tqdm import tqdm

class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, audio_dir):
        self.audio_dir = Path(audio_dir)

        self._parse_feature_config()
        self.transform = torchaudio.transform.MelSpectrogram(
            sample_rate=44100,
            n_fft=2048,
            hop_length=512,
            f_min=27.5,
            f_max=16000,
            n_mels=80
        )

        self._load_data()

    def _load_data(self):
        audio_path = self.audio_dir / "vocals.wav"
        label_path = self.audio_dir / f"{self.audio_dir.name}_groundtruth.txt"

        audio, sr = torchaudio.load(audio_path)
        audio = audio.sum(0).view(-1)

        self.spectrogram = self.transform(audio)

        self.onset_list = get_onset_list(label_path)

        frame_time = (audio.size(0) / sr) / self.spectrogram.size(1)

        for i in range(self.spectrogram.size(1)):
            self.frame_time.append((i + 7) * frame_time)

    def __getitem__(self, index):
        return self.spectrogram[index:index + 15].float(), self.frame_time[index]

    def __len__(self):
        return len(self.data)

