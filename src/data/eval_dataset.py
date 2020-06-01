import json
from collections import defaultdict
import torchaudio
from pathlib import Path
import pandas as pd

from .utils import get_onset_list

import torch

from tqdm import tqdm

class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, audio_list):
        self.audio_list = Path(audio_list)

        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=44100,
            n_fft=2048,
            hop_length=512,
            f_min=27.5,
            f_max=16000,
            n_mels=80
        ).cuda()

        self._load_data()

    def _load_data(self):
        self.onset_list = {}
        self.index = []
        self.data = []
        self.frame_time = {}
        with tqdm(json.load(open(self.audio_list)), unit="audio") as t:
            for e, audio_dir in enumerate(t):
                audio_dir = Path(audio_dir)
                audio_path = audio_dir / "vocals.wav"
                label_path = audio_dir / f"{self.audio_dir.name}_groundtruth.txt"

                audio, sr = torchaudio.load(audio_path)
                audio = audio.sum(0).view(-1)

                spectrogram = self.transform(audio.cuda()).detach().cpu()
                self.data.append(spectrogram)

                self.onset_list[audio_dir.name] = get_onset_list(label_path)

                frame_time = (audio.size(0) / sr) / spectrogram.size(1)

                self.index.extend([[e, i, i + 15] for i in range(spectrogram.size(1) - 15 + 1)])

                self.frame_time[e] = [(audio_dir.name, (i + 7) * frame_time) for i in range(spectrogram.size(1) - 15 + 1)]

    def __getitem__(self, index):
        audio_id, start, end = self.index[index]

        return self.spectrogram[audio_id][start:end].float(), self.frame_time[audio_id][start]

    def __len__(self):
        return len(self.index)

