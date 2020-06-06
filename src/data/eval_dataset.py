import json
import librosa
from pathlib import Path
import pandas as pd

import torch

from tqdm import tqdm

class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, data_json, thres, segment_length=4, sr=16000):
        self.audio_dir = Path(data_json)
        self.thres = thres
        self.sr = sr
        self.segment_length = segment_length
        self.segment_frame = sr * segment_length

        self.data = []

        self.load_data()

    def load_data(self):
        self.data = []
        
        if (self.audio_dir / "vocals-16k.wav").exists():
            audio_path = self.audio_dir / "vocals-16k.wav"
        else:
            audio_path = self.audio_dir / f"vocals.wav"

        audio, _ = librosa.load(audio_path, sr=self.sr)

        for i, frame in enumerate(range(0, audio.shape[0] - self.segment_frame + 1, self.segment_frame)):
            start_time = i * self.segment_length
            end_time = (i + 1) * self.segment_length

            self.data.append([torch.tensor(audio[frame:frame + self.segment_frame]).float(), start_time])

    def __getitem__(self, index):
        segment, start_time = self.data[index]

        return segment, start_time

    def __len__(self):
        return len(self.data)

