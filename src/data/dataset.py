import json
import librosa
from pathlib import Path
import pandas as pd

import random
from .utils import get_onset_list, make_target_tensor

import torch

from tqdm import tqdm

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_json, thres, target_length, segment_length=4, sr=16000):
        self.data_json = Path(data_json)
        self.thres = thres
        self.target_length = target_length
        self.sr = sr
        self.segment_length = segment_length
        self.segment_frame = sr * segment_length

        self.data = []

        self.load_data()

    def load_data(self):
        self.data = []

        with tqdm(json.load(open(self.data_json)), unit="audio") as t:
            t.set_description("Loading audios")

            for audio_dir in t:
                audio_dir = Path(audio_dir)
                audio_path = audio_dir / f"vocals-16k.wav"
                label_path = audio_dir / f"{audio_dir.name}_groundtruth.txt"

                audio, _ = librosa.load(audio_path, sr=self.sr)
                
                label = pd.read_csv(label_path, sep=" ", header=None, names=["start", "end", "pitch"])
                onset_list = get_onset_list(label, self.thres)

                onset_idx = 0
                for i, frame in enumerate(range(0, audio.shape[0] - self.segment_frame + 1, self.segment_frame)):
                    start_time = i * self.segment_length
                    end_time = (i + 1) * self.segment_length

                    onsets = []
                    while onset_idx < len(onset_list) and onset_list[onset_idx] < end_time:
                        onsets.append(onset_list[onset_idx])
                        onset_idx += 1

                    target = make_target_tensor(onsets, start_time, self.segment_length, self.target_length)
                    self.data.append([audio[frame:frame + self.segment_frame], target])

    def __getitem__(self, index):
        audio, target = self.data[index]
        audio = self.augmentation(audio, self.sr)

        return torch.tensor(audio).float(), target.float()

    @staticmethod
    def augmentation(audio, sr):
        pitch_factor = random.randint(-12, 12)
        noise_factor = 0.01
        # noise
        audio = librosa.effects.pitch_shift(audio, sr, pitch_factor)
        
        noise = np.random.randn(audio.shape[0]) * noise_factor
        audio = audio + noise

        return audio


    def __len__(self):
        return len(self.data)

