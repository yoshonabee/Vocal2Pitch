import json
import librosa
from pathlib import Path
import pandas as pd

from .utils import get_onset_list, make_target_tensor

import torch

from tqdm import tqdm

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_json, thres, target_length, domain="mfcc", segment_length=4, sr=16000, n_mfcc=None, window_len=None, hop_len=None):
        self.data_json = Path(data_json)
        self.thres = thres
        self.domain = domain
        self.target_length = target_length
        self.sr = sr
        self.segment_length = segment_length
        self.segment_frame = int(sr * segment_length)

        self.window_len = window_len
        self.hop_len = hop_len

        self.data = []

        self.load_data()

    def load_data(self):
        self.data = []

        with tqdm(json.load(open(self.data_json)), unit="audio") as t:
            t.set_description("Loading audios")
            for audio_dir in t:
                audio_dir = Path(audio_dir)
                audio_path = audio_dir / f"vocal.wav"
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

                    segment = audio[frame:frame + self.segment_frame]

                    if self.domain == 'mfcc':
                        segment = torch.tensor(librosa.feature.mfcc(segment, sr=self.sr, n_mfcc=self.n_mfcc))
                    elif self.domain == 'logmfcc':
                        segment = torch.tensor(librosa.feature.mfcc(segment, sr=self.sr, n_mfcc=self.n_mfcc))
                        segment = torch.log(segment + 1e-6)
                    else:
                        segment = torch.tensor(segment)

                    self.data.append([segment, target])

    def __getitem__(self, index):
        audio, target = self.data[index]

        return torch.tensor(audio).float(), target.long()

    def __len__(self):
        return len(self.data)

