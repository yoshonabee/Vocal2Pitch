import json
import librosa
from pathlib import Path
import pandas as pd

from .utils import get_onset_list, make_target_tensor

import torch
import random

from tqdm import tqdm

from audiolazy.lazy_midi import freq2midi

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_json, thres, segment_length=4):
        self.data_json = Path(data_json)
        self.thres = thres
        self.segment_length = segment_length
        self.segment_frame = int(segment_length / 0.01)

        self.data = []
        self.index = []

        self.load_data()

    def load_data(self):
        self.data = []
        self.index = []

        with tqdm(json.load(open(self.data_json)), unit="audio") as t:
            t.set_description("Loading audios")

            for i, audio_dir in enumerate(t):
                audio_dir = Path(audio_dir)
                label_path = audio_dir / f"{audio_dir.name}_groundtruth.txt"
                pitch_path = audio_dir / f"{audio_dir.name}_crepe.csv"

                pitch = pd.read_csv(pitch_path)

                del pitch['time']
                pitch['frequency'] = pd.Series(list(map(freq2midi, pitch['frequency'].tolist())))
                pitch = pitch.values
                
                label = pd.read_csv(label_path, sep=" ", header=None, names=["start", "end", "pitch"])
                onset_list = get_onset_list(label, self.thres)

                target = make_target_tensor(onset_list, pitch.shape[0] // 2)
                
                self.index.extend([[i, (j, j + self.segment_frame), (k, k + self.segment_frame // 2)] for j, k in zip(range(0, pitch.shape[0] - self.segment_frame + 1), range(0, target.shape[0] - self.segment_frame // 2 + 1))])
                self.data.append([torch.tensor(pitch).float(), target])



    def __getitem__(self, index):
        i, (start, end), (val_start, val_end) = self.index[index]

        return self.augmentation(self.data[i][0][start:end]), self.data[i][1][val_start:val_end]

    @staticmethod
    def augmentation(tensor):
        pitch_shift = random.randint(-12, 12)
        tensor[:,0] += pitch_shift

        return tensor

    def __len__(self):
        return len(self.index)

