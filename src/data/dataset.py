import json
import librosa
from pathlib import Path
import pandas as pd

from .utils import get_onset_list, make_target_tensor

import torch

from tqdm import tqdm

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_json, thres, segment_length=4):
        self.data_json = Path(data_json)
        self.thres = thres
        self.segment_length = segment_length
        self.segment_frame = int(segment_length // 0.032)

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
                feature_path = audio_dir / f"{audio_dir.name}_feature.json"

                feature = json.load(open(feature_path))
                del feature['time']
                del feature['pitch']
                feature = pd.DataFrame(feature).values
                
                label = pd.read_csv(label_path, sep=" ", header=None, names=["start", "end", "pitch"])
                onset_list = get_onset_list(label, self.thres)

                target = make_target_tensor(onset_list, feature.shape[0])
                
                self.index.extend([[i, j, j + self.segment_frame] for j in range(0, feature.shape[0] - self.segment_frame + 1)])
                self.data.append(torch.tensor(feature).float(), torch.tensor(target).float())


    def __getitem__(self, index):
        i, start, end = self.index[index]

        return self.data[i][0][start:end], self.data[i][1][start:end]

    def __len__(self):
        return len(self.index)

