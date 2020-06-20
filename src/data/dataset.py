import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
import torchaudio

from .utils import get_onset_list, make_target_tensor



class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_json, feature_config):
        self.data_json = Path(data_json)

    def __init__(self, data_json, thres, target_length, segment_length=4, sr=16000, data_amount=5, augment=True):
        self.data_json = Path(data_json)
        self.thres = thres
        self.target_length = target_length
        self.sr = sr
        self.segment_length = segment_length
        self.segment_frame = sr * segment_length
        self.augment = augment
        self.data_amount = data_amount

        self.frame_hop = 1
        self.frame_width = 31

        assert self.frame_hop % 2 == 1

        self.data = []

        self._parse_feature_config()
        self.transform = self._get_transform()
        self._load_data()

    def _parse_feature_config(self):
        config = json.load(open(self.feature_config))

        for name, item in config.items():
            setattr(self, name, item)

    def _get_transform(self):
        return torchaudio.transforms.MelSpectrogram(44100, 2048, hop_length=512, f_min=27.5, f_max=16000, n_mels=120).cuda()

    def _load_data(self):
        self.data = []
        self.index = []

        segment_frame = self.sr * self.segment_length
        # target_frame = self._get_target_frame()
        # target_frame = 86

        with tqdm(json.load(open(self.data_json)), unit="audio") as t:

            for i, audio_dir in enumerate(t):
                audio_dir = Path(audio_dir)

                t.set_description(audio_dir.name)

                audio_path = audio_dir / f"vocals.wav"

                label_path = audio_dir / f"{audio_dir.name}_groundtruth.txt"

                audio, _ = torchaudio.load(str(audio_path))
                label = pd.read_csv(label_path, sep=" ", header=None, names=["start", "end", "pitch"])
                onset_list = get_onset_list(label, 0.032)
                
                audio = audio.sum(0).view(-1)
                audio_length = audio.size(0) / 44100

                spectrogram = self.transform(audio.cuda()).detach().cpu()

                self.data.append(spectrogram)

                idxs = self.parse_index(i, onset_list, spectrogram.size(1), audio_length)
                self.index.extend(idxs)
                
                #if i == 5:
                #    break

    def __getitem__(self, index):
        audio_id, start, end, target = self.index[index]

        context = self.data[audio_id][:,start:end]

        return context, float(target)

    def __len__(self):
        # return self.max_index
        return len(self.index)

    def parse_index(self, audio_id, onset_list, n_frames, audio_length):
        result = []
        frame_width = audio_length / n_frames
        frame_times = np.array([i * frame_width for i in range(self.frame_width)])

        onset_idx = 0
        for i in range(0, n_frames - self.frame_width - self.frame_hop // 2 + 1, self.frame_hop):

            while onset_list[onset_idx] < frame_times[0] and onset_idx < len(onset_list) - 1:
                onset_idx += 1

            start_idx = (self.frame_width // 2) - 2 * self.frame_hop
            end_idx = (self.frame_width // 2) + 3 * self.frame_hop

            if onset_list[onset_idx] >= frame_times[end_idx] or onset_list[onset_idx] < frame_times[start_idx]:
                target = 0
            else:
                for j in range(start_idx, end_idx):
                    if onset_list[onset_idx] >= frame_times[j] and onset_list[onset_idx] < frame_times[j + 1]:
                        if j == self.frame_width // 2:
                            target = 1
                        elif j == self.frame_width // 2 - self.frame_hop or j == self.frame_width // 2 + self.frame_hop:
                            target = 0.6
                        elif j == self.frame_width // 2 - 2 * self.frame_hop or j == self.frame_width // 2 + 2 * self.frame_hop:
                            target = 0.2
                        else: target = 0

                        break

            result.append([audio_id, i, i + self.frame_width, target])
            frame_times += frame_width * self.frame_hop

        return result
