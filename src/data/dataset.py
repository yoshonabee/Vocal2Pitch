import json
import random
import torchaudio
from pathlib import Path
import numpy as np
import pandas as pd

from .utils import get_onset_list, make_target_tensor

import torch

from tqdm import tqdm

from IPython import embed
from time import sleep
import math

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_json, feature_config):
        self.data_json = Path(data_json)
        self.feature_config = Path(feature_config)
        # self.model_down_sampling_rate = model_down_sampling_rate

        self.frame_hop = 1
        self.frame_width = 15

        assert self.frame_hop % 2 == 1

        self.data = []

        self._parse_feature_config()
        self.transform = self._get_transform()
        self._load_data()

    def _parse_feature_config(self):
        config = json.load(open(self.feature_config))

        for name, item in config.items():
            setattr(self, name, item)

    # def _get_target_frame(self):
    #     if self.domain == "time":
    #         return int(self.sr * self.segment_length / self.model_down_sampling_rate)
    #     else:
    #         return int(((self.sr * self.segment_length) / self.hop_length + 1) // self.model_down_sampling_rate)

    def _get_transform(self):
        return torchaudio.transforms.MelSpectrogram(44100, 2048, hop_length=512, f_min=27.5, f_max=16000, n_mels=80).cuda()
        if self.domain == "time":
            return lambda x: x
        elif self.domain == "spectrogram":
            return torchaudio.transforms.Spectrogram(self.n_fft, self.win_length, self.hop_length, normalized=self.normalized)
        elif self.domain == "mfcc":
            return torchaudio.transforms.MFCC(
                sample_rate=self.sr,
                n_mfcc=self.n_mfcc,
                log_mels=self.log_mels,
                melkwargs={
                    "n_fft": self.n_fft,
                    "win_length":self.win_length,
                    "hop_length":self.hop_length
                }
            )

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
                #     break

            # self.index = sorted(self.index, key=lambda x: -x[-1])

            # for i in range(len(self.index)):
            #     if self.index[i][-1] == 0:
            #         break

            
            # embed()

            # if i * 2 < len(self.index):
            #    self.max_index = i * 2
            #else:
            #    self.max_index = len(self.index)


                

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

        center_idx = math.ceil(self.frame_width / 2) - 1
        
        onset_idx = 0
        dont_add = 0
        for i in range(0, n_frames - self.frame_width - 2 * self.frame_hop + 1, self.frame_hop):
            if dont_add > 0:
                dont_add -= 1
                continue

            while onset_list[onset_idx] < frame_times[0] and onset_idx < len(onset_list) - 1:
                onset_idx += 1

            if onset_list[onset_idx] < frame_times[center_idx - self.frame_hop // 2] or onset_list[onset_idx] >= frame_times[center_idx + self.frame_hop // 2 + 1]:
                result.append([audio_id, i, i + self.frame_width, 0])
            else:
                if i != 0:
                    result[-1][-1] = 0.25
                result.append([audio_id, i, i + self.frame_width, 1])
                result.append([audio_id, i + self.frame_hop, i + self.frame_hop + self.frame_width, 0.25])
                dont_add = 1
            
            frame_times += frame_width * self.frame_hop

        return result

