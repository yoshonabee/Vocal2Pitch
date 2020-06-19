import json
import librosa
from multiprocessing import Pool
from pathlib import Path
import pandas as pd
import numpy as np
import random
from .utils import get_onset_list, make_target_tensor, make_pitch_tensor

import torch

from tqdm import tqdm

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_json, thres, target_length, segment_length=4, sr=16000, data_amount=1, augment=True):
        self.data_json = Path(data_json)
        self.thres = thres
        self.target_length = target_length
        self.sr = sr
        self.segment_length = segment_length
        self.segment_frame = sr * segment_length
        self.augment = augment
        self.data_amount = data_amount

        self.data = []

        self.load_data()
        self.augmentation()

    def load_data(self):
        self.data = []
        self.target = []
        self.pitch = []

        with tqdm(json.load(open(self.data_json)), unit="audio") as t:
            t.set_description("Loading audios")
            j = 0
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
                    pitchs = [onset_list[onset_idx - 1]] if onset_idx > 0 else []
                    
                    while onset_idx < len(onset_list) and onset_list[onset_idx][0] < end_time:
                        onsets.append(onset_list[onset_idx][0])
                        pitchs.append(onset_list[onset_idx])
                        onset_idx += 1
                    
                    target = make_target_tensor(onsets, start_time, self.segment_length, self.target_length)
                    pitch = make_pitch_tensor(pitchs, start_time, self.segment_length, self.target_length)
                    
                    self.data.append([audio[frame:frame + self.segment_frame], len(self.target), [len(self.pitch), 0]])
                    self.target.append(target)
                    self.pitch.append(pitch)
                #j += 1
                #if j ==5 :
                #    break
    def __getitem__(self, index):
        audio, target, (pitch, pitch_factor) = self.data[index]
        target = self.target[target]
        pitch = self.pitch[pitch] + pitch_factor
        

        return torch.tensor(audio).float(), target.float(), pitch.float()

    def augmentation(self):
        if self.augment == False or self.data_amount == 1:
            return

        data = list(zip(self.data, [self.data_amount for _ in range(len(self.data))], [self.sr for _ in range(len(self.data))]))

        with tqdm(total=len(data), unit="audios") as t:
            t.set_description("Augmentating audios")

            with Pool(4) as p:
                for results, targets, pitches in p.imap(_augmentation, data):
                    self.data.extend(list(zip(results, targets, pitches)))
                    t.update(1)

    def __len__(self):
        return len(self.data)

def _augmentation(inputs):
    (audio, target, pitch), data_amount, sr = inputs

    results = []
    pitches = []
    for i in range(data_amount - 1):
        pitch_factor = random.randint(-5, 5)
        noise_factor = 9 * random.random() / 100 + 1e-3 # [1e-3, 1e-2]
        
        pitches.append([pitch[0], pitch_factor])
        # noise
        noise = np.random.randn(audio.shape[0]) * noise_factor
        augmented = librosa.effects.pitch_shift(audio, sr, pitch_factor) + noise
        results.append(augmented)

    return results, [target for _ in range(len(results))], pitches



