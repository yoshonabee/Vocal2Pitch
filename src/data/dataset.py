import json
import torchaudio
from pathlib import Path
import pandas as pd

from .utils import get_onset_list, make_target_tensor

import torch

from tqdm import tqdm

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_json, feature_config, model_down_sampling_rate):
        self.data_json = Path(data_json)
        self.feature_config = Path(feature_config)
        self.model_down_sampling_rate = model_down_sampling_rate

        self.data = []

        self._parse_feature_config()
        self.transform = self._get_transform()
        self._load_data()

    def _parse_feature_config(self):
        config = json.load(open(self.feature_config))

        for name, item in config.items():
            setattr(self, name, item)

    def _get_target_frame(self):
        # if args.domain == "time":
        return int(self.sr * self.segment_length / self.model_down_sampling_rate)
        # else:
        #     target_length = int(((args.sr * args.segment_length - args.window_len) / args.hop_len + 1) // model.down_sampling_factor)

    def _get_transform(self):
        if self.domain == "time":
            return lambda x: x
        elif self.domain == "spectrogram":
            return torchaudio.transforms.spectrogram(self.n_fft, self.win_length, self.hop_length, normalized=self.normalized)

    def _load_data(self):
        self.data = []

        segment_frame = self.sr * self.segment_length
        target_frame = self._get_target_frame()

        with tqdm(json.load(open(self.data_json)), unit="audio") as t:
            t.set_description("Loading audios")
            for audio_dir in t:
                audio_dir = Path(audio_dir)
                # audio_path = audio_dir / f"vocal.wav"
                audio_path = audio_dir / f"{audio_dir.name}.wav"
                label_path = audio_dir / f"{audio_dir.name}_groundtruth.txt"

                audio, _ = torchaudio.load(audio_path)
                
                label = pd.read_csv(label_path, sep=" ", header=None, names=["start", "end", "pitch"])
                onset_list = get_onset_list(label, self.thres)

                print(audio.shape[0], segment_frame)
                
                onset_idx = 0
                for i, frame in enumerate(range(0, audio.shape[0] - segment_frame + 1, segment_frame)):
                    start_time = i * self.segment_length
                    end_time = (i + 1) * self.segment_length

                    onsets = []
                    while onset_idx < len(onset_list) and onset_list[onset_idx] < end_time:
                        onsets.append(onset_list[onset_idx])
                        onset_idx += 1

                    target = make_target_tensor(onsets, start_time, self.segment_length, self.target_length)

                    segment = audio[frame:frame + segment_frame]
                    segment = self.transform(segment)

                    from IPython import embed
                    embed()

                    self.data.append([segment, target])

    def __getitem__(self, index):
        audio, target = self.data[index]

        return torch.tensor(audio).float(), target.long()

    def __len__(self):
        return len(self.data)

