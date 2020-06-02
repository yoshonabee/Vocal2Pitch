from argparse import ArgumentParser

from collections import defaultdict
from pathlib import Path

import json
import math
import torch
from model import CNN

import numpy as np

from data import EvalDataset
from utils import get_predicting_args, set_seed

from tqdm import tqdm


def main(args):
    model = CNN()
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model_name = args.model_path.split("/")[-2]

    pred_onset_list = defaultdict(list)

    model.to(args.device)
    model.eval()

    audio_list = json.load(open(args.audio_list))

    with tqdm(audio_list, unit="audio") as t:
        for audio_dir in t:
            audio_dir = Path(audio_dir)
            dataset = EvalDataset(audio_dir)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

            for x, idx in dataloader:
                idx = idx.tolist()
                
                x = x.to(args.device)

                pred = (model(x) > args.thres).view(-1).long().tolist()

                for t, p in zip(idx, pred):
                    if p == 1:
                        pred_onset_list[audio_dir.name].append(float(t))


    for name in pred_onset_list:
        pred_onset_list[name] = sorted(pred_onset_list[name], key=lambda x: float(x))

    print(len(pred_onset_list))

    results = {}
    
    alpha = 0.2

    with tqdm(audio_list, unit="audio") as t:
        for audio_dir in t:
            audio_dir = Path(audio_dir)
            name = audio_dir.name

            pitch_list = np.array(json.load(open(audio_dir / f"{name}_vocal.json")))

            result = []

            pitch_list_idx = 0
            for i in range(1, len(pred_onset_list[name])):
                onset = pred_onset_list[name][i - 1]
                offset = pred_onset_list[name][i]

                if offset - onset <= 0.05:
                    continue
                try:
                    while pitch_list[pitch_list_idx][0] < onset * (1 - alpha) + offset * alpha:
                        pitch_list_idx += 1

                    start_idx = pitch_list_idx

                    while pitch_list[pitch_list_idx][0] < offset:
                        pitch_list_idx += 1

                    end_idx =  pitch_list_idx
                except:
                    if pitch_list_idx >= len(pitch_list):
                        break
                
                try:          

                    t = pitch_list[start_idx:end_idx,0]
                    pitch = pitch_list[start_idx:end_idx,1]

                    final_pitch = round(pitch.dot(t - t.min()) / (t - t.min()).sum())

                    if final_pitch > 35:
                        result.append([onset, offset, final_pitch])
                except:
                    from IPython import embed
                    from time import sleep
                    embed()
                    sleep(3)
            results[name] = result

    Path("results").mkdir(exist_ok=True)
    json.dump(results, open(f"results/{model_name}_{args.audio_list.split('/')[-1].split('.')[0]}.json", "w"))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = get_predicting_args(parser)
    args = parser.parse_args()

    set_seed(args.seed)

    print(args)
    main(args)
