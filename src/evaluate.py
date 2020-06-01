from argparse import ArgumentParser

from collections import defaultdict
from pathlib import Path

import json
import math
import torch
from model import CNN
from data import EvalDataset
from utils import get_evaluating_args, set_seed

from tqdm import tqdm


def main(args):
    model = CNN()
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))

    pred_onset_list = defaultdict(list)
    onset_list = {}

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
                        pred_onset_list[audio_dir.name].append(t)


            onset_list[audio_dir.name] = dataset.onset_list

    print(len(onset_list), len(pred_onset_list))

    tp = 0
    n_target = 0
    n_pred = 0

    for name in onset_list:
        tp += cal_tp(pred_onset_list[name], onset_list[name])
        n_target += len(onset_list[name])
        n_pred += len(pred_onset_list[name])

    precision = tp / n_pred
    recall = tp / n_target

    f1_score = 2 * precision * recall / (precision + recall)

    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"f1_score: {f1_score}")

def cal_tp(pred, target):
    i, j, tp = 0, 0, 0

    while i < len(pred) and j < len(target):
        if pred[i] < target[j] - 0.05:
            i += 1
        elif target[j] < pred[i] - 0.05:
            j += 1
        else:
            tp += 1
            i += 1
            j += 1

    return tp

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = get_evaluating_args(parser)
    args = parser.parse_args()

    set_seed(args.seed)

    print(args)
    main(args)
