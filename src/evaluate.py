from argparse import ArgumentParser

from pathlib import Path

import torchaudio
import math
import torch
from model import CNN
from data import EvalDataset
from utils import get_evaluating_args, set_seed

transform = torchaudio.transforms.MelSpectrogram()
def main(args):
    model = CNN()
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))

    dataset = EvalDataset(args.audio_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    times = []
    predictions = []
    model = model.to(args.device)
    with tqdm(dataloader, total=math.ceil(len(dataset) / args.batch_size), unit="sample") as t:
        for i, (x, idx) in t:
            times.extend(idx.tolist())
            x = x.to(args.device)

            pred = (model(x) > args.thres).long()
            predictions.extend(pred.tolist())

    pred_onset_list = []
    for t, label in zip(times, predictions):
        if label == 1:
            pred_onset_list.append(t)

    onset_list = dataset.onset_list

    tp = cal_tp(pred_onset_list, onset_list)

    precision = tp / len(pred_onset_list)
    recall = tp / len(onset_list)

    f1_score = 2 * precision * recall / (precision + recall)

    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"f1_score": {f1_score})

def cal_tp(pred, target):
    i, j = 0, 0

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