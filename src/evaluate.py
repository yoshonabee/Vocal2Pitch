from argparse import ArgumentParser

from pathlib import Path

import json

import torch

import numpy as np
import pandas as pd

from utils import get_evaluating_args

from tqdm import tqdm

def main(args):
    setattr(args, 'data_dir', "../data/MIR-ST500/" if "val" in args.predict_json else "../data/AIcup_testset_ok/")

    predict_json = json.load(open(args.predict_json))

    data_dir = Path(args.data_dir)

    with tqdm(total=len(predict_json), unit="audio") as t:
        total_COn = 0
        total_COnP = 0
        total_COnPOff = 0

        for name in predict_json:
            if len(predict_json[name]) == 0:
                continue

            pred = np.array(predict_json[name])
            gt_path = data_dir / name / f"{name}_groundtruth.txt"

            gt = pd.read_csv(gt_path, sep=" ", header=None).values

            assert pred.shape[1] == gt.shape[1]

            con, conp, conpoff = evaluate(pred, gt)
            # print(name, len(pred), len(gt), con)

            total_COn += con
            total_COnP += conp
            total_COnPOff += conpoff

            t.update(1)

    print(f"total audios: {len(predict_json)}")
    print(f"COn: {total_COn / len(predict_json)}")
    print(f"COnP: {total_COnP / len(predict_json)}")
    print(f"COnPOff: {total_COnPOff / len(predict_json)}")
    print(f"score: {(total_COn * 0.2 + total_COnP * 0.6 + total_COnPOff * 0.2) / len(predict_json)}")

def evaluate(pred, target):
    i, j, con_tp, conp_tp, conpoff_tp = 0, 0, 0, 0, 0

    while i < len(pred) and j < len(target):

        if pred[i][0] < target[j][0] - 0.05:
            i += 1
        elif target[j][0] < pred[i][0] - 0.05:
            j += 1
        else:
            con_tp += 1

            if pred[i][2] == target[j][2]:
                conp_tp += 1

                if abs(pred[i][1] - target[j][1]) <= max(0.05, 0.2 * (target[j][1] - target[j][0])):
                    conpoff_tp += 1

            i += 1
            j += 1

    precision = con_tp / len(pred)
    recall = con_tp / len(target)
    con = 2 * precision * recall / (precision + recall + 1e-6)

    precision = conp_tp / len(pred)
    recall = conp_tp / len(target)
    conp = 2 * precision * recall / (precision + recall + 1e-6)

    precision = conpoff_tp / len(pred)
    recall = conpoff_tp / len(target)
    conpoff = 2 * precision * recall / (precision + recall + 1e-6)


    return con, conp, conpoff

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = get_evaluating_args(parser)
    args = parser.parse_args()

    print(args)
    main(args)
