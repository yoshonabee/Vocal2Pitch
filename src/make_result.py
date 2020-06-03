from argparse import ArgumentParser

from collections import defaultdict
from pathlib import Path

import json
import math
import torch
from model import CNN

import numpy as np

from data import EvalDataset
from utils import get_make_result_args

from tqdm import tqdm


def main(args):

    pred_onset_list = json.load(open(args.pred_onset_list))
    onset_list_length = 500 if len(pred_onset_list) <= 500 else 1500
    
    for i in range(1, onset_list_length + 1):
        name = str(i)
        
        if name not in pred_onset_list:
            if "test" in args.pred_onset_list:
                pred_onset_list[name] = []

        else:
            new_pred_onset_list = []

            for t, pred in pred_onset_list[name]:
                if pred >= args.confident_thres:
                    new_pred_onset_list.append(t)

            pred_onset_list[name] = new_pred_onset_list


    print(len(pred_onset_list))
    pred_onset_list = dict(sorted(list(pred_onset_list.items()), key=lambda x: int(x[0])))

    print(len(pred_onset_list))

    data_dir = Path(args.data_dir)

    results = {}

    with tqdm(pred_onset_list, unit="audio") as t:
        for name in t:
            audio_dir = data_dir / name

            pitch_list = np.array(json.load(open(audio_dir / f"{name}_vocal.json")))

            result = []

            pitch_list_idx = 0
            for i in range(1, len(pred_onset_list[name])):
                onset = pred_onset_list[name][i - 1]
                offset = pred_onset_list[name][i]

                if offset - (onset * (1 - args.alpha) + offset * args.alpha) <= args.min_onset_offset_thres:
                    continue
                try:
                    while pitch_list[pitch_list_idx][0] < onset * (1 - args.alpha) + offset * args.alpha:
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
                    if final_pitch >= args.min_pitch:
                        result.append([onset, offset, final_pitch])
                except:
                    from IPython import embed
                    from time import sleep
                    embed()
                    sleep(3)
            results[name] = result

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    json.dump(results, open(output_dir / f"{args.pred_onset_list.split('/')[-1].split('.')[0]}_result.json", "w"))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = get_make_result_args(parser)
    args = parser.parse_args()

    print(args)
    main(args)
