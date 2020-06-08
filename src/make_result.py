from argparse import ArgumentParser

from collections import defaultdict
from pathlib import Path

import json
import math

import numpy as np
import pandas as pd

from data import EvalDataset
from utils import get_make_result_args

from tqdm import tqdm

def find_max_c(ts):
    max_c = 0
    max_t = 0

    for t, c in ts:
        if c > max_c:
            max_c = c
            max_t = t

    return max_t

def main(args):

    pred_onset_list = json.load(open(args.pred_onset_list))
    onset_list_length = 500 if len(pred_onset_list) <= 500 else 1500

    for i in range(1, onset_list_length + 1):
        name = str(i)

        if name not in pred_onset_list:
            if "test" in args.pred_onset_list:
                pred_onset_list[name] = []

        else:
            pred_list = np.array(pred_onset_list[name])
            times, preds = pred_list[:,0], pred_list[:,1]

            segments = []
            segment = []

            ts = []

            last_onset = None
            mean = preds.mean()
            std = preds.std()

            preds = (preds - mean) / std
            preds = (preds - preds.min()) / (preds.max() - preds.min())

            for t, pred in zip(times, preds):
                if pred > args.onset_thres:
                    ts.append(t)
                else:
                    if len(ts) > 0:
                        offset = ts[0]
                        onset = ts[-1]

                        if last_onset:
                            if offset - last_onset >= 0.08:
                                segments.append([last_onset, offset])

                        last_onset = onset

                        ts = []

            pred_onset_list[name] = segments
            # from IPython import embed
            # embed()



    pred_onset_list = dict(sorted(list(pred_onset_list.items()), key=lambda x: int(x[0])))

    print(len(pred_onset_list))

    data_dir = Path(args.data_dir)

    results = {}

    with tqdm(pred_onset_list, unit="audio") as t:
        for name in t:
            if len(pred_onset_list[name]) == 0:
                results[name] = []
                continue

            audio_dir = data_dir / name

            if not args.crepe:
                pitch_list = np.array(json.load(open(audio_dir / f"{name}_vocal.json")))
            else:
                raw_pitch_list = pd.read_csv(audio_dir / f"{name}_crepe.csv")
                pitch_list = raw_pitch_list.values[:,:2]

                for i in range(pitch_list.shape[0]):
                    if raw_pitch_list['confidence'][i] < args.crepe_confidence_thres:
                        pitch_list[i][1] = 0

            result = []

            pitch_list_idx = 0
            for onset, offset in pred_onset_list[name]:
                try:
                    while pitch_list[pitch_list_idx][0] < onset * (1 - args.alpha) + offset * args.alpha:
                        pitch_list_idx += 1

                    start_idx = pitch_list_idx

                    while pitch_list[pitch_list_idx][0] < offset:
                        pitch_list_idx += 1

                    end_idx = pitch_list_idx
                except:
                    if pitch_list_idx >= len(pitch_list):
                        break
                
                # try:
                final_pitch = process_pitch_outlier(pitch_list, start_idx, end_idx, freq2midi=args.crepe)          
                if final_pitch >= args.min_pitch and final_pitch <= args.max_pitch:
                    result.append([onset, offset, final_pitch])


            results[name] = result

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    json.dump(results, open(output_dir / f"{args.pred_onset_list.split('/')[-1][:-5]}_result.json", "w"))

def process_pitch_classical(pitch_list, start_idx, end_idx, freq2midi=False):
    t = pitch_list[start_idx:end_idx, 0]
    pitch = pitch_list[start_idx:end_idx, 1]

    final_pitch = pitch.dot(t - t.min()) / (t - t.min()).sum()

    if freq2midi:
        from audiolazy.lazy_midi import freq2midi

        return round(freq2midi(final_pitch))
    return round(final_pitch)

def process_pitch_mean(pitch_list, start_idx, end_idx, freq2midi=False):
    pitch = pitch_list[start_idx:end_idx, 1]

    final_pitch = pitch.mean()

    if freq2midi:
        from audiolazy.lazy_midi import freq2midi

        return round(freq2midi(final_pitch))
    return round(final_pitch)

def process_pitch_outlier(pitch_list, start_idx, end_idx, freq2midi=False):
    pitch = np.array([p for p in pitch_list[start_idx:end_idx, 1] if p > 0])
    
    final_pitch = pitch.mean()
    try:
        if freq2midi:
            from audiolazy.lazy_midi import freq2midi

            return round(freq2midi(final_pitch))
        return round(final_pitch)
    except:
        return 0


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = get_make_result_args(parser)
    args = parser.parse_args()

    print(args)
    main(args)
