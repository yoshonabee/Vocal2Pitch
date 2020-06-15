from argparse import ArgumentParser

from collections import defaultdict
from pathlib import Path

import json
import math

import torch

import numpy as np
import pandas as pd

from data import EvalDataset
from utils import get_make_result_args

from tqdm import tqdm
from multiprocessing import Pool

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

            segments = get_segments(times, preds, onset_thres=args.onset_thres, normalize=args.normalize)

            pred_onset_list[name] = segments
            # from IPython import embed
            # embed()

    pred_onset_list = dict(sorted(list(pred_onset_list.items()), key=lambda x: int(x[0])))

    data_dir = Path(args.data_dir)

    inputs = [
        (
            name,
            pred_onset_list[name],
            data_dir / name,
            args.alpha,
            args.crepe_confidence_thres,
            args.min_pitch,
            args.max_pitch
        ) for name in pred_onset_list

    ]

    print(len(inputs))

    results = {}

    with tqdm(total=len(inputs), unit="audio") as t:
        with Pool(args.num_workers) as p:
            for name, result in p.imap(process_audio, inputs):
                results[name] = result
                t.update(1)

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

def process_pitch_outlier_most(pitch_list, start_idx, end_idx, freq2midi=False):
    pitch = np.array([p for p in pitch_list[start_idx:end_idx, 1] if p > 0])

    try:
        if freq2midi:
            from audiolazy.lazy_midi import freq2midi
            pitch = np.array(list(map(freq2midi, pitch)))
        
        counts = np.bincount(pitch.round().astype(np.int64))
        final_pitch = int(np.argmax(counts))
        if counts[final_pitch] < 3:
            final_pitch = 0
    except:
        final_pitch = 0

    return final_pitch

def process_pitch_outlier_mid(pitch_list, start_idx, end_idx, freq2midi=False):
    pitch = np.array([p for p in pitch_list[start_idx:end_idx, 1] if p > 0])

    try:
        if freq2midi:
            from audiolazy.lazy_midi import freq2midi
            pitch = np.array(list(map(freq2midi, pitch)))
            
        final_pitch = int(np.median(pitch).round())
            
    except:
        final_pitch = 0
    
    return final_pitch

def get_segments(times, preds, onset_thres=None, normalize=False):
    if normalize:
        mean = preds.mean()
        std = preds.std()

        preds = (preds - mean) / std
        preds = (preds - preds.min()) / (preds.max() - preds.min())

    # p_counts = np.bincount((preds * 1000).astype(np.int64))
    # count_thres = args.onset_thres * preds.shape[0]

    # for onset_thres in range(p_counts.shape[0]):
    #     if p_counts[:onset_thres].sum() >= count_thres:
    #         break

    # onset_thres /= 1000

    # p_counts = np.bincount((preds * 1000).astype(np.int64))
    # count_thres = args.onset_thres


    # for onset_thres in range(p_counts.shape[0]):
    #     if p_counts[onset_thres] < count_thres:
    #         break

    # onset_thres /= 1000

    if onset_thres is None:
        onset_thres = 0.08

    segments = []
    segment = []

    ts = []

    last_onset = None

    for t, pred in zip(times, preds):
        if pred > onset_thres:
            ts.append((t, pred))
        else:
            if len(ts) > 0:
                # if ts[-1][0] - ts[0][0] > 0.05:
                if True:
                    offset = ts[0][0]
                    onset = ts[-1][0]
                else:
                    offset = 0
                    max_p = 0

                    for pair in ts:
                        if pair[1] > max_p:
                            offset = pair[0]
                            max_p = pair[1]

                    onset = offset + 1e-6

                if last_onset:
                    if offset - last_onset >= 0.1:
                        segments.append([last_onset, offset])

                last_onset = onset

                ts = []

    return segments

def get_segments_conv(times, preds):
    frame_width = 5
    half_length = int((frame_width - 1) / 2)
    kernel = np.zeros(frame_width).float()
    kernel[0] = -1
    kernel[-1] = -1
    kernel[half_length] == 2

    new_times = []
    scores = []

    for i in range(preds.shape[0]):
        if i - half_length < 0 or i + half_length >= preds.shape[0]:
            continue

        new_times.append(times[i])
        frame = preds[i - half_length:i + half_length + 1]

        frame = (frame - frame.mean()) / frame.std()
        score = frame * kernel

        scores.append(score)

    return new_times, scores

def process_audio(inputs):
    name, pred, audio_dir, alpha, crepe_confidence_thres, min_pitch, max_pitch = inputs

    if len(pred) == 0:
        return name, []

    if crepe_confidence_thres < 0:
        pitch_list = np.array(json.load(open(audio_dir / f"{name}_vocal.json")))
    else:
        raw_pitch_list = pd.read_csv(audio_dir / f"{name}_crepe.csv")
        pitch_list = raw_pitch_list.values[:,:2]

        for i in range(pitch_list.shape[0]):
            if raw_pitch_list['confidence'][i] < crepe_confidence_thres:
                pitch_list[i][1] = 0

    result = []

    pitch_list_idx = 0
    for onset, offset in pred:
        try:
            while pitch_list[pitch_list_idx][0] < onset * (1 - alpha) + offset * alpha:
                pitch_list_idx += 1

            start_idx = pitch_list_idx

            while pitch_list[pitch_list_idx][0] < offset:
                pitch_list_idx += 1

            end_idx = pitch_list_idx
        except:
            if pitch_list_idx >= len(pitch_list):
                break
        
        # try:
        final_pitch = process_pitch_outlier_most(pitch_list, start_idx, end_idx, freq2midi=crepe_confidence_thres > 0)          

        if final_pitch >= min_pitch and final_pitch <= max_pitch:
            result.append([onset, offset, final_pitch])

    return name, result


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = get_make_result_args(parser)
    args = parser.parse_args()

    print(args)
    main(args)
