from argparse import ArgumentParser

from collections import defaultdict
from pathlib import Path

import json
import math
import torch
from model import CNN
from data import EvalDataset
import numpy as np

from utils import get_predicting_args, set_seed

from tqdm import tqdm


def main(args):
    model = CNN()
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model_name = args.model_path.split("/")[-2]
    split_name = args.audio_list.split("/")[-1].split(".")[0]

    pred_onset_list = defaultdict(list)

    model.to(args.device)
    model.eval()

    audio_list = json.load(open(args.audio_list))
    pred_onset_list = defaultdict(list)

    with tqdm(audio_list, unit="audio") as t:
        for audio_dir in t:
            audio_dir = Path(audio_dir)
            dataset = EvalDataset(audio_dir)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

            for x, idx in dataloader:
                x = x.to(args.device)

                pred = model(x).view(-1).tolist()
                idx = idx.view(-1).tolist()

                pred_onset_list[audio_dir.name].extend(list(zip(idx, pred)))

    for name in pred_onset_list:
        pred_onset_list[name] = sorted(pred_onset_list[name], key=lambda x: float(x[0]))

    print(len(pred_onset_list))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    with (output_dir / f"{model_name}_{split_name}.json").open("w") as f:
        f.write(json.dumps(pred_onset_list))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = get_predicting_args(parser)
    args = parser.parse_args()

    set_seed(args.seed)

    print(args)
    main(args)
