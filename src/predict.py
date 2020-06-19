from argparse import ArgumentParser

from collections import defaultdict
from pathlib import Path

import json
import math
import torch

import numpy as np
from model import CNN_Transformer
from utils import get_predicting_args, get_model_args, get_data_args, set_seed

from tqdm import tqdm
from data import EvalDataset

torch.set_num_threads(4)

def main(args):
    model = CNN_Transformer()

    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model_name = args.model_path.split("/")[-2]
    split = args.audio_list.split("/")[-1].split(".")[0]
    pred_onset_list = defaultdict(list)

    model.to(args.device)
    model.eval()

    audio_list = json.load(open(args.audio_list))
    pred_onset_list = defaultdict(list)

    with tqdm(audio_list, unit="audio") as t:
        for audio_dir in t:
            audio_dir = Path(audio_dir)
            dataset = EvalDataset(
                audio_dir,
                thres=args.thres,
                segment_length=args.segment_length,
                sr=args.sr,
            )

            dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

            with torch.no_grad():
                for x, idx in dataloader:
                    x = x.to(args.device)

                    # pred = model(x).view(-1).tolist()
                    # idx = idx.view(-1).tolist()

                    # pred_onset_list[audio_dir.name].extend(list(zip(idx, pred)))

                    pred = model(x)[0].view(x.size(0), -1).cpu().numpy()
                    
                    time_frames = np.array([i for i in range(0, int(pred.shape[1] * args.segment_length), args.segment_length)])
                    time_frames = time_frames / pred.shape[1]
                    
                    for start_time, batch in zip(idx.view(-1).numpy(), pred):
                        times = time_frames + start_time

                        pred_onset_list[audio_dir.name].extend(list(zip(times.tolist(), batch.tolist())))

    for name in pred_onset_list:
        pred_onset_list[name] = sorted(pred_onset_list[name], key=lambda x: float(x[0]))

    print(len(pred_onset_list))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    with (output_dir / f"{model_name}_{split}.json").open("w") as f:
        f.write(json.dumps(pred_onset_list))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = get_predicting_args(parser)
    parser = get_model_args(parser)
    parser = get_data_args(parser)

    args = parser.parse_args()

    set_seed(args.seed)

    print(args)
    main(args)
