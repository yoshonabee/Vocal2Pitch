import json
import random
from pathlib import Path
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--val_set_size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=39)
    parser.add_argument("--test", action="store_true")

    return parser.parse_args()

def main(args: dict) -> None:
    random.seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    audio_list = [
        str(audio_dir.absolute())
        for audio_dir in data_dir.glob("*")
        if audio_dir.is_dir()
        and (audio_dir / "vocals.wav").exists()
        and ((audio_dir / f"{audio_dir.name}_groundtruth.txt").exists() or args.test)
    ]

    random.shuffle(audio_list)
    
    if args.test:
        json.dump(audio_list, (output_dir / "test.json").open("w"))
    else:
        val_audio_list = audio_list[:args.val_set_size]
        train_audio_list = audio_list[args.val_set_size:]

        json.dump(train_audio_list, (output_dir / "train.json").open("w"))
        json.dump(val_audio_list, (output_dir / "val.json").open("w"))

if __name__ == '__main__':
    args = get_args()
    main(args)
