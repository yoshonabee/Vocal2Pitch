import subprocess
from multiprocessing import Pool
from pathlib import Path
from argparse import ArgumentParser

from tqdm import tqdm

def get_args():
    parser = ArgumentParser()
    parser.add_argument("data_dir")

    return parser.parse_args()

def main(args):
    data_dir = Path(args.data_dir)

    audio_list = [
        audio_dir / f"{audio_dir.name}.wav"
        for audio_dir in data_dir.glob("*")
        if (audio_dir / f"{audio_dir.name}.wav").exists()
    ]

    with tqdm(audio_list, unit="audio") as t:
        for audio in t:
            subprocess.run(["spleeter", "separate", "-i", str(audio), "-p", "spleeter:2stems", "-o", args.data_dir])

if __name__ == '__main__':
    args = get_args()
    print(args)

    main(args)
