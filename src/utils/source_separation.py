import subprocess
from multiprocessing import Pool
from pathlib import Path
from argparse import ArgumentParser

from tqdm import tqdm

def get_args():
    parser = ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("output_dir")

    return parser.parse_args()

def main(args):
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not output_dir.exists() or not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)

    for audio in data_dir.glob("*.wav"):
        subprocess.run(["spleeter", "separate", "-i", str(audio), "-p", "spleeter:2stems", "-o", args.output_dir])

if __name__ == '__main__':
    args = get_args()
    print(args)

    main(args)
