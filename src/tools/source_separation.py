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
        if (audio_dir / f"{audio_dir.name}.wav").exists() and not (audio_dir / "vocals.wav").exists()
    ]

    print(len(audio_list))

    command = ["spleeter", "separate", "-i"]
    for audio in audio_list:
        command.append(str(audio))

    command.extend(["-p", "spleeter:2stems-16kHz", "-o", args.data_dir])

    subprocess.run(command)

if __name__ == '__main__':
    args = get_args()
    print(args)

    main(args)
