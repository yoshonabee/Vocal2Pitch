from argparse import ArgumentParser
from pathlib import Path
import subprocess

from multiprocessing import Pool

import librosa

from tqdm import tqdm

MIXTURE_INDEX = 0
VOCAL_INDEX = 4

def get_args():
    parser = ArgumentParser()
    parser.add_argument("audio_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--sample_rate", default=44100, type=int)
    parser.add_argument("--concurrency", type=int, default=5)


    return parser.parse_args()

def transform_to_wav(input):
    audio, output_path, sr = input
    try:
        command = ["ffmpeg", "-loglevel", "panic", "-i", audio, "-y", "-ac", "1", "-ar", f"{sr}", "-f", "wav", output_path]
        subprocess.run(command)

        return True
    except Exception as e:
        print(e)
        return False



def main(args):
    audio_dir = Path(args.audio_dir)

    if not audio_dir.is_dir():
        raise ValueError("Please input valid audio dir")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_list = [
        [audio, output_dir / f"{audio.name.split('.')[0]}.wav", args.sample_rate]
        for audio in audio_dir.glob("*.m4a")
    ]

    p = Pool(args.concurrency)

    with tqdm(total=len(audio_list), unit="audio") as t:
        for ok in p.imap(transform_to_wav, audio_list):
            if ok:
                t.update(1)
            else:
                raise RuntimeError("ffmpeg failed, process terminate")

if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)



