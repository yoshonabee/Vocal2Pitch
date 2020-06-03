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
    parser.add_argument("data_dir")
    parser.add_argument("--sample_rate", default=16000, type=int)
    parser.add_argument("--concurrency", type=int, default=5)


    return parser.parse_args()

def transform_to_wav(input):
    audio, output_path, sr = input
    try:
        command = ["ffmpeg", "-loglevel", "panic", "-i", str(audio), "-y", "-ac", "1", "-ar", str(sr), "-f", "wav", str(output_path)]
        subprocess.run(command)

        return True
    except Exception as e:
        print(e)
        return False

def main(args):
    data_dir = Path(args.data_dir)

    if not data_dir.is_dir():
        raise ValueError("Please input valid audio dir")

    audio_list = [
        (
            audio_dir / f"vocals.wav",
            audio_dir / f"vocals-16k.wav",
            args.sample_rate
        ) 
        for audio_dir in data_dir.glob("*")
        # for audio in audio_dir.glob(f"{audio_dir.name}.*")
        # for audio in audio_dir.glob(f"{audio_dir.name}.*")
        # if audio_dir.name.isdigit() and audio.suffix.strip('.') in ("wav", "m4a", "aac", "mp3")
        if (audio_dir / "vocals.wav").exists()
    ]
    print(len(audio_list))
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



