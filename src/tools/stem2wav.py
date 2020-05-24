from argparse import ArgumentParser
from pathlib import Path
import subprocess

import librosa

from tqdm import tqdm

MIXTURE_INDEX = 0
VOCAL_INDEX = 4

def get_args():
    parser = ArgumentParser()
    parser.add_argument("musdb_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--sample_rate", default=16000, type=int)


    return parser.parse_args()

def extract_wav_from_stem(stem, output_path, i, sr=16000):
    command = ["ffmpeg", "-loglevel", "panic", "-i", stem, "-map", f"0:{i}", "-vn", "-y", "-ac", "1", "-ar", f"{sr}", output_path]
    subprocess.run(command)

def extract_accompaniment(mixture, vocal, output_path, sr):
    mixture, _ = librosa.load(mixture, sr=sr)
    vocal, _ = librosa.load(vocal, sr=sr)

    assert mixture.shape == vocal.shape
    librosa.output.write_wav(output_path, mixture - vocal, sr=sr)

def main(args):
    musdb_dir = Path(args.musdb_dir)

    if not musdb_dir.is_dir():
        raise ValueError("Please input valid musdb dir")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "test"):
        print(f"Processing split '{split}'")

        stem_list = [stem for stem in (musdb_dir / split).glob("*.stem.mp4")]
        with tqdm(stem_list, unit="audio") as t:
            for stem in t:
                name = ".".join(stem.name.split('.')[:-2])
                stem_dir = output_dir / split / name

                stem_dir.mkdir(parents=True, exist_ok=True)

                mixture_path = stem_dir / "mixture.wav"
                vocal_path = stem_dir / "vocal.wav"
                accompaniment_path = stem_dir / "accompaniment.wav"

                try:
                    extract_wav_from_stem(stem, mixture_path, MIXTURE_INDEX, args.sample_rate)
                    extract_wav_from_stem(stem, vocal_path, VOCAL_INDEX, args.sample_rate)
                    extract_accompaniment(mixture_path, vocal_path, accompaniment_path, args.sample_rate)
                except Exception as e:
                    print(e)

if __name__ == '__main__':
    args = get_args()
    main(args)



