import subprocess
from multiprocessing import Pool
from pathlib import Path
from argparse import ArgumentParser

import librosa
from tqdm import tqdm

def get_args():
    parser = ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("-j", "--concurrency", type=int, default=4)

    return parser.parse_args()

def predict(inputs):
    audio_path, sr = inputs

    audio, _ = librosa.load(audio_path, sr=sr)

    time, frequency, confidence, _ = crepe.predict(audio, sr, viterbi=True)

    return audio_path, time, frequency, confidence

def main(args):
    data_dir = Path(args.data_dir)

    audio_list = [
        audio,
        args.sample_rate
        for audio in data_dir.glob("*/vocals.wav")
    ]

    print(len(audio_list))

    with Pool(args.concurrency) as p:
        for audio_path, time, frequency, confidence in p.imap(predict, audio_list):
            output_path = audio_path.parent / f"{audio_path.parent.name}_crepe.csv"
            df = pd.DataFrame({
                'time': time,
                'frequency': frequency,
                'confidence': confidence
            })

            df.to_csv(output_path, index=False)
        

if __name__ == '__main__':
    args = get_args()
    print(args)

    main(args)
