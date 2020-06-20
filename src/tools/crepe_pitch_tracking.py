import subprocess
from pathlib import Path
from argparse import ArgumentParser
import random
import pandas as pd

import librosa
from tqdm import tqdm

import tensorflow as tf

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.4)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
tf.compat.v1.keras.backend.set_session(sess)

import crepe
def get_args():
    parser = ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("-j", "--concurrency", type=int, default=1)

    return parser.parse_args()

def predict(inputs):
    audio_path, sr = inputs
    print(f"{audio_path.parent.name}_crepe.csv")
    if (audio_path.parent / f"{audio_path.parent.name}_crepe.csv").exists():
        return audio_path, None, None, None
    audio, _ = librosa.load(audio_path, sr=sr)
   
    time, frequency, confidence, _ = crepe.predict(audio, sr, viterbi=True)

    return audio_path, time, frequency, confidence

def main(args):
    data_dir = Path(args.data_dir)

    audio_list = [
        (audio, args.sample_rate)
        for audio in data_dir.glob("*/vocals.wav")
    ]
    
    random.shuffle(audio_list)

    with tqdm(total=len(audio_list), unit="audio") as t:
        for audio, sr in audio_list:
            audio_path, time, frequency, confidence = predict((audio, sr))
            output_path = audio_path.parent / f"{audio_path.parent.name}_crepe.csv"
            if not output_path.exists():
                df = pd.DataFrame({
                    'time': time,
                    'frequency': frequency,
                    'confidence': confidence
                })

                df.to_csv(output_path, index=False)
            t.update(1)
        

if __name__ == '__main__':
    args = get_args()
    print(args)

    main(args)
