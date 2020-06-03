from argparse import ArgumentParser
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import sys

import json
from pathlib import Path

import numpy as np
import pandas as pd

def get_args():
    parser = ArgumentParser()
    parser.add_argument("prediction")
    parser.add_argument("data_dir")

    return parser.parse_args()

def main(args):
    data_dir = Path(args.data_dir)
    df = json.load(open(args.prediction))

    print("Load prediction done")
    print(df.keys())

    try:
        while True:
            name = input("Please input audio name: ")

            if not name.isdigit():
                print("invalid audio name")
                continue

            pred = np.array(df[name])
            gt = pd.read_csv(str(data_dir / name / f"{name}_groundtruth.txt"), sep=" ", header=None).values

            fig, ax = plt.subplots()
            ax.set_ylim(0, 2)
            ax.plot(pred[:,0], pred[:,1], color="blue")
            ax.set_xlabel("time (sec)")
            ax.set_ylabel("prediction (confidence)")
            ax.set_title(name)

            ax2 = ax.twinx()
            ax2.set_ylim(50, 80)

            segments = [[(t[0], t[2]), (t[1], t[2])] for t in gt]
            lines = LineCollection(segments, linestyle="solid", colors=["orange" for _ in range(len(segments))], linewidths=[2 for _ in range(len(segments))])
            ax2.add_collection(lines)

            fig.tight_layout()
            fig.show()
            plt.show()

    except KeyboardInterrupt as e:
        sys.exit(0)

    except Exception as e:
        print(e)

if __name__ == '__main__':
    args = get_args()
    print(args)

    main(args)