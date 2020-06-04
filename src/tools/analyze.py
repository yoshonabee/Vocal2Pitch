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

            if name == "exit":
                break

            if not name.isdigit():
                print("invalid audio name")
                continue

            pred = np.array(df[name])
            gt = pd.read_csv(str(data_dir / name / f"{name}_groundtruth.txt"), sep=" ", header=None).values

            fig, ax = plt.subplots()

            ax.set_ylim(50, 80)
            ax.set_xlim(0, 300)

            segments = [[(t[0], t[2]), (t[1], t[2])] for t in gt]
            lines = LineCollection(segments, linestyle="solid", colors=["orange" for _ in range(len(segments))], linewidths=[2 for _ in range(len(segments))])
            ax.add_collection(lines)

            ax2 = ax.twinx()

            if "result" not in args.prediction:
                ax2.set_xlabel("time (sec)")
                ax2.set_ylabel("prediction (confidence)")
                ax2.set_title(name)

                ax2.set_ylim(0, 2)
                ax2.set_xlim(0, 300)
                
                t = pred[:,0]
                p = pred[:,1]
                
                mean = p.mean()
                std = p.std()
                p = (p - mean) / std
                p = (p - p.min())
                p = p / p.max()

                print(p.min(), p.max())

                print(mean, std)

                ax2.plot(t, p, color="blue")
                # line = LineCollection([[[0, mean], [300, mean]]], linestyle="solid", colors=["red"])
                # ax2.add_collection(line)
                
            else:
                ax2.set_ylim(50, 80)
                points = np.array([(t[0], t[2]) for t in pred])
                ax2.scatter(points[:,0], points[:,1], color="blue", s=1)

                points = np.array([(t[1], t[2]) for t in pred])
                ax2.scatter(points[:,0], points[:,1], color="green", s=1)

            
            

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