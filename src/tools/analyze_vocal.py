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
            gt = pd.read_csv(str(data_dir / name / f"{name}_crepe.csv")).values

            thres = float(input("Please input confidence threshold (0 to 1)"))

            fig, ax = plt.subplots()

            points = np.array([[t[0], t[1]] if t[2] >= thres else [t[0], 27.5] for t in gt ])
            ax.scatter(points[:,0], points[:,1], s=1, c="orange")

            ax2 = ax.twinx()

            if "result" not in args.prediction:
                ax2.set_xlabel("time (sec)")
                ax2.set_ylabel("prediction (confidence)")
                ax2.set_title(name)

                ax2.set_ylim(0, 2)
                ax2.set_xlim(0, 300)
                
                t = pred[:,0]
                p = pred[:,1]

                ax2.plot(t, p, color="blue")
                # line = LineCollection([[[0, mean], [300, mean]]], linestyle="solid", colors=["red"])
                # ax2.add_collection(line)
                
            else:
                ax2.set_ylim(50, 80)
                segments = np.array([[(t[0], 0), (t[0], 100)] for t in pred])
                lines = LineCollection(segments, linestyle="solid", colors=["green"], linewidths=[1 for _ in range(len(segments))])
                ax2.add_collection(lines)
                segments = np.array([[(t[1], 0), (t[1], 100)] for t in pred])
                lines = LineCollection(segments, linestyle="solid", colors=["red"], linewidths=[1 for _ in range(len(segments))])
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