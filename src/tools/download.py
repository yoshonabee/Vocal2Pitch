import os
from pathlib import Path
from argparse import ArgumentParser
from time import sleep

from multiprocessing import Pool

from tqdm import tqdm

def get_args():
    parser = ArgumentParser("download audio from youtube")
    parser.add_argument("data_dir")
    parser.add_argument("--cover", action="store_true")
    parser.add_argument("--concurrency", type=int, default=10)

    return parser.parse_args()

def download_audio(video_dir):
    link_file = (video_dir / f"{video_dir.name}_link.txt")
    video_link = link_file.read_text().strip()

    command = f"youtube-dl -f 'bestaudio[ext=wav]/bestaudio[ext=m4a]/bestaudio[ext=mp3]' {video_link} -o '{video_dir / f'{video_dir.name}.%(ext)s'}' > /dev/null"
    ok = os.system(command)

    if ok != 0:
        print(video_dir.name)

    return True

def main(args):
    root_dir = Path(args.data_dir)

    video_list = [
        video_dir
        for video_dir in root_dir.glob("*")
        if video_dir.is_dir() and (video_dir / f"{video_dir.name}_link.txt").exists()
    ]

    with tqdm(total=len(video_list), unit="audios") as t:
        p = Pool(args.concurrency)
        for ok in p.imap(download_audio, video_list):
            t.update(1)
if __name__ == '__main__':
    args = get_args()
    main(args)
