import os
from pathlib import Path
from argparse import ArgumentParser
from time import sleep

from tqdm import tqdm

def get_args():
    parser = ArgumentParser("download audio from youtube")
    parser.add_argument("data_dir")
    parser.add_argument("--cover", action="store_true")

    return parser.parse_args()

def main(opt):
    root_dir = Path(opt.data_dir)
    output_dir = root_dir / "audios"
    
    try:
        output_dir.mkdir(parents=True)
    except FileExistsError as e:
        print("Output dir already exists")

    video_list = tqdm(sorted(root_dir.glob("*")), unit="videos")

    for video_dir in video_list:
        if not video_dir.is_dir() or not (video_dir / f"{video_dir.name}.txt").exists():
            continue

        video_list.set_description(f"Downloading {video_dir.name}")

        txt_file = (video_dir / f"{video_dir.name}_link.txt")
        video_link = txt_file.read_text().strip()
        video_id = video_link[-11:]

        if opt.cover or not (output_dir / f"{video_dir.name}.{video_id}.m4a").exists():
            try:
                command = f"youtube-dl -f 'bestaudio[ext=m4a]' {video_link} -o '{output_dir / f'{video_dir.name}.%(id)s.%(ext)s'}' > /dev/null"
                os.system(command)
                sleep(0.3)
            except KeyboardInterrupt as e:
                video_list.close()
                break


    # remove part files generated while downloading
    for trash in output_dir.glob("*.part"):
        os.remove(trash)

if __name__ == '__main__':
    args = get_args()
    main(args)