import json
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

import torchaudio

def get_args():
	parser = ArgumentParser()
	parser.add_argument("data_dir", type=str)
	parser.add_argument("output_dir", type=str)
	parser.add_argument("--split", type=str, choices=["train", "test", ""], default="")

	return parser.parse_args()

def main(args: dict) -> None:
	splits = ["train", "test"] if not args.split else [args.split.lower()]

	data_dir = Path(args.data_dir)
	output_dir = Path(args.output_dir)


	for split in splits:
		print(f"Processing split '{split}'")

		data_dict = {}

		stem_dir_list = [stem_dir if stem_dir.is_dir() for stem_dir in (data_dir / split).glob("*")]
		with tqdm(stem_dir_list, unit="audio") as t:
			for stem_dir in t:
				t.set_description(stem_dir.name)

				mixture_path = stem_dir / "mixture.wav"
				vocal_path = stem_dir / "vocal.wav"
				accompaniment_path = stem_dir / "accompaniment.wav"

				si, _ = torchaudio.info(mixture_path)
				audio_length = si.length

				data_dict[stem_dir.name] = {
					"mixture": mixture_path,
					"vocal": vocal_path,
					"accompaniment.wav": accompaniment_path,
					"length": audio_length,
				}

		json.dump(data_dict, open(output_dir / f"{split}.json", "w"))

if __name__ == '__main__':
	args = get_args()
	main(args)