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

		stem_dir_list = [stem_dir for stem_dir in (data_dir / split).glob("*") if stem_dir.is_dir()]
		with tqdm(stem_dir_list, unit="audio") as t:
			for stem_dir in t:
				t.set_description(stem_dir.name)

				mixture_path = (stem_dir / "mixture.wav").absolute()
				vocal_path = (stem_dir / "vocal.wav").absolute()
				accompaniment_path = (stem_dir / "accompaniment.wav").absolute()

				si, _ = torchaudio.info(str(mixture_path))
				audio_length = si.length

				assert audio_length == torchaudio.info(str(vocal_path))[0].length and audio_length == torchaudio.info(str(accompaniment_path))[0].length

				data_dict[stem_dir.name] = {
					"mixture": str(mixture_path),
					"vocal": str(vocal_path),
					"accompaniment": str(accompaniment_path),
					"length": audio_length,
				}

		json.dump(data_dict, (output_dir / f"{split}.json").open("w"))

if __name__ == '__main__':
	args = get_args()
	main(args)