import argparse
import shutil
import os

from tqdm import tqdm

from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from helpers.sys_helpers import create_dir

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--source", "-s", type=str, required=True, help="Path to the source directory containing training images.")
	parser.add_argument("--output", "-o", type=str, required=True, help="Path to the output directory where images will be copied.")
	parser.add_argument("--num_test_images", default=12, type=int, help="Number of test images to exclude from copying (default: 12).")
	args = parser.parse_args()

	assert os.path.exists(args.source)

	create_dir(args.output)


	scenes = ["unpolarized", "global", "direct"]

	for scene in scenes:
		if not os.path.exists(os.path.join(args.source, scene)):
			continue

		source_path = os.path.join(args.source, scene, "images")
		output_path = os.path.join(args.output, scene, "images")
		create_dir(output_path)

		shutil.copyfile(os.path.join(args.source, scene, "transforms.json"), os.path.join(args.output, scene, "transforms.json"))

		files = sorted(os.listdir(source_path))
		train_files = files[:-args.num_test_images]

		for file in tqdm(train_files, desc=f"Copying {scene}", total=len(train_files)):
			shutil.copyfile(os.path.join(source_path, file), os.path.join(output_path, file))