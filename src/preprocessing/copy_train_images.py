import argparse
import shutil
import os

from tqdm import tqdm
import skimage as ski

from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from helpers.sys_helpers import create_dir

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--source", "-s", type=str, required=True, help="Path to the source directory containing training images.")
	parser.add_argument("--output", "-o", type=str, required=True, help="Path to the output directory where images will be copied.")
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
		shutil.copyfile(os.path.join(args.source, scene, "transforms_test.json"), os.path.join(args.output, scene, "transforms_test.json"))

		files = sorted(os.listdir(source_path))

		for file in tqdm(files, desc=f"Copying {scene}", total=len(files)):
			shutil.copyfile(os.path.join(source_path, file), os.path.join(output_path, file))

		mask_path = os.path.join(args.source, "masks")
		for i, img_name in tqdm(enumerate(sorted(os.listdir(os.path.join(mask_path)))), desc="Copying masks", total=len(os.listdir(os.path.join(mask_path)))):
			img = ski.io.imread(os.path.join(mask_path, img_name))
			img = 255 - img
			ski.io.imsave(os.path.join(args.output, scene, "images", "dynamic_mask_" + str(i).zfill(4) + ".png"), img)