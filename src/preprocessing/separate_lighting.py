import argparse
import os

import skimage as ski
from tqdm import tqdm

from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from helpers.polarization_helpers import separate
from helpers.sys_helpers import create_dir, to_np_image, to_ski_image

def separate_images(source_path: str, direct_path: str, global_path: str) -> None:
	polarized_0_path = os.path.join(source_path, "polarized_0", "images")
	polarized_90_path = os.path.join(source_path, "polarized_90", "images")
	assert os.path.exists(polarized_0_path) and os.path.exists(polarized_90_path)
	
	for img_0_name, img_90_name in tqdm(zip(sorted(os.listdir(polarized_0_path)), sorted(os.listdir(polarized_90_path))), desc="Separating lighting", total=len(os.listdir(polarized_0_path))):
		if img_0_name != img_90_name:
			print(f"Image names do not match: {img_0_name} and {img_90_name}. Skipping these images.")
			continue

		img_0 = to_np_image(ski.io.imread(os.path.join(polarized_0_path, img_0_name)))
		img_90 = to_np_image(ski.io.imread(os.path.join(polarized_90_path, img_90_name)))

		global_img, direct_img = separate(img_0, img_90)

		ski.io.imsave(os.path.join(global_path, img_0_name), to_ski_image(global_img), check_contrast=False)
		ski.io.imsave(os.path.join(direct_path, img_0_name), to_ski_image(direct_img), check_contrast=False)

def main() -> None:
	parser = argparse.ArgumentParser(description="Separate lighting into global and direct components.")
	parser.add_argument("--source", "-s", default="", type=str, required=True, help="Path to the source directory containing the polarized images.")
	parser.add_argument("--output", "-o", default="", type=str, required=False, help="Path to the output directory where the separated images will be saved (default: SOURCE).")
	parser.add_argument("--overwrite", action="store_true", help="Overwrite existing images.")
	args = parser.parse_args()

	if args.output == "":
		args.output = args.source

	direct_path = os.path.join(args.output, "direct", "images")
	global_path = os.path.join(args.output, "global", "images")

	direct_exists = not create_dir(direct_path)
	global_exists = not create_dir(global_path)

	if direct_exists and global_exists and not args.overwrite:
		print(f"Separated images already exist in {args.output}. Use --overwrite to overwrite them.")
		return

	separate_images(args.source, direct_path, global_path)

if __name__ == "__main__":
	main()