import argparse
import os
import shutil

import skimage as ski

from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from helpers.sys_helpers import to_np_image, to_ski_image, create_dir

def mask_and_copy_renders(path, scene):
	render_path = os.path.join(path, "results", scene, "test", "ours_30000", "renders")
	mask_path = os.path.join(path, scene, "images")

	assert os.path.exists(render_path)
	assert os.path.exists(mask_path)

	for i, img_name in enumerate(sorted(os.listdir(render_path))):
		img = to_np_image(ski.io.imread(os.path.join(render_path, img_name)))
		
		mask = to_np_image(ski.io.imread(os.path.join(mask_path, "dynamic_mask_" + img_name), as_gray=True))
		mask = 1.0 - mask
		mask = ski.filters.gaussian(mask, sigma=3)

		img[..., :3] *= mask[..., None]
		ski.io.imsave(os.path.join(render_path, img_name), to_ski_image(img))

def copy_gt_images(path, scene):
	image_path = os.path.join(path, scene, "images")
	output_path = os.path.join(path, "results", scene, "test", "ours_30000", "gt")
	render_path = os.path.join(path, "results", scene, "test", "ours_30000", "renders")

	assert os.path.exists(image_path)
	create_dir(output_path)

	for i, img_name in enumerate(sorted(os.listdir(render_path))):
		shutil.copyfile(os.path.join(image_path, img_name), os.path.join(output_path, img_name))

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--source", "-s", type=str, required=True)
	args = parser.parse_args()

	for scene in ["unpolarized", "global", "direct"]:
		mask_and_copy_renders(args.source, scene)
		copy_gt_images(args.source, scene)



if __name__ == "__main__":
	main()