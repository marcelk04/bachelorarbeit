import argparse
import os
from tqdm import tqdm
import skimage as ski
import shutil

from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from helpers.polarization_helpers import *
from helpers.sys_helpers import *

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--indirect", "-i", type=str, required=True)
	parser.add_argument("--direct", "-d", type=str, required=True)
	parser.add_argument("--output", "-o", type=str, required=True)
	parser.add_argument("--gt", default="", type=str, required=False)
	args = parser.parse_args()

	# Only needed if no gt path was passed
	if args.gt == "":
		gt_indirect = os.path.join(args.indirect, "gt")
		gt_direct = os.path.join(args.direct, "gt")

		assert os.path.exists(gt_indirect)
		assert os.path.exists(gt_direct)

	renders_indirect = os.path.join(args.indirect, "renders")
	renders_direct = os.path.join(args.direct, "renders")

	assert os.path.exists(renders_indirect)
	assert os.path.exists(renders_direct)

	gt_output = os.path.join(args.output, "gt")
	renders_output = os.path.join(args.output, "renders")

	create_dir(gt_output)
	create_dir(renders_output)

	images = os.listdir(gt_indirect)

	for image in tqdm(images, desc="Writing", total=len(images)):
		# gt
		if args.gt == "":
			indirect = to_np_image(ski.io.imread(os.path.join(gt_indirect, image)))
			direct = to_np_image(ski.io.imread(os.path.join(gt_direct, image)))

			combined = reconstruct(indirect, direct)

			ski.io.imsave(os.path.join(gt_output, image), to_ski_image(combined), check_contrast=False)
		else:
			shutil.copyfile(os.path.join(args.gt, image), os.path.join(gt_output, image))

		# renders
		indirect = to_np_image(ski.io.imread(os.path.join(renders_indirect, image)))
		direct = to_np_image(ski.io.imread(os.path.join(renders_direct, image)))

		combined = reconstruct(indirect, direct)

		ski.io.imsave(os.path.join(renders_output, image), to_ski_image(combined), check_contrast=False)

if __name__ == "__main__":
	main()