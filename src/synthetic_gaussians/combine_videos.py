import argparse
import numpy as np
import cv2
import os
from tqdm import tqdm

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
	args = parser.parse_args()

	# Make sure output directory exists
	create_dir(os.path.dirname(args.output))

	assert os.path.exists(args.indirect)
	assert os.path.exists(args.direct)

	indirect_stream = cv2.VideoCapture(args.indirect)
	direct_stream = cv2.VideoCapture(args.direct)

	width = int(indirect_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(indirect_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = indirect_stream.get(cv2.CAP_PROP_FPS)
	frame_count = int(indirect_stream.get(cv2.CAP_PROP_FRAME_COUNT))

	assert width == int(direct_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
	assert height == int(direct_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

	print(f"Width: {width}, Height: {height}")
	print(f"FPS: {fps}")
	print(f"Frame count: {frame_count}")

	out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (height, width))

	for i in tqdm(range(frame_count), desc="Writing", total=frame_count):
		try:
			_, indirect = indirect_stream.read()
			_, direct = direct_stream.read()

			indirect = cv2.cvtColor(indirect, cv2.COLOR_BGR2RGB)
			direct = cv2.cvtColor(direct, cv2.COLOR_BGR2RGB)

			indirect = indirect.astype(np.float64) / 255.0
			direct = direct.astype(np.float64) / 255.0

			indirect = indirect ** 2.2
			direct = direct ** 2.2

			combined = reconstruct(indirect, direct)

			combined = combined ** (1.0 / 2.2) # gamma correction
			combined = (np.clip(combined, 0, 1) * 255).astype(np.uint8) # continuous range [0, 1] to discrete range [0, 255]
			combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

			out.write(combined)

		except:
			print(f"Encountered an error at frame {i} :(")
			break

	indirect_stream.release()
	direct_stream.release()
	out.release()

	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()