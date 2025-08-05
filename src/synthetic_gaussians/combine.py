import argparse
import numpy as np
import cv2

from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from helpers.polarization_helpers import *

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--indirect", "-i", type=str, required=True)
	parser.add_argument("--direct", "-d", type=str, required=True)
	parser.add_argument("--output", "-o", type=str, required=True)
	args = parser.parse_args()

	indirect_stream = cv2.VideoCapture(args.indirect)
	direct_stream = cv2.VideoCapture(args.direct)

	out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), 30, (1024, 1024))

	# _, frame = indirect_stream.read()

	# print(frame.dtype)

	frame_count = 0

	while True:
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

			combined = combined ** (1.0 / 2.2)

			combined = (np.clip(combined, 0, 1) * 255).astype(np.uint8)

			combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

			out.write(combined)

			frame_count += 1

		except:
			break

	print(f"frames: {frame_count}")

	indirect_stream.release()
	direct_stream.release()
	out.release()

	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()