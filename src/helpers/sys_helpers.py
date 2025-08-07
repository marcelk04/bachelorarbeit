import numpy as np
import os

def create_dir(path: str) -> bool:
	if not os.path.exists(path):
		os.makedirs(path)
		return True
	
	return False

def write_lines_to_file(lines: list[str], path: str) -> None:
	with open(path, "w") as f:
		for line in lines:
			f.write(line)

def exec_cmd(cmd: str) -> None:
	print(f"Executing '{cmd}'")

	exit_code = os.system(cmd)

	if exit_code != 0:
		exit(exit_code)

	print()

def to_np_image(image):
	image = image.astype(float) / 255.0 # Convert to float range [0, 1]
	image = image ** 2.2 # Gamma correction

	return image

def to_ski_image(image):
	image[image < 0] = 0 # Clip negative values

	image = image ** (1.0 / 2.2) # Gamma correction
	image = np.clip(image, 0, 1) # Clip again
	image = np.uint8(image * 255.0) # Convert to integer range [0, 255]

	return image