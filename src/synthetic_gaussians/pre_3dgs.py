import argparse
import os
import shutil
import json
import numpy as np
import skimage as ski
from tqdm import tqdm

# dont question this ... :/
from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from thirdparty.database import *
from helpers.math_helpers import *
from helpers.sys_helpers import *
from helpers.polarization_helpers import *

def separate_lighting(scenes: list[str], output: str) -> None:
	indirect_path = os.path.join(output, "indirect", "images")
	direct_path = os.path.join(output, "direct", "images")

	create_dir(indirect_path)
	create_dir(direct_path)

	images = sorted(os.listdir(os.path.join(scenes[0], "images")))

	for img in tqdm(images, desc="Separating lighting", total=len(images)):
		img_0 = (ski.io.imread(os.path.join(scenes[0], "images", img)) / 255.0) ** 2.2
		img_90 = (ski.io.imread(os.path.join(scenes[1], "images", img)) / 255.0) ** 2.2

		indirect, direct = separate(img_0, img_90)

		ski.io.imsave(os.path.join(indirect_path, img), to_ski_image(indirect), check_contrast=False)
		ski.io.imsave(os.path.join(direct_path, img), to_ski_image(direct), check_contrast=False)

def extract_poses(calibration_file: str, output_path: str) -> None:
	# Set camera model
	camera_model = 1 # PINHOLE

	# Load calibration file
	assert os.path.exists(calibration_file)
	file = open(calibration_file)
	calibration = json.load(file)
	file.close()

	# Create necessary directories
	create_dir(output_path)
	distorted_path = os.path.join(output_path, "distorted")
	create_dir(distorted_path)
	manual_path = os.path.join(output_path, "manual")
	create_dir(manual_path)
	db_path = os.path.join(distorted_path, "database.db")

	db = COLMAPDatabase.connect(db_path)
	db.create_tables()

	imagetxt_list = []
	cameratxt_list = []

	print(f"Start writing to new database at '{db_path}'")

	for i, camera in tqdm(enumerate(calibration["cameras"]), desc="Reading camera calibration", total=len(calibration['cameras'])):
		camera_name = camera["camera_id"]
		image_name = camera_name

		# Extract pose information from the JSON file
		view_matrix = np.array(camera["extrinsics"]["view_matrix"], dtype=np.float64).reshape((4, 4)) # View Matrix is given in World-To-Camera Space 

		camera_matrix = np.array(camera["intrinsics"]["camera_matrix"], dtype=np.float64).reshape((3, 3))

		# Camera rotation and translation
		R = view_matrix[:3, :3]
		T = view_matrix[:3, 3]
		Q = rotmat2qvec(R)

		# focal length
		f_x = camera_matrix[0, 0]
		f_y = camera_matrix[1, 1]

		# principal point
		c_x = camera_matrix[0, 2]
		c_y = camera_matrix[1, 2]
		
		# image size
		width = int(camera["intrinsics"]["resolution"][0])
		height = int(camera["intrinsics"]["resolution"][1])

		params = np.array([f_x, f_y, c_x, c_y])

		camera_id = db.add_camera(camera_model, width, height, params)

		# image_id = db.add_image(image_name, camera_id, Q, T, image_id=i+1) # For COLMAP <= 3.9
		image_id = db.add_image(image_name, camera_id, image_id=i+1) # For COLMAP >= 3.10

		db.commit()

		# Append lines for images.txt and cameras.txt
		Q_string = " ".join([str(q) for q in Q])
		T_string = " ".join([str(t) for t in T])
		params_string = " ".join([str(num) for num in params])

		image_line = f"{image_id} {Q_string} {T_string} {camera_id} {image_name}\n"
		imagetxt_list.append(image_line)
		imagetxt_list.append("\n")

		camera_line = f"{camera_id} PINHOLE {width} {height} {params_string}\n"
		cameratxt_list.append(camera_line)

	db.close()

	print("Done writing to database")
	
	# Write prepared data into images.txt and cameras.txt
	write_lines_to_file(imagetxt_list, os.path.join(manual_path, "images.txt"))
	write_lines_to_file(cameratxt_list, os.path.join(manual_path, "cameras.txt"))
	write_lines_to_file([], os.path.join(manual_path, "points3D.txt"))

	print("Done writing text output")
	print()

def run_colmap(image_source: str, mask_source: str, output_path: str):
	manual_path = os.path.join(output_path, "manual")
	# undistorted_path = os.path.join(output_path, "images")
	distorted_path = os.path.join(output_path, "distorted")
	db_path = os.path.join(distorted_path, "database.db")
	sparse_path = os.path.join(distorted_path, "sparse", "0")

	create_dir(sparse_path)

	print("Starting sparse reconstruction...")
	print()

	feature_extract = f"colmap feature_extractor \
		--database_path {db_path} \
		--image_path {image_source} \
		--SiftExtraction.num_threads=16 \
		--SiftExtraction.estimate_affine_shape=true \
		--SiftExtraction.domain_size_pooling=true" # --SiftExtraction.estimate_affine_shape=true  --SiftExtraction.domain_size_pooling=true 
	if type(mask_source) != type(None):
		feature_extract += f" --ImageReader.mask_path {mask_source}"
	exec_cmd(feature_extract)

	feature_matcher = f"colmap exhaustive_matcher \
		--database_path {db_path} \
		--SiftMatching.guided_matching=true \
		--SiftMatching.max_ratio=0.9" # --SiftMatching.guided_matching=true
	exec_cmd(feature_matcher)

	tri_and_map = f"colmap point_triangulator \
		--database_path {db_path} \
		--image_path {image_source} \
		--input_path {manual_path} \
		--output_path {sparse_path} \
		--Mapper.ba_global_function_tolerance=0.000001" # --Mapper.ba_global_function_tolerance=0.000001
	exec_cmd(tri_and_map)

	image_undistortion = f"colmap image_undistorter \
		--image_path {image_source} \
		--input_path {sparse_path} \
		--output_path {output_path}"
	exec_cmd(image_undistortion)

	print("Sparse reconstruction done.")
	print()

	sparse = os.path.join(output_path, "sparse")
	sparse0 = os.path.join(output_path, "sparse", "0")

	files = os.listdir(sparse)
	create_dir(sparse0)

	# Copy each file from the source directory to the destination directory (required by 3DGS)
	for file in files:
		if file == '0':
			continue

		source_file = os.path.join(sparse, file)
		destination_file = os.path.join(sparse0, file)
		shutil.move(source_file, destination_file)


def reconstruct(scene_path: str, calibration_path: str, mask_path: str) -> None:
	input_path = os.path.join(scene_path, "images")
	output_path = os.path.join(scene_path, "colmap")

	if os.path.exists(output_path):
		print(f"Removing old reconstruction for {scene_path}")
		shutil.rmtree(output_path)

	print(f"Starting reconstruction for {scene_path}...")
	print()

	extract_poses(calibration_path, output_path)
	run_colmap(input_path, mask_path, output_path)


def main():
	parser = argparse.ArgumentParser(prog="python pre_vci.py")
	parser.add_argument("--workspace", "-w", type=str, required=True)
	parser.add_argument("--skip_polarized", action="store_true")
	parser.add_argument("--skip_unpolarized", action="store_true")
	args = parser.parse_args()

	# assert workspace exists
	assert os.path.exists(args.workspace)

	# assert calibration file exists
	calibration_path = os.path.join(args.workspace, "poses.json")
	assert os.path.exists(calibration_path)

	# look for masks
	mask_path = os.path.join(args.workspace, "masks")
	if not os.path.exists(mask_path):
		mask_path = None

	# look for unpolarized scene
	unpolarized_scene = os.path.join(args.workspace, "unpolarized")
	if args.skip_unpolarized:
		unpolarized_scene = None
	elif not os.path.exists(unpolarized_scene):
		print("Unpolarized scene not found. Skipping this one.")
		unpolarized_scene = None

	# look for polarized scenes
	polarized_scenes = sorted([os.path.join(args.workspace, elem) for elem in os.listdir(args.workspace) if elem.startswith("polarized_")])
	if args.skip_polarized:
		polarized_scenes = None
	elif len(polarized_scenes) < 2:
		print("Not enough polarized scenes found. Skipping this one.")
		polarized_scenes = None
	else:
		print(f"Found polarized scenes: {polarized_scenes}")

	# reconstruct unpolarized scene
	if type(unpolarized_scene) != type(None):
		reconstruct(unpolarized_scene, calibration_path, mask_path)

	# separate polarized lighting and reconstruct scenes
	if type(polarized_scenes) != type(None):
		separate_lighting(polarized_scenes, args.workspace)

		indirect_path = os.path.join(args.workspace, "indirect")
		reconstruct(indirect_path, calibration_path, mask_path)

		direct_path = os.path.join(args.workspace, "direct")
		reconstruct(direct_path, calibration_path, mask_path)

if __name__ == "__main__":
	main()
