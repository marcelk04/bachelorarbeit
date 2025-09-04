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
from helpers.math_helpers import rotmat2qvec
from helpers.sys_helpers import create_dir, exec_cmd, write_lines_to_file

def extract_poses(calibration_file: str, output_path: str, include_test_cams: bool) -> None:
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
	sparse_path = os.path.join(output_path, "sparse", "0")
	create_dir(sparse_path)
	db_path = os.path.join(distorted_path, "database.db")

	db = COLMAPDatabase.connect(db_path)
	db.create_tables()

	imagetxt_list = []
	cameratxt_list = []
	test_cams = []

	print(f"Start writing to new database at '{db_path}'")

	for i, camera in tqdm(enumerate(calibration["cameras"]), desc="Reading camera calibration", total=len(calibration['cameras'])):
		camera_name = camera["camera_id"]
		image_name = camera_name

		# TODO: Test cameras should not be used for reconstruction
		if camera["is_test_cam"]:
			test_cams.append(f"{camera_name}\n")

			if not include_test_cams:
				continue

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
	write_lines_to_file(test_cams, os.path.join(sparse_path, "test.txt"))

	print("Done writing text output")
	print()

def run_colmap(image_source: str, mask_source: str, output_path: str) -> None:
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

def reconstruct(image_path: str, output_path: str, calibration_path: str, mask_path: str, include_test_cams: bool) -> None:
	if os.path.exists(output_path):
		print(f"Removing old reconstruction for {output_path}")
		shutil.rmtree(output_path)

	print(f"Starting reconstruction for {image_path}...")
	print()

	extract_poses(calibration_path, output_path, include_test_cams)
	run_colmap(image_path, mask_path, output_path)


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--source", "-s", default="", type=str, required=True, help="Path to the source images.")
	parser.add_argument("--output", "-o", default="", type=str, required=True, help="Path to the output directory (default: SOURCE).")
	parser.add_argument("--calibration_path", "-c", default="", type=str, required=False, help="Path to the calibration file (default: SOURCE/poses.json).")
	parser.add_argument("--mask_path", "-m", default="", type=str, required=False, help="Path to the alpha masks. If no path is passed, the program will look in SOURCE/masks.")
	parser.add_argument("--include_test_cams", action="store_true", help="Include test cameras in the reconstruction.")
	args = parser.parse_args()

	# Make sure image input exists
	if args.output == "":
		args.output = args.source
	assert os.path.exists(args.source)

	# Make sure the calibration file exists
	if args.calibration_path == "":
		args.calibration_path = os.path.join(args.source, "poses.json")
	assert os.path.exists(args.calibration_path)

	# Search for alpha masks
	if args.mask_path == "":
		args.mask_path = os.path.join(args.source, "masks")
	if not os.path.exists(args.mask_path):
		args.mask_path = None

	# Make sure all scenes exist
	scenes = ["unpolarized", "global", "direct"]
	scene_sources = [os.path.join(args.source, scene, "images") for scene in scenes]
	scene_outputs = [os.path.join(args.output, scene, "colmap") for scene in scenes]

	for scene in scene_sources:
		assert os.path.exists(scene)

	# Run COLMAP reconstruction for all scenes
	for scene_src, scene_dst in zip(scene_sources, scene_outputs):
		reconstruct(scene_src, scene_dst, args.calibration_path, args.mask_path, args.include_test_cams)

if __name__ == "__main__":
	main()
