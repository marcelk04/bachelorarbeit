import argparse
import os
import json
import math

import numpy as np
import mitsuba as mi
from tqdm import tqdm

from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from helpers.math_helpers import fov_to_focal, spherical_to_cartesian, sharpness, golden_spiral, closest_point_2_lines

def output_transforms(scene, scene_path, radius, thetas, phis, num_train_cams, name="transforms.json", start_from=0):
	params = mi.traverse(scene)

	width, height = params["sensor.film.size"]
	principal_point_x = width / 2 + params["sensor.principal_point_offset_x"][0]
	principal_point_y = height / 2 + params["sensor.principal_point_offset_y"][0]
	focal_length = fov_to_focal(params["sensor.x_fov"][0], width)

	transforms = {}
	transforms["w"] = width
	transforms["h"] = height
	transforms["fl_x"] = focal_length
	transforms["fl_y"] = focal_length
	transforms["cx"] = principal_point_x
	transforms["cy"] = principal_point_y
	transforms["k1"] = 0.0
	transforms["k2"] = 0.0
	transforms["p1"] = 0.0
	transforms["p2"] = 0.0
	transforms["camera_angle_x"] = math.atan(width / (2 * focal_length)) * 2
	transforms["camera_angle_y"] = math.atan(height / (2 * focal_length)) * 2

	transforms["aabb_scale"] = 1
	transforms["frames"] = []

	for i, theta, phi in tqdm(zip(range(num_train_cams), thetas, phis), desc="Writing", total=num_train_cams):
		t2 = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

		camera_position = spherical_to_cartesian(radius, theta, phi)
		transform = mi.Transform4f().look_at(origin=camera_position, target=[0, 0, 0], up=[0, 1, 0])
		# transform = to_frame @ transform
		view_matrix = transform.matrix.numpy()[..., 0]

		# view_matrix[:3,:3] = t1 @ view_matrix[:3,:3]
		view_matrix = t2 @ view_matrix
		view_matrix[:3,0] *= -1
		view_matrix[:3,2] *= -1

		# view_matrix = view_matrix_inverse(view_matrix)

		file_path = os.path.join(scene_path, "images", str(i + start_from).zfill(4) + ".png")
		b = sharpness(file_path)

		camera = {}
		camera["file_path"] = os.path.join("images", str(i + start_from).zfill(4) + ".png")
		camera["sharpness"] = b
		camera["transform_matrix"] = view_matrix

		transforms["frames"].append(camera)

	totw = 0.0
	totp = np.array([0.0, 0.0, 0.0])
	for f in transforms["frames"]:
		mf = f["transform_matrix"][0:3,:]
		for g in transforms["frames"]:
			mg = g["transform_matrix"][0:3,:]
			p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
			if w > 0.00001:
				totp += p*w
				totw += w
	if totw > 0.0:
		totp /= totw
	print(totp) # the cameras are looking at totp

	for f in transforms["frames"]:
		f["transform_matrix"] = f["transform_matrix"].tolist()

	with open(os.path.join(scene_path, name), "w") as outfile:
		json.dump(transforms, outfile, indent=2)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--scene", "-s", type=str, required=True)
	parser.add_argument("--output", "-o", type=str, required=True)
	parser.add_argument("--resolution", "--res", "-r", default=512, type=int, required=False)
	parser.add_argument("--samples", "--spp", default=512, type=int, required=False)
	parser.add_argument("--image_count", "-c", default=64, type=int, required=False)
	args = parser.parse_args()

	# Train views
	radius = 4 # for nerf scale
	thetas, phis = golden_spiral(args.image_count)

	thetas_0 = np.array([0.35*np.pi, 0.5*np.pi, 0.65*np.pi])
	phis_0 = np.linspace(0, 2*np.pi, 4, endpoint=False)

	thetas_test, phis_test = np.meshgrid(thetas_0, phis_0)
	thetas_test = thetas_test.flatten()
	phis_test = phis_test.flatten()

	print("Loading scenes...")
	mi.set_variant("cuda_ad_spectral_polarized")
	polarized_scene = mi.load_file(args.scene, res=args.resolution)
	print()

	for scene in ["unpolarized", "direct", "global"]:
		output_transforms(polarized_scene, os.path.join(args.output, scene), radius, thetas, phis, args.image_count)
		output_transforms(polarized_scene, os.path.join(args.output, scene), radius, thetas_test, phis_test, len(thetas_test), name="transforms_test.json", start_from=args.image_count)


if __name__ == "__main__":
	main()