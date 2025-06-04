import mitsuba as mi
import skimage as ski
import numpy as np
import argparse
import os
import json

import sys
sys.path.append("..")

from helpers.polarization_helpers import *
from helpers.render_helpers import *

def to_ski_image(image):
	image = image.numpy() # Convert to numpy array
	image[image < 0] = 0 # Clip negative values

	image = image ** (1.0 / 2.2) # Gamma correction
	image = np.clip(image, 0, 1) # Clip again
	image = np.uint8(image * 255.0) # Convert to integer range [0, 255]

	return image

def view_matrix_inverse(view_matrix):
	R = view_matrix[:3, :3]
	T = view_matrix[:3, 3]

	R_inv = R.T
	T_inv = -np.dot(R_inv, T)

	view_matrix[:3, :3] = R_inv
	view_matrix[:3, 3] = T_inv

	return view_matrix

def fov_to_focal(fov, w):
	return w / (2 * np.tan(np.deg2rad(fov) / 2))

def render_masks(scene_path, output_path, radius, thetas, phis):
	os.makedirs(os.path.join(output_path, "masks"), exist_ok=True)
	mi.set_variant("cuda_ad_rgb")

def render_unpolarized_images(scene_path, output_path, radius, thetas, phis):
	os.makedirs(os.path.join(output_path, "unpolarized"), exist_ok=True)
	mi.set_variant("cuda_ad_spectral")

	print("Loading scene")
	scene = mi.load_file(scene_path)

	print("Rendering unpolarized images")
	images = render_from_angles(scene, radius, thetas, phis, polarized=False)

	print("Saving")
	for i, image in enumerate(images):
		ski.io.imsave(os.path.join(output_path, "unpolarized", str(i).zfill(4) + ".png"), to_ski_image(image), check_contrast=False)
	
	print()

def render_polarized_images(scene_path, output_path, radius, thetas, phis):
	os.makedirs(os.path.join(output_path, "polarized_0"), exist_ok=True)
	os.makedirs(os.path.join(output_path, "polarized_90"), exist_ok=True)
	mi.set_variant("cuda_ad_spectral_polarized")

	print("Loading scene")
	scene = mi.load_file(scene_path)

	print("Rendering polarized images")
	images = render_from_angles(scene, radius, thetas, phis, polarized=True)

	print("Saving")
	for i, res in enumerate(images):
		img_0, img_90 = res

		ski.io.imsave(os.path.join(output_path, "polarized_0", str(i).zfill(4) + ".png"), to_ski_image(img_0), check_contrast=False)
		ski.io.imsave(os.path.join(output_path, "polarized_90", str(i).zfill(4) + ".png"), to_ski_image(img_90), check_contrast=False)

	print()

def output_camera_calibration(scene_path, output_path, radius, thetas, phis):
	print("Outputting camera poses")
	mi.set_variant("cuda_ad_rgb")
	scene = mi.load_file(scene_path)
	params = mi.traverse(scene)

	width, height = params["sensor.film.size"]
	principal_point_x = width / 2 + params["sensor.principal_point_offset_x"][0]
	principal_point_y = height / 2 + params["sensor.principal_point_offset_y"][0]
	focal_length = fov_to_focal(params["sensor.x_fov"][0], width)

	cameras = []

	for i, theta, phi in zip(range(len(thetas)), thetas, phis):
		camera_position = spherical_to_cartesian(radius, theta, phi)
		transform = mi.Transform4f().look_at(origin=camera_position, target=[0, 0, 0], up=[0, -1, 0])
		view_matrix = transform.matrix.numpy()[..., 0]

		# R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
		# view_matrix[:3, :3] = R @ view_matrix[:3, :3]
		# view_matrix[:3, 3] = R @ view_matrix[:3, 3]

		view_matrix = view_matrix_inverse(view_matrix)

		extr_obj = {}
		extr_obj["view_matrix"] = view_matrix.flatten().tolist()

		cam_matrix = np.identity(3)
		# cam_matrix[0, 0] = focal_length
		# cam_matrix[1, 1] = focal_length
		cam_matrix[0, 0] = 1228.8
		cam_matrix[1, 1] = 1228.8
		cam_matrix[0, 2] = principal_point_x
		cam_matrix[1, 2] = principal_point_y

		intr_obj = {}
		intr_obj["camera_matrix"] = cam_matrix.flatten().tolist()
		intr_obj["resolution"] = [width, height]

		cam_obj = {}
		cam_obj["camera_id"] = str(i).zfill(4) + ".png"
		cam_obj["extrinsics"] = extr_obj
		cam_obj["intrinsics"] = intr_obj
		
		cameras.append(cam_obj)

	poses = {}
	poses["cameras"] = cameras

	json_object = json.dumps(poses, indent=2)

	with open(os.path.join(output_path, "poses.json"), "w") as outfile:
		outfile.write(json_object)

	print()


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--scene", "-s", type=str)
	parser.add_argument("--output", "-o", type=str)
	args = parser.parse_args()

	assert os.path.exists(args.scene)
	
	if not os.path.exists(args.output):
		os.makedirs(args.output)

	thetas_0 = np.array([0.25*np.pi, 0.5*np.pi, 0.75*np.pi])
	phis_0 = np.linspace(0, 2*np.pi, 16, endpoint=False)

	thetas, phis = np.meshgrid(thetas_0, phis_0)
	thetas = thetas.flatten()
	phis = phis.flatten()

	radius = 4

	# render_unpolarized_images(args.scene, args.output, radius, thetas, phis)
	# render_polarized_images(args.scene, args.output, radius, thetas, phis)

	output_camera_calibration(args.scene, args.output, radius, thetas, phis)

if __name__ == "__main__":
	main()