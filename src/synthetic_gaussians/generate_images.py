import mitsuba as mi
import drjit as dr
import skimage as ski
import numpy as np
import argparse
import os
import json
from tqdm import tqdm

from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from helpers.polarization_helpers import *
from helpers.render_helpers import *
from helpers.math_helpers import *
from helpers.sys_helpers import *


def render_masks(scene, output_path, radius, thetas, phis):
	os.makedirs(os.path.join(output_path, "masks"), exist_ok=True)
	print("Generating alpha masks...")

	params = mi.traverse(scene)

	# Save old transforms
	polarizer_light_transform = mi.Transform4f(params["polarizer_light.to_world"])
	polarizer_cam_transform = mi.Transform4f(params["polarizer_cam.to_world"])

	# Move polarizers away
	params["polarizer_light.to_world"] = mi.Transform4f().translate([100, 100, 100])
	params["polarizer_cam.to_world"] = mi.Transform4f().translate([100, 100, 100])
	params.update()

	positions = np.array(params["head.vertex_positions"]).reshape(3, -1)

	dist = np.linalg.norm(positions, ord=2, axis=0)

	threshold = radius + dist.max()

	integrator = mi.load_dict({
		"type": "depth"
	})

	images = render_from_angles(scene, radius, thetas, phis, polarized=False, integrator=integrator)
	images = np.average(images, axis=-1) # convert to grayscale

	masks = np.where(images > threshold, 0.0, 1.0)

	# Undo changes to scenes
	params["polarizer_light.to_world"] = polarizer_light_transform
	params["polarizer_cam.to_world"] = polarizer_cam_transform
	params.update()

	return masks

def render_unpolarized_images(scene, output_path, radius, thetas, phis, spp):
	print("Generating unpolarized images...")
	os.makedirs(os.path.join(output_path, "unpolarized", "images"), exist_ok=True)

	params = mi.traverse(scene)

	# Save old transforms
	polarizer_cam_transform = mi.Transform4f(params["polarizer_cam.to_world"])

	# Move polarizers away
	params["polarizer_cam.to_world"] = mi.Transform4f().translate([100, 100, 100])
	params.update()

	images = render_from_angles(scene, radius, thetas, phis, polarized=False, spp=spp)
	
	# Undo changes to scenes
	params["polarizer_cam.to_world"] = polarizer_cam_transform
	params.update()

	return images

def render_polarized_images(scene, output_path, radius, thetas, phis, spp):
	print("Generating polarized images...")
	os.makedirs(os.path.join(output_path, "polarized_0", "images"), exist_ok=True)
	os.makedirs(os.path.join(output_path, "polarized_90", "images"), exist_ok=True)

	images = render_from_angles(scene, radius, thetas, phis, polarized=True, spp=spp)

	return images

def mask_images(images, masks):
	return images * masks[..., None]

def save_images(images, path, extension=".png"):
	N = images.shape[0]
	os.makedirs(os.path.join(path, "images"), exist_ok=True)

	for i in tqdm(range(N), desc="Saving", total=N):
		output = os.path.join(path, "images", str(i).zfill(4) + extension)
		image = np.squeeze(images[i])

		ski.io.imsave(output, to_ski_image(image), check_contrast=False)

def output_camera_calibration(scene, output_path, radius, thetas, phis):
	print("Generating camera poses...")
	
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

		view_matrix = view_matrix_inverse(view_matrix)

		extr_obj = {}
		extr_obj["view_matrix"] = view_matrix.flatten().tolist()

		cam_matrix = np.identity(3)
		cam_matrix[0, 0] = focal_length
		cam_matrix[1, 1] = focal_length
		# cam_matrix[0, 0] = 1228.8
		# cam_matrix[1, 1] = 1228.8
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
	parser.add_argument("--scene", "-s", type=str, required=True)
	parser.add_argument("--output", "-o", type=str, required=True)
	parser.add_argument("--resolution", "-r", default=512, type=int, required=False)
	parser.add_argument("--samples", "--spp", default=512, type=int, required=False)
	args = parser.parse_args()

	assert os.path.exists(args.scene)
	
	if not os.path.exists(args.output):
		os.makedirs(args.output)

	thetas_0 = np.array([0.35*np.pi, 0.5*np.pi, 0.65*np.pi])
	phis_0 = np.linspace(0, 2*np.pi, 16, endpoint=False)

	thetas, phis = np.meshgrid(thetas_0, phis_0)
	thetas = thetas.flatten()
	phis = phis.flatten()

	radius = 75

	print("Loading scenes...")
	dr.set_flag(dr.JitFlag.Debug, True)
	mi.set_variant("cuda_ad_spectral_polarized")
	polarized_scene = mi.load_file(args.scene, res=args.resolution)
	unpolarized_scene = mi.load_file(args.scene, res=args.resolution, polarizing=False)
	print("Done.")
	print()

	masks = render_masks(polarized_scene, args.output, radius, thetas, phis)
	save_images(masks, os.path.join(args.output, "masks"), extension=".png.png")

	unpolarized_images = render_unpolarized_images(unpolarized_scene, args.output, radius, thetas, phis, args.samples)
	unpolarized_images = mask_images(unpolarized_images, masks)
	save_images(unpolarized_images, os.path.join(args.output, "unpolarized"))

	polarized_images = render_polarized_images(polarized_scene, args.output, radius, thetas, phis, args.samples)
	polarized_images = mask_images(polarized_images, masks)
	save_images(polarized_images[:, 0, ...], os.path.join(args.output, "polarized_0"))
	save_images(polarized_images[:, 1, ...], os.path.join(args.output, "polarized_90"))

	output_camera_calibration(polarized_scene, args.output, radius, thetas, phis)

if __name__ == "__main__":
	main()