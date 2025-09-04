import mitsuba as mi
import drjit as dr
import skimage as ski
import numpy as np
import argparse
import os
import json
import math
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from helpers.polarization_helpers import *
from helpers.render_helpers import *
from helpers.math_helpers import *
from helpers.sys_helpers import *


def render_masks(scene, radius, thetas, phis):
	params = mi.traverse(scene)

	# Save old transforms
	polarizer_light_transform = mi.Transform4f(params["polarizer_light.to_world"])
	polarizer_cam_transform = mi.Transform4f(params["polarizer_cam.to_world"])

	# Move polarizers away
	params["polarizer_light.to_world"] = mi.Transform4f().translate([0, 10000, 0])
	params["polarizer_cam.to_world"] = mi.Transform4f().translate([0, 10000, 0])
	params.update()

	# Calculate maximum distance of vertices to camera
	head_positions = np.array(params["head.vertex_positions"]).reshape(3, -1)
	hair_positions = np.array(params["hair.control_points"]).reshape(4, -1)[:3, :]

	head_dist = np.linalg.norm(head_positions, ord=2, axis=0)
	hair_dist = np.linalg.norm(hair_positions, ord=2, axis=0)

	threshold = radius + max(head_dist.max(), hair_dist.max())

	integrator = mi.load_dict({
		"type": "depth"
	})

	images = render_from_angles(scene, radius, thetas, phis, polarized=False, integrator=integrator)
	images = np.average(images, axis=-1) # convert to grayscale

	masks = gaussian_filter(np.where(images > threshold, 0.0, 1.0), sigma=3, axes=(-1,-2))

	# Undo changes to scenes
	params["polarizer_light.to_world"] = polarizer_light_transform
	params["polarizer_cam.to_world"] = polarizer_cam_transform
	params.update()

	return masks

def render_unpolarized_images(scene, radius, thetas, phis, spp):
	params = mi.traverse(scene)

	# Save old transforms
	polarizer_cam_transform = mi.Transform4f(params["polarizer_cam.to_world"])

	# Move polarizers away
	params["polarizer_cam.to_world"] = mi.Transform4f().translate([0, 10000, 0])
	params.update()

	images = render_from_angles(scene, radius, thetas, phis, polarized=False, spp=spp)
	
	# Undo changes to scenes
	params["polarizer_cam.to_world"] = polarizer_cam_transform
	params.update()

	return images

def render_polarized_images(scene, radius, thetas, phis, spp):
	return render_from_angles(scene, radius, thetas, phis, polarized=True, spp=spp)

def mask_images(images, masks):
	return images * masks[..., None]

def save_images(images, path, extension=".png"):
	N = images.shape[0]
	os.makedirs(path, exist_ok=True)

	for i in tqdm(range(N), desc="Saving", total=N):
		output = os.path.join(path, str(i).zfill(4) + extension)
		image = np.squeeze(images[i])

		ski.io.imsave(output, to_ski_image(image), check_contrast=False)

def output_poses(scene, output_path, radius, thetas, phis, num_train_cams):
	params = mi.traverse(scene)

	width, height = params["sensor.film.size"]
	principal_point_x = width / 2 + params["sensor.principal_point_offset_x"][0]
	principal_point_y = height / 2 + params["sensor.principal_point_offset_y"][0]
	focal_length = fov_to_focal(params["sensor.x_fov"][0], width)

	cameras = []

	for i, theta, phi in tqdm(zip(range(len(thetas)), thetas, phis), desc="Writing", total=len(thetas)):
		camera_position = spherical_to_cartesian(radius, theta, phi)
		transform = mi.Transform4f().look_at(origin=camera_position, target=[0, 0, 0], up=[0, -1, 0])
		view_matrix = transform.matrix.numpy()[..., 0]

		view_matrix = view_matrix_inverse(view_matrix) # COLMAP expects Cam-to-World Transformation

		extr_obj = {}
		extr_obj["view_matrix"] = view_matrix.flatten().tolist()

		cam_matrix = np.identity(3)
		cam_matrix[0, 0] = focal_length
		cam_matrix[1, 1] = focal_length
		cam_matrix[0, 2] = principal_point_x
		cam_matrix[1, 2] = principal_point_y

		intr_obj = {}
		intr_obj["camera_matrix"] = cam_matrix.flatten().tolist()
		intr_obj["resolution"] = [width, height]

		cam_obj = {}
		cam_obj["camera_id"] = str(i).zfill(4) + ".png"
		cam_obj["extrinsics"] = extr_obj
		cam_obj["intrinsics"] = intr_obj
		cam_obj["is_test_cam"] = (i >= num_train_cams)
		
		cameras.append(cam_obj)

	poses = {}
	poses["cameras"] = cameras

	with open(os.path.join(output_path, "poses.json"), "w") as outfile:
		json.dump(poses, outfile, indent=2)

def output_transforms(scene, scene_path, radius, thetas, phis, num_train_cams):
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

	transforms["aabb_scale"] = 128
	transforms["frames"] = []

	up = np.zeros(3)

	for i, theta, phi in tqdm(zip(range(num_train_cams), thetas, phis), desc="Writing", total=num_train_cams):
		camera_position = spherical_to_cartesian(radius, theta, phi)
		transform = mi.Transform4f().look_at(origin=camera_position, target=[0, 0, 0], up=[0, -1, 0])
		view_matrix = transform.matrix.numpy()[..., 0]

		view_matrix = view_matrix_inverse(view_matrix)

		# do weird shit
		R = view_matrix[:3, :3]
		tvec = view_matrix[:3, 3]
		qvec = rotmat2qvec(R)

		R = qvec2rotmat(-qvec)
		t = tvec.reshape((3,1))
		m = np.concatenate([np.concatenate([R, t], axis=1), np.array([[0.0,0.0,0.0,1.0]])], axis=0)
		c2w = np.linalg.inv(m)

		# wtf
		c2w[0:3,2] *= -1
		c2w[0:3,1] *= -1
		c2w = c2w[[1,0,2,3],:]
		c2w[2,:] *= -1

		up += c2w[0:3,1]

		# flip_mat = np.array([
		# 	[1, 0, 0, 0],
		# 	[0, -1, 0, 0],
		# 	[0, 0, -1, 0],
		# 	[0, 0, 0, 1]
		# ])

		# c2w = np.matmul(c2w, flip_mat)

		# view_matrix = view_matrix_inverse(view_matrix)

		file_path = os.path.join(scene_path, "images", str(i).zfill(4) + ".png")
		b = sharpness(file_path)

		camera = {}
		camera["file_path"] = os.path.join("images", str(i).zfill(4) + ".png")
		camera["sharpness"] = b
		camera["transform_matrix"] = c2w

		transforms["frames"].append(camera)

	# idek anymore
	up = up / np.linalg.norm(up)
	R = rotmat(up, [0,0,1])
	R = np.pad(R, [0,1])
	R[-1, -1] = 1

	for f in transforms["frames"]:
		f["transform_matrix"] = np.matmul(R, f["transform_matrix"])

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
		f["transform_matrix"][0:3,3] -= totp

	avglen = 0.
	for f in transforms["frames"]:
		avglen += np.linalg.norm(f["transform_matrix"][0:3,3])
	avglen /= num_train_cams
	for f in transforms["frames"]:
		f["transform_matrix"][0:3,3] *= 4.0 / avglen # scale to "nerf sized"

	for f in transforms["frames"]:
		f["transform_matrix"] = f["transform_matrix"].tolist()

	with open(os.path.join(scene_path, "transforms.json"), "w") as outfile:
		json.dump(transforms, outfile, indent=2)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--scene", "-s", type=str, required=True)
	parser.add_argument("--output", "-o", type=str, required=True)
	parser.add_argument("--resolution", "--res", "-r", default=512, type=int, required=False)
	parser.add_argument("--samples", "--spp", default=512, type=int, required=False)
	parser.add_argument("--image_count", "-c", default=64, type=int, required=False)
	args = parser.parse_args()

	assert os.path.exists(args.scene)
	
	if not os.path.exists(args.output):
		os.makedirs(args.output)

	radius = 75

	# Train views
	thetas_train, phis_train = golden_spiral(args.image_count)

	# Test views
	thetas_0 = np.array([0.35*np.pi, 0.5*np.pi, 0.65*np.pi])
	phis_0 = np.linspace(0, 2*np.pi, 4, endpoint=False)

	thetas_test, phis_test = np.meshgrid(thetas_0, phis_0)

	thetas = np.concatenate([thetas_train, thetas_test.flatten()])
	phis = np.concatenate([phis_train, phis_test.flatten()])

	print("Loading scenes...")
	dr.set_flag(dr.JitFlag.Debug, True)
	mi.set_variant("cuda_ad_spectral_polarized")
	polarized_scene = mi.load_file(args.scene, res=args.resolution)
	unpolarized_scene = mi.load_file(args.scene, res=args.resolution, polarizing=False)
	print()

	print("Generating alpha masks...")
	masks = render_masks(polarized_scene, radius, thetas, phis)
	save_images(masks, os.path.join(args.output, "masks"), extension=".png.png")
	print()

	print("Generating unpolarized images...")
	unpolarized_images = render_unpolarized_images(unpolarized_scene, radius, thetas, phis, args.samples)
	unpolarized_images = mask_images(unpolarized_images, masks)
	save_images(unpolarized_images, os.path.join(args.output, "unpolarized", "images"))
	print()

	print("Generating polarized images...")
	polarized_images = render_polarized_images(polarized_scene, radius, thetas, phis, args.samples)
	polarized_images = mask_images(polarized_images, masks)
	save_images(polarized_images[:, 0, ...], os.path.join(args.output, "polarized_0", "images"))
	save_images(polarized_images[:, 1, ...], os.path.join(args.output, "polarized_90", "images"))
	print()

	print("Generating camera poses...")
	output_poses(polarized_scene, args.output, radius, thetas, phis, args.image_count)
	# output_transforms(polarized_scene, os.path.join(args.output, "unpolarized"), radius, thetas, phis, args.image_count)

if __name__ == "__main__":
	main()