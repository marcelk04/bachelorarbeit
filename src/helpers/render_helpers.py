import numpy as np
import mitsuba as mi
from tqdm import tqdm

def spherical_to_cartesian(radius, theta, phi):
	x = radius * np.sin(theta) * np.sin(phi)
	y = radius * np.cos(theta)
	z = radius * np.sin(theta) * np.cos(phi)

	return np.array([x, y, z])

def render_np(scene, spp, integrator=None):
	if type(integrator) == type(None):
		return mi.render(scene, spp=spp).numpy()
	else:
		return mi.render(scene, spp=spp, integrator=integrator).numpy()

def render_from_angle(scene, radius, theta, phi, polarized=True, spp=512, integrator=None):
	cam_pos = spherical_to_cartesian(radius, theta, phi)
	polarizer_pos = spherical_to_cartesian(radius - 0.1, theta, phi)

	params = mi.traverse(scene)
	params["sensor.to_world"] = mi.ScalarTransform4f().look_at(origin=cam_pos, target=[0, 0, 0], up=[0, 1, 0])

	if polarized:
		params["polarizer_cam.to_world"] = mi.ScalarTransform4f().look_at(origin=polarizer_pos, target=[0, 0, 0], up=[0, 1, 0]).rotate(axis=[0, 0, 1], angle=90)
		params["polarizer_cam.bsdf.theta.value"] = 0 # Make sure the polarizer angle is 0 (might have been altered by previous executions)

	params.update()

	if polarized:
		img_0 = render_np(scene, spp, integrator)

		params["polarizer_cam.bsdf.theta.value"] = 90
		params.update()

		img_90 = render_np(scene, spp, integrator)

		return np.stack((img_0, img_90)) # Dimensions: (2, W, H, 3)
	else:
		return render_np(scene, spp, integrator)[None, ...] # Dimensions: (1, W, H, 3)
	
def render_from_angles(scene, radius, thetas, phis, polarized=True, spp=512, integrator=None):
	images = []
	for theta, phi in tqdm(zip(thetas, phis), desc="Rendering", total=len(thetas)):
		images.append(render_from_angle(scene, radius, theta, phi, polarized, spp, integrator))
	return np.stack(images) # Dimensions: (N, 1/2, W, H, 3)