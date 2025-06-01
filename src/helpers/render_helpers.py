import numpy as np
import mitsuba as mi

def spherical_to_cartesian(radius, theta, phi):
	x = radius * np.sin(theta) * np.sin(phi)
	y = radius * np.cos(theta)
	z = radius * np.sin(theta) * np.cos(phi)

	return np.array([x, y, z])

def render_from_angle(scene, radius, theta, phi, polarized=True, spp=512):
	cam_pos = spherical_to_cartesian(radius, theta, phi)
	polarizer_pos = spherical_to_cartesian(radius - 0.1, theta, phi)

	params = mi.traverse(scene)
	params["polarizer_cam.bsdf.theta.value"] = 0 # Make sure the polarizer angle is 0 (might have been altered by previous executions)
	params["sensor.to_world"] = mi.ScalarTransform4f().look_at(origin=cam_pos, target=[0, 0, 0], up=[0, 1, 0])
	params["polarizer_cam.to_world"] = mi.ScalarTransform4f().look_at(origin=polarizer_pos, target=[0, 0, 0], up=[0, 1, 0]).rotate(axis=[0, 0, 1], angle=90)
	params.update()

	if polarized:
		img_0 = mi.render(scene, spp=spp)

		params["polarizer_cam.bsdf.theta.value"] = 90
		params.update()

		img_90 = mi.render(scene, spp=spp)

		return (img_0, img_90)
	else:
		return mi.render(scene, spp=spp)
	
def render_from_angles(scene, radius, thetas, phis, polarized=True, spp=512):
	return [render_from_angle(scene, radius, theta, phi, polarized, spp) for theta, phi in zip(thetas, phis)]