import numpy as np

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

def rotmat2qvec(R):
	Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
	K = np.array([
		[Rxx - Ryy - Rzz, 0, 0, 0],
		[Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
		[Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
		[Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
	eigvals, eigvecs = np.linalg.eigh(K)
	qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
	if qvec[0] < 0:
		qvec *= -1
	return qvec

# implementation from stack overflow: https://stackoverflow.com/a/26127012
def fibonacci_sphere(samples):
	points = np.zeros((samples, 3))
	phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians

	for i in range(samples):
		y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
		radius = np.sqrt(1 - y * y)  # radius at y

		theta = phi * i  # golden angle increment

		x = np.cos(theta) * radius
		z = np.sin(theta) * radius

		points[i] = np.array([x, y, z])

	return points

# implementation from stack overflow: https://stackoverflow.com/a/44164075
def golden_spiral(samples):
	indices = np.arange(0, samples, dtype=float) + 0.5

	thetas = np.arccos(1 - 2 * indices / samples)
	phis = np.pi * (1 + np.sqrt(5.0)) * indices

	return thetas, phis

def spherical_to_cartesian(radius, theta, phi):
	x = radius * np.sin(theta) * np.sin(phi)
	y = radius * np.cos(theta)
	z = radius * np.sin(theta) * np.cos(phi)

	return np.stack([x, y, z], axis=-1)
