import numpy as np
import cv2

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

def qvec2rotmat(qvec):
	return np.array([
		[
			1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
			2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
			2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
		], [
			2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
			1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
			2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
		], [
			2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
			2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
			1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
		]
	])

def rotmat(a, b):
	a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
	v = np.cross(a, b)
	c = np.dot(a, b)
	# handle exception for the opposite direction input
	if c < -1 + 1e-10:
		return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

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

def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()

def sharpness(image_path):
	image = cv2.imread(image_path)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = variance_of_laplacian(gray)
	return fm

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
	da = da / np.linalg.norm(da)
	db = db / np.linalg.norm(db)
	c = np.cross(da, db)
	denom = np.linalg.norm(c)**2
	t = ob - oa
	ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
	tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
	if ta > 0:
		ta = 0
	if tb > 0:
		tb = 0
	return (oa+ta*da+ob+tb*db) * 0.5, denom