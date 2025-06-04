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