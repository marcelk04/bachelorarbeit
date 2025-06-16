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