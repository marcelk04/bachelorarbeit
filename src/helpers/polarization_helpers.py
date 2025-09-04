import numpy as np

def separate(img_0, img_90):
	I_max = np.fmax(img_0, img_90)
	I_min = np.fmin(img_0, img_90)

	indirect = 2 * I_min
	direct = I_max - I_min

	return indirect, direct

def separate_2(img_0, img_90):
	indirect = 2 * img_90
	direct = img_0 - img_90

	return indirect, direct

def reconstruct(indirect, direct):
	return indirect + direct