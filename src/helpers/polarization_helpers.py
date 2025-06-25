import numpy as np

def separate(img_0, img_90):
	indirect = 2 * np.fmin(img_0, img_90)
	direct = np.fmax(img_0, img_90) - indirect / 2

	return indirect, direct

def reconstruct(indirect, direct):
	return indirect + direct