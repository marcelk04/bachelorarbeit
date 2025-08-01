import numpy as np

def color_to_absorption(color, beta=0.3):
	return (np.log(color) / (5.969 - 0.215 * beta + 2.532 * beta ** 2 - 10.73 * beta ** 3 + 5.574 * beta ** 4 + 0.245 * beta ** 5 )) ** 2

def absorption_to_extinction(absorption, albedo):
	return absorption / (1 - albedo + 1e-6)

def melanin_to_absorption(eumelanin=1.3, pheomelanin=0.2):
	eumelanin_sigma_a = np.array([0.419, 0.697, 1.37])
	pheomelanin_sigma_a = np.array([0.187, 0.4, 1.05])

	return eumelanin * eumelanin_sigma_a + pheomelanin * pheomelanin_sigma_a