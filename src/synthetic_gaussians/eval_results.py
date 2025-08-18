import numpy as np
from matplotlib import pyplot as plt
import os
import json
import argparse

from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from helpers.sys_helpers import *

def load_per_view(path, scenes, metric):
	result = []

	for scene in scenes:
		with open(os.path.join(path, scene, "per_view.json")) as fp:
			result.append(json.load(fp)["ours_30000"][metric])

	return result

def plot_boxplot(data, scenes):
	values = [np.array(list(x.values())) for x in data]
	values = np.stack(values).transpose()

	fig, ax = plt.subplots()
	ax.boxplot(values, tick_labels=scenes, showmeans=True, meanline=True)

	return fig

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--source", "-s", type=str, required=True)
	args = parser.parse_args()

	source = os.path.join(args.source, "results")
	assert os.path.exists(source)
	
	output_path = os.path.join(source, "graphs")
	create_dir(output_path)

	scenes = ["unpolarized", "composite", "global", "direct"]
	metrics = ["PSNR", "SSIM", "LPIPS"]

	for metric in metrics:
		data = load_per_view(source, scenes, metric)
		plot_boxplot(data, scenes).savefig(os.path.join(output_path, metric + ".svg"), bbox_inches="tight")

if __name__ == "__main__":
	main()