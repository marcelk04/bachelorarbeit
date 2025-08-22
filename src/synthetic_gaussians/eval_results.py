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

def plot_train_metric(ax, data, metric, label):
	if type(data[metric]) == list:
		ax.plot(data[metric], label=label)
	else:
		ax.plot(data[metric]["iteration"], data[metric]["value"], label=label)

def plot_graph(data, scenes, metric):
	fig, ax = plt.subplots()

	for d, scene in zip(data, scenes):
		plot_train_metric(ax, d, metric, scene)

	ax.set_xlabel("iteration")
	ax.set_ylabel(metric)
	ax.grid()
	ax.legend()

	return fig

def load_train_results(paths):
	result = []

	for path in paths:
		with open(path) as fp:
			result.append(json.load(fp))

	return result

def plot_train_results(source, output, scenes, metrics):
	unpolarized_path = os.path.join(source, "unpolarized", "model", "train_results.json")
	composite_path = os.path.join(source, "composite", "model", "train_results_combined.json")
	global_path = os.path.join(source, "composite", "model", "train_results1.json")
	direct_path = os.path.join(source, "composite", "model", "train_results2.json")

	data = load_train_results([unpolarized_path, composite_path, global_path, direct_path])
	
	for metric in metrics:
		plot_graph(data, scenes, metric).savefig(os.path.join(output, metric + "_train.svg"), bbox_inches="tight")

	additional_metrics = ["loss", "points"]

	for metric in additional_metrics:
		plot_graph(data, scenes, metric).savefig(os.path.join(output, metric + "_train.svg"), bbox_inches="tight")


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

	plot_train_results(args.source, output_path, scenes, metrics)

if __name__ == "__main__":
	main()