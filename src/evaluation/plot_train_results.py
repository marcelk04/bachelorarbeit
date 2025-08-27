import argparse
import json
import os

from matplotlib import pyplot as plt

from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from helpers.sys_helpers import create_dir

def load_train_results(paths):
	result = []

	for path in paths:
		with open(path) as fp:
			result.append(json.load(fp))

	return result

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

def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--unpolarized_path", "-u", default="", type=str, required=False)
	parser.add_argument("--composite_path", "-c", default="", type=str, required=False)
	parser.add_argument("--global_path", "-g", default="", type=str, required=False)
	parser.add_argument("--direct_path", "-d", default="", type=str, required=False)
	parser.add_argument("--output", "-o", type=str, required=True)
	parser.add_argument("--metrics", "-m", type=str, nargs="+", default=["PSNR", "SSIM", "LPIPS"], required=False)
	args = parser.parse_args()

	create_dir(args.output)

	scenes = []
	scene_paths = []

	if args.unpolarized_path != "":
		scenes.append("unpolarized")
		scene_paths.append(args.unpolarized_path)

	if args.composite_path != "":
		scenes.append("composite")
		scene_paths.append(args.composite_path)

	if args.global_path != "":
		scenes.append("global")
		scene_paths.append(args.global_path)

	if args.direct_path != "":
		scenes.append("direct")
		scene_paths.append(args.direct_path)

	data = load_train_results(scene_paths)

	for metric in args.metrics:
		plot_graph(data, scenes, metric).savefig(os.path.join(args.output, f"{metric}_train.svg"), bbox_inches="tight")

	additional_metrics = ["loss", "points"]

	for metric in additional_metrics:
		plot_graph(data, scenes, metric).savefig(os.path.join(args.output, f"{metric}_train.svg"), bbox_inches="tight")

if __name__ == "__main__":
	main()