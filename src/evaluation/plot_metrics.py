import argparse
import json
import os

from matplotlib import pyplot as plt
import numpy as np

from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from helpers.sys_helpers import create_dir

def load_per_view(path: str, scenes: list[str], metric: str) -> list[dict[str, float]]:
	result = []

	for scene in scenes:
		with open(os.path.join(path, scene, "per_view.json")) as fp:
			result.append(json.load(fp)["ours_30000"][metric])

	return result

def plot_boxplot(data: list[dict[str, float]], scenes: list[str]) -> plt.Figure:
	values = [np.array(list(x.values())) for x in data]
	values = np.stack(values).transpose()

	fig, ax = plt.subplots()
	ax.boxplot(values, tick_labels=scenes, showmeans=True, meanline=True)

	return fig

def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--source", "-s", type=str, required=True, help="Path to results directory.")
	parser.add_argument("--metrics", "-m", type=str, nargs="+", default=["PSNR", "SSIM", "LPIPS"], required=False, help="Metrics to plot.")
	parser.add_argument("--scenes", type=str, nargs="+", default=["unpolarized", "composite", "global", "direct"], required=False, help="Scenes to plot.")
	args = parser.parse_args()

	assert os.path.exists(args.source)

	output_path = os.path.join(args.source, "graphs")
	create_dir(output_path)

	for metric in args.metrics:
		data = load_per_view(args.source, args.scenes, metric)
		plot_boxplot(data, args.scenes).savefig(os.path.join(output_path, f"{metric}_test.svg"), bbox_inches="tight")

if __name__ == "__main__":
	main()
