import argparse

from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from helpers.polarization_helpers import *

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--indirect", "-i", type=str, required=True)
	parser.add_argument("--direct", "-d", type=str, required=True)

if __name__ == "__main__":
	main()