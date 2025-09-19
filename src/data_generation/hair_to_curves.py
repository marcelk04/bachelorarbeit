import argparse
import os
import struct
import random
import numpy as np
from tqdm import tqdm

def rand_float(min: float, max: float):
	return min + random.random() / (max - min)

def read_uint(bytes):
	return int.from_bytes(bytes, byteorder="little", signed=False)

def read_int(bytes):
	return int.from_bytes(bytes, byteorder="little", signed=True)

def read_float(bytes):
	return struct.unpack("f", bytearray(bytes))[0]

class Header:
	def __init__(self, header_bytes):
		assert (header_bytes[0:4]).decode("ascii") == "HAIR"

		self.num_strands = read_uint(header_bytes[4:8])
		self.num_points = read_uint(header_bytes[8:12])

		self.bits = read_int(header_bytes[12:16])

		self.has_segments_arr     = bool(self.bits & (1 << 0))
		self.has_points_arr       = bool(self.bits & (1 << 1))
		self.has_thickness_arr    = bool(self.bits & (1 << 2))
		self.has_transparency_arr = bool(self.bits & (1 << 3))
		self.has_color_arr        = bool(self.bits & (1 << 4))
		
		assert self.has_points_arr

		self.default_num_segments = read_uint(header_bytes[16:20])
		self.default_thickness    = read_float(header_bytes[20:24])
		self.default_transparency = read_float(header_bytes[24:28])
		self.default_color       = [read_float(header_bytes[28:32]), read_float(header_bytes[32:36]), read_float(header_bytes[36:40])]

def get_strand(strand_idx: int, header: Header, points, segments, start_indices, offset=(0, 0, 0)) -> list[tuple[int, int, int]]:
	if header.has_segments_arr:
		strand_length = segments[strand_idx] + 1
	else:
		strand_length = header.default_num_segments + 1

	strand = []

	start = start_indices[strand_idx]
	end = start + strand_length

	for point in range(start, end):
		segment = (round(points[3 * point + 0], 6) + offset[0], round(points[3 * point + 1], 6) + offset[1], round(points[3 * point + 2], 6) + offset[2])
		strand.append(segment)

	return strand

def strand_to_string(strand: list[tuple[int, int, int]], radius=0.004):
	return "".join([" ".join([str(i) for i in point]) + " " + str(radius) + "\n" for point in strand]) + "\n"

def convert(input: str, output: str, target_strands: int):
	with open(input, "rb") as f:
		header = Header(f.read(128)) # Header consists of first 128 bytes

		print("Number of strands:", header.num_strands)
		print("Number of points:", header.num_points)
		print(f"Default color: {header.default_color}")

		print()
		print("Present Arrays:")

		if header.has_segments_arr:
			print("Segments array present")
			segments_bytes = f.read(2 * header.num_strands)
			segments = struct.unpack("<"+"H"*header.num_strands, segments_bytes)
		else:
			segments = None

		if header.has_points_arr:
			print("Points array present")
			points_bytes = f.read(3 * 4 * header.num_points)
			points = struct.unpack("<"+"f"*3*header.num_points, points_bytes)
		else:
			points = None

		if header.has_thickness_arr:
			print("Thickness array present")
			thickness_bytes = f.read(4 * header.num_points)
			thickness = struct.unpack("<"+"f"*header.num_points, thickness_bytes)
		else:
			thickness = None

		if header.has_transparency_arr:
			print("Transparency array present")
			transparency_bytes = f.read(4 * header.num_points)
			transparency = struct.unpack("<"+"f"*header.num_points, transparency_bytes)
		else:
			transparency = None

		if header.has_color_arr:
			print("Color array present")
			color_bytes = f.read(3 * 4 * header.num_points)
			color = struct.unpack("<"+"f"*3*header.num_points, color_bytes)
		else:
			color = None

		print()

	start_indices = []

	lines = []

	radius = 0.004
	point_idx = 0

	print("Converting Strands:")
	for strand_idx in tqdm(range(header.num_strands), total=header.num_strands):
		start_indices.append(point_idx)

		strand = get_strand(strand_idx, header, points, segments, start_indices)
		line = strand_to_string(strand, radius=radius)
		lines.append(line)

		point_idx += len(strand)

	print("Densifiying:")
	for i in tqdm(range(max(target_strands - header.num_strands, 0)), total=max(target_strands - header.num_strands, 0)):
		strand_idx = random.randint(0, header.num_strands-1)
		offset = (rand_float(-0.1, 0.1), rand_float(-0.1, 0.1), rand_float(-0.1, 0.1))
		strand = get_strand(strand_idx, header, points, segments, start_indices, offset)
		line = strand_to_string(strand, radius=radius)
		lines.append(line)


	with open(output, "w") as f:
		f.writelines(lines)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--input", "-i", type=str, required=True)
	parser.add_argument("--output", "-o", default="", type=str, required=False)
	parser.add_argument("--target_strands", default=75_000, type=int, required=False)
	args = parser.parse_args()

	assert os.path.exists(args.input)

	in_path = args.input
	out_path = args.output

	if out_path == "":
		name_wo_extension = os.path.splitext(in_path)[0]
		out_path = name_wo_extension + ".txt"

	convert(in_path, out_path, args.target_strands)

if __name__ == "__main__":
	main()