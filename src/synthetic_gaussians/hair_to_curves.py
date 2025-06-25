import argparse
import os
import struct

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

def convert(input: str, output: str):
	with open(input, "rb") as f:
		header = Header(f.read(128)) # Header consists of first 128 bytes

		if header.has_segments_arr:
			segments_bytes = f.read(2 * header.num_strands)
			segments = struct.unpack("<"+"H"*header.num_strands, segments_bytes)

		if header.has_points_arr:
			points_bytes = f.read(3 * 4 * header.num_points)
			points = struct.unpack("<"+"f"*3*header.num_points, points_bytes)

		if header.has_thickness_arr:
			thickness_bytes = f.read(4 * header.num_points)
			thickness = struct.unpack("<"+"f"*header.num_points, thickness_bytes)

		if header.has_transparency_arr:
			transparency_bytes = f.read(4 * header.num_points)
			transparency = struct.unpack("<"+"f"*header.num_points, transparency_bytes)

		if header.has_color_arr:
			color_bytes = f.read(3 * 4 * header.num_points)
			color = struct.unpack("<"+"f"*3*header.num_points, color_bytes)

	lines = []

	point_idx = 0

	for strand in range(header.num_strands):
		if header.has_segments_arr:
			num_control_points = segments[strand] + 1
		else:
			num_control_points = header.default_num_segments + 1

		for point in range(point_idx, point_idx + num_control_points):
			if header.has_thickness_arr:
				point_thickness = thickness[point]
			else:
				point_thickness = header.default_thickness

			line = " ".join([str(points[3 * point + i]) for i in range(3)]) + " " + str(point_thickness) + "\n"

			lines.append(line)

		point_idx += num_control_points
		lines.append("\n")

	with open(output, "w") as f:
		f.writelines(lines)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--source", "-s", type=str, required=True)
	parser.add_argument("--output", "-o", type=str, required=True)
	args = parser.parse_args()

	assert os.path.exists(args.source)

	convert(args.source, args.output)

if __name__ == "__main__":
	main()