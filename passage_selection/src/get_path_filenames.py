import sys
import os
from collections import deque

input_paths = deque(sys.argv[1:-2])
output_filename = sys.argv[-2]
recurse = sys.argv[-1].lower() == "true"

with open(output_filename, "w") as output_file:
	while len(input_paths) > 0:
		input_path = input_paths.popleft()
		dir = os.listdir(input_path)
		for item in dir:
			input_name = input_path + "/" + item
			if os.path.isfile(input_name):
				output_file.write(input_name + "\n")
			elif os.path.isdir(input_name) and recurse:
				input_paths.append(input_name)
print("Done.")