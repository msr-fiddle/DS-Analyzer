import os
import json
import sys


def main():
	if len(sys.argv) < 2:
		print("Input file path")
		sys.exit(1)

	path = sys.argv[1]
	for files in os.listdir(path):
		for subdir in os.listdir(os.path.join(path, files)):
			json_file = os.path.join(path, files,subdir, "MODEL.json")

			data = json.load(open(json_file, 'r')) 
			print(files, data['SPEED_INGESTION'], data['SPEED_DISK'], data['SPEED_CACHED']) 

if __name__ == "__main__":
	main()
