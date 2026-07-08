import os
import subprocess

dirs = os.listdir('.')
data_dirs = [data_dir for data_dir in dirs if 'data_' in data_dir]

for data_dir in data_dirs:
	os.chdir(data_dir)
	subprocess.run(["python","run.py"])
	os.chdir('../')
