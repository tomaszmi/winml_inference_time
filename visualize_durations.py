
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def get_files():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--file', action='append', help='Path to file with durations')
	args = parser.parse_args()
	if not args.file:
		args.file = ['winml_durations.txt']
	return args.file

def get_inference_times_from_file(file_path):
	inference_times  = []
	with open(file_path) as f:
		for line in f.readlines():
			if len(line) != 0:
				(nap_time, duration) = line.split()
				inference_times.append((int(nap_time), int(duration)))
	return inference_times

for file_path in get_files():
	inference_times = get_inference_times_from_file(file_path)
	df = pd.DataFrame(inference_times, columns=['nap_time', 'inference_dur'])
	groupped = df.groupby('nap_time').agg({'inference_dur': ['min', 'max', 'mean', 'std']})
	print(groupped.to_string())
	p = df.boxplot(column=['inference_dur'], by='nap_time')
	p.set_xlabel('nap_time [ms]')
	p.set_ylabel('duration [ms]')

	plt.title(file_path) 
	plt.suptitle('') 

plt.show()
