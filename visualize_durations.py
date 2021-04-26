
import pandas as pd
import matplotlib.pyplot as plt

inference_times  = []

with open('winml_durations.txt') as f:
	for line in f.readlines():
		if len(line) != 0:
			(nap_time, duration) = line.split()
			inference_times.append((int(nap_time), int(duration)))

df = pd.DataFrame(inference_times, columns=['nap_time', 'inference_dur'])
groupped = df.groupby('nap_time').agg({'inference_dur': ['min', 'max', 'mean', 'std']})
print(groupped.to_string())
p = df.boxplot(column=['inference_dur'], by='nap_time')
p.set_xlabel('nap_time [ms]')
p.set_ylabel('duration [ms]')
plt.title('winml inference durations') 
plt.suptitle('') 
plt.show()
