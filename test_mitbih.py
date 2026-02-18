import wfdb
import matplotlib.pyplot as plt

# load record
record = wfdb.rdrecord("data/mitdb/100")
ann = wfdb.rdann("data/mitdb/100", 'atr')

signal = record.p_signal[:,0]

print("Total samples:", len(signal))
print("Heartbeats:", len(ann.sample))

# plot
plt.figure(figsize=(12,4))
plt.plot(signal[:2000])

for r in ann.sample:
    if r < 2000:
        plt.axvline(r, color='red', linewidth=0.7)

plt.title("MIT-BIH ECG with Beat Annotations")
plt.show()
