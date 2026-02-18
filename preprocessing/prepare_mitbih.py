import wfdb
import numpy as np
import os

DATA_PATH = "data/mitdb"


def extract_beats(record_name, window=120):
    record_path = os.path.join(DATA_PATH, record_name)

    record = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, 'atr')

    signal = record.p_signal[:, 0]  # lead 1

    beats = []
    labels = []

    for i, r in enumerate(ann.sample):

        if r - window < 0 or r + window > len(signal):
            continue

        beat = signal[r - window:r + window]

        beats.append(beat)
        labels.append(ann.symbol[i])

    return beats, labels


all_beats = []
all_labels = []

records = [f.split('.')[0] for f in os.listdir(DATA_PATH) if f.endswith('.dat')]

print("Total records:", len(records))

for rec in records[:10]:   # start small first
    beats, labels = extract_beats(rec)
    all_beats.extend(beats)
    all_labels.extend(labels)
    print(f"{rec} -> {len(beats)} beats")

X = np.array(all_beats)
y = np.array(all_labels)

print("Dataset shape:", X.shape)
print("Unique labels:", set(y))
