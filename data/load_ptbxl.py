import wfdb
import os
import numpy as np

PTB_PATH = "data/ptbxl"

signals = []

for patient in os.listdir(PTB_PATH):
    p_dir = os.path.join(PTB_PATH, patient)
    if not os.path.isdir(p_dir):
        continue

    for file in os.listdir(p_dir):
        if file.endswith(".hea"):
            record_name = file.replace(".hea","")
            record_path = os.path.join(p_dir, record_name)

            try:
                record = wfdb.rdrecord(record_path)

                ecg = record.p_signal  # (time, channels)

                # ---- FIX HERE ----
                ecg = ecg[:, :12]      # keep only 12 leads

                # ensure 1000 length
                if ecg.shape[0] < 1000:
                    continue

                ecg = ecg[:1000]

                signals.append(ecg)

            except Exception as e:
                print("skip:", record_name)

signals = np.array(signals)

print("Loaded PTBXL:", signals.shape)

np.save("data/ptbxl_X.npy", signals)
