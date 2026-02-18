import numpy as np
x = np.load("data/mitbih_X.npy")[100]
np.save("sample_ecg.npy", x)
