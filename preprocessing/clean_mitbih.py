import numpy as np
from prepare_mitbih import X, y

# AAMI mapping
mapping = {
    'N':'N','L':'N','R':'N','e':'N','j':'N',

    'A':'S','a':'S','J':'S','S':'S',

    'V':'V','E':'V',

    'F':'F',

    '/':'Q','f':'Q','Q':'Q','x':'Q','~':'Q','+':'Q','|':'Q'
}

clean_X = []
clean_y = []

for beat, label in zip(X, y):
    if label in mapping:
        clean_X.append(beat)
        clean_y.append(mapping[label])

clean_X = np.array(clean_X)
clean_y = np.array(clean_y)

print("New dataset:", clean_X.shape)
print("Classes:", set(clean_y))

# save dataset
np.save("data/mitbih_X.npy", clean_X)
np.save("data/mitbih_y.npy", clean_y)
