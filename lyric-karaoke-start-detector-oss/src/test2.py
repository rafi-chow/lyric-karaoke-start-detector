import numpy as np
p = r"C:\Users\psult\Downloads\Harmonix_melspecs\melspecs\0833_northshore-mel.npy"
mel = np.load(p)
print(mel.shape, mel.min(), mel.max(), mel.mean(), mel.std())
