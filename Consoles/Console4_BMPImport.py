import cv2
import os

import matplotlib.pyplot as plt
import numpy as np
from EvaluationSoftware.main import *

path = '/Users/nico_brosda/Cyrce_Messungen/Gafchromic_111024/gafchromic_111024_beam_010.bmp'
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
print(np.shape(image))

fig, ax = plt.subplots()
ax.imshow(image, cmap='gray', vmin=0, vmax=255)
plt.show()