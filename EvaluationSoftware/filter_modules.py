import numpy as np
from copy import deepcopy


def simple_artefact_filter(signal_array_2dim, threshold=None):
    filtered_array = deepcopy(signal_array_2dim)
    if threshold is None:
        threshold = signal_array_2dim.flatten().mean()
    array_col = np.shape(signal_array_2dim)[0]
    array_row = np.shape(signal_array_2dim)[1]
    for i in range(array_col):
        if i == 0 or i == array_col - 1:
            continue
        for j in range(array_row):
            if j == 0 or j == array_row - 1:
                continue
            if signal_array_2dim[i][j] > threshold > signal_array_2dim[i - 1][j] \
                    and signal_array_2dim[i + 1][j] < threshold \
                    and signal_array_2dim[i][j - 1] < threshold \
                    and signal_array_2dim[i][j + 1] < threshold:
                filtered_array[i][j] = (signal_array_2dim[i - 1][j] + signal_array_2dim[i + 1][j] +
                                        signal_array_2dim[i][j - 1] + signal_array_2dim[i][j + 1]) / 4
    return filtered_array
