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


def simple_zero_replace(image_map, direction=0):
    for i in range(np.shape(image_map)[0]):
        for j in range(np.shape(image_map)[1]):
            if image_map[i, j] == 0:
                if direction == 0 and i != 0 and i != np.shape(image_map)[0]-1:
                    if image_map[i-1, j] != 0 and image_map[i+1, j] != 0:
                        image_map[i, j] = (image_map[i-1, j] + image_map[i+1, j])/2
                if direction == 1 and j != 0 and j != np.shape(image_map)[1]-1:
                    if image_map[i, j-1] != 0 and image_map[i, j+1] != 0:
                        image_map[i, j] = (image_map[i, j-1] + image_map[i, j+1])/2
    return image_map


def zero_pixel_replace(image_map):
    for i in range(np.shape(image_map)[0]):
        for j in range(np.shape(image_map)[1]):
            try:
                if image_map[i, j] == 0:
                    cache = []
                    if i == 0:
                        if image_map[i+1, j] != 0:
                            cache.append(image_map[i+1, j])
                        if image_map[i, j+1] != 0:
                            cache.append(image_map[i, j+1])
                        if image_map[i, j-1] != 0:
                            cache.append(image_map[i, j-1])
                    elif j == 0:
                        if image_map[i+1, j] != 0:
                            cache.append(image_map[i+1, j])
                        if image_map[i, j+1] != 0:
                            cache.append(image_map[i, j+1])
                        if image_map[i-1, j] != 0:
                            cache.append(image_map[i-1, j])
                    elif i == np.shape(image_map)[0] - 1:
                        if image_map[i-1, j] != 0:
                            cache.append(image_map[i - 1, j])
                        if image_map[i, j + 1] != 0:
                            cache.append(image_map[i, j + 1])
                        if image_map[i, j - 1] != 0:
                            cache.append(image_map[i, j - 1])
                    elif j == np.shape(image_map)[1] - 1:
                        if image_map[i + 1, j] != 0:
                            cache.append(image_map[i + 1, j])
                        if image_map[i-1, j] != 0:
                            cache.append(image_map[i-1, j])
                        if image_map[i, j - 1] != 0:
                            cache.append(image_map[i, j - 1])
                    else:
                        if image_map[i + 1, j] != 0:
                            cache.append(image_map[i + 1, j])
                        if image_map[i - 1, j] != 0:
                            cache.append(image_map[i - 1, j])
                        if image_map[i, j + 1] != 0:
                            cache.append(image_map[i, j + 1])
                        if image_map[i, j - 1] != 0:
                            cache.append(image_map[i, j - 1])
                    if len(cache) > 0:
                        image_map[i, j] = np.mean(cache) / len(cache)
            except IndexError:
                continue
    return image_map
