import numpy as np
from copy import deepcopy


def linear_function(x, a, b):
    return a * x + b


def super_resolution_matrix(size):
    matrix = []
    for i in range(size):
        line = np.zeros(size)
        if i < size - 1:
            line[[i, i + 1]] = 1
        else:
            line[i] = 1
        matrix.append(line)

    return np.linalg.inv(matrix)


def super_resolution_matrix2(size):
    matrix = []
    for i in range(size):
        line = np.zeros(size)
        if i ==0:
            line[[i]] = 1
        else:
            line[[i-1, i]] = 1
        matrix.append(line)

    return np.linalg.inv(matrix)


def apply_super_resolution(input_array):
    super_resolved_array1 = np.zeros_like(input_array)
    super_resolved_array2 = np.zeros_like(input_array)
    super_resolved_array = np.zeros_like(input_array)
    for i, line in enumerate(input_array):
        super_resolved_array1[i] = np.dot(super_resolution_matrix(np.shape(input_array)[1]), line)
        super_resolved_array2[i] = np.dot(super_resolution_matrix2(np.shape(input_array)[1]), line)
        for j, k in enumerate(line):
            super_resolved_array[i][j] = (j-1)/(np.shape(input_array)[1]-1) * super_resolved_array1[i][j] \
                            + (np.shape(input_array)[1]-j)/(np.shape(input_array)[1]-1) * super_resolved_array2[i][j]
    return super_resolved_array


def current_conversion(signal, full_scale=848717, int_time=1e-3):
    full_scale_charge = 848717 * 12.5e-12 * 1e-3 / 2000
    return 2 * signal / full_scale * full_scale_charge / int_time
