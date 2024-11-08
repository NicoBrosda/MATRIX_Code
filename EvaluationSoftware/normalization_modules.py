import os
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from Plot_Methods.plot_standards import *
from tqdm import tqdm
from skimage.filters import threshold_otsu as ski_threshold_otsu


def simple_normalization(list_of_files, instance):
    cache = []
    for i, file in enumerate(list_of_files):
        signal = instance.readout(file, instance)['signal'].flatten()
        cache.append(signal)

    max_signal = np.amax(np.array(cache), axis=0)
    mean = np.mean(max_signal)
    factor = []
    for signal in max_signal:
        if signal/mean <= 0.1:
            factor.append(0)
        else:
            factor.append(mean/signal)
    factor = np.array(factor)
    if len(factor) >= 64:
        factor[-3:] = 1
        factor[:2] = 1

    return factor


def normalization_from_translated_array(list_of_files, instance, method='least_squares', correction=0):
    # Little helper function to group the recalculated positions
    def group(input_list, group_range):
        class Group:
            def __init__(self, value):
                self.mean = value
                self.start_value = value
                self.members = 1

            def add(self, value):
                self.mean = (self.mean * self.members + value)/(self.members + 1)
                self.members += 1

            def check_add(self, value):
                return (self.mean * self.members + value)/(self.members + 1)

        if not isinstance(input_list, (np.ndarray, pd.DataFrame)):
            input_list = np.array(input_list)
        input_list = input_list.flatten()

        groups = []
        for j in input_list:
            if len(groups) == 0:
                groups.append(Group(j))
                continue
            in_group = False
            for group in groups:
                if group.mean - group_range <= j <= group.mean + group_range:
                    # check for drift of the group:
                    if group.start_value - group_range < group.check_add(j) < group.start_value + group_range:
                        group.add(j)
                        in_group = True
                        break

            if not in_group:
                groups.append(Group(j))

        return [g.mean for g in groups if g.members != 0]

    # Direction of the measurement: x or y
    if instance.diode_dimension[0] > instance.diode_dimension[1]:
        sp = 0
    elif instance.diode_dimension[0] < instance.diode_dimension[1]:
        sp = 1
    else:
        sp = 0
    # Load in the data from a list of files in a folder; save position and signal
    position = []
    signals = []
    for file in list_of_files:
        # The parsing of the position out of the name and save it
        position.append(instance.pos_parser(file))
        signal = instance.readout(file, instance)['signal']
        signals.append((np.array(signal)-instance.dark).flatten())

    # Sort the arrays by the position
    indices = np.argsort(np.array(position)[:, sp])
    signals = np.array(signals)[indices]
    position = np.array(position)[indices]

    # Recalculate the positions considering the size of the diodes and thus the expected real positions
    positions = []
    for i in range(np.shape(signals)[1]):
        cache = deepcopy(position)
        cache[:, sp] = cache[:, sp] + (int(np.shape(position)[0]/2) - i) * (instance.diode_size[sp] +
                                                                            instance.diode_spacing[sp] + correction)
        positions.append(cache)
    positions = np.array(positions)

    # Group the positions after their recalculation to gain a grid, from which the mean calculation is meaningful
    group_distance = instance.diode_size[sp]
    groups = group(positions[:, :, sp].flatten(), group_distance)

    # Calculate the mean for each grouped position, consider only the diode signals that were close to this position
    mean_result = []
    groups = np.sort(groups)
    for mean in groups:
        indices = []
        for k, channel in enumerate(np.array(positions)[:, :, sp]):
            index_min = np.argsort(np.abs(channel - mean))[0]
            if np.abs(channel[index_min] - mean) <= group_distance:
                indices.append(index_min)
            else:
                indices.append(None)
        cache = 0
        j = 0
        for i in range(len(indices)):
            if indices[i] is not None and signals[indices[i]][i] != 0:
                cache += signals[indices[i]][i]
                j += 1
        if j > 0:
            mean_result.append(cache / j)
        else:
            mean_result.append(cache)
    mean_result = np.array(mean_result)
    mean_x = np.array(groups)

    # Interpolation
    factor = np.array([])
    for channel in range(np.shape(signals)[1]):
        mean_interp = np.interp(positions[channel, :, sp], mean_x, mean_result)
        if isinstance(method, (float, int, np.float64)):
            # Method 1: Threshold for range consideration, for each diode channel mean of the factor between points
            factor2 = mean_interp[signals[:, channel] > method] / signals[:, channel][signals[:, channel] > method]
            factor2 = np.mean(factor2)
            if np.isnan(factor2):
                factor2 = 0
        elif method == 'least_squares':
            # Method 2: Optimization with least squares method, recommended
            func_opt = lambda a: mean_interp - signals[:, channel] * a
            factor2 = least_squares(func_opt, 1)
            if factor2.nfev == 1 and factor2.optimality == 0.0:
                factor2 = 0
            else:
                factor2 = factor2.x
        else:
            # Standard method: For the moment method 1 with automatic threshold
            factor2 = mean_interp[signals[:, channel] > np.mean(signals[:, channel])] / \
                      signals[:, channel][signals[:, channel] > np.mean(signals[:, channel])]
            factor2 = np.mean(factor2)
            if np.isnan(factor2):
                factor2 = 0
        factor = np.append(factor, factor2)
    # Save a raw factor to allow later changes on the factor's parameters
    return factor.reshape(instance.diode_dimension)


def normalization_from_translated_array_v2(list_of_files, instance, method='least_squares', correction=0):
    # Direction of the measurement: x or y
    if instance.diode_dimension[0] > instance.diode_dimension[1]:
        sp = 0
    elif instance.diode_dimension[0] < instance.diode_dimension[1]:
        sp = 1
    else:
        sp = 0
    # Load in the data from a list of files in a folder; save position and signal
    position = []
    signals = []
    for file in tqdm(list_of_files):
        # The parsing of the position out of the name and save it
        position.append(instance.pos_parser(file))
        signal = instance.readout(file, instance)['signal']
        signals.append((np.array(signal) - instance.dark).flatten())

    # Sort the arrays by the position
    indices = np.argsort(np.array(position)[:, sp])
    signals = np.array(signals)[indices]
    position = np.array(position)[indices]

    # Recalculate the positions considering the size of the diodes and thus the expected real positions
    positions = []
    for i in range(np.shape(signals)[1]):
        cache = deepcopy(position)
        cache[:, sp] = cache[:, sp] + i * (instance.diode_size[sp] + instance.diode_spacing[sp] + correction)
        positions.append(cache)
    positions = np.array(positions)

    # Try to detect the signal level
    threshold = ski_threshold_otsu(signals)
    print(threshold)

    # Group the positions after their recalculation to gain a grid, from which the mean calculation is meaningful
    group_distance = instance.diode_size[sp]
    groups = group(positions[:, :, sp].flatten(), group_distance)

    # Calculate the mean for each grouped position, consider only the diode signals that were close to this position
    mean_new = []
    mean_x_new = []
    groups = np.sort(groups)
    for mean in groups:
        indices = []
        for k, channel in enumerate(np.array(positions)[:, :, sp]):
            index_min = np.argsort(np.abs(channel - mean))[0]
            if np.abs(channel[index_min] - mean) <= group_distance:
                indices.append(index_min)
            else:
                indices.append(None)
        cache_new = 0
        j_new = 0
        for i in range(len(indices)):
            if indices[i] is not None and signals[indices[i]][i] >= threshold:
                cache_new += signals[indices[i]][i]
                j_new += 1
        if j_new > 0:
            mean_new.append(cache_new / j_new)
            mean_x_new.append(mean)

    mean_x = np.array(groups)
    mean_x_new, mean_new = np.array(mean_x_new), np.array(mean_new)

    # Interpolation
    factor_new = np.array([])
    for channel in range(np.shape(signals)[1]):
        restrained_position = (np.min(mean_x_new) <= positions[channel, :, sp]) & (
                    positions[channel, :, sp] <= np.max(mean_x_new))
        mean_interp_new = np.interp(positions[channel, :, sp][restrained_position], mean_x_new, mean_new)
        if isinstance(method, (float, int, np.float64)):
            # Method 1: Threshold for range consideration, for each diode channel mean of the factor between points
            factor_new_cache = mean_interp_new[signals[:, channel] > method] / signals[:, channel][signals[:, channel] > method]
            factor_new_cache = np.mean(factor_new_cache)
            if np.isnan(factor_new_cache):
                factor_new_cache = 0
        elif method == 'least_squares':
            # Method 2: Optimization with least squares method, recommended
            func_opt_new = lambda a: mean_interp_new - signals[:, channel][restrained_position] * a
            factor_new_cache = least_squares(func_opt_new, 1)
            if factor_new_cache.nfev == 1 and factor_new_cache.optimality == 0.0:
                factor_new_cache = 0
            else:
                factor_new_cache = factor_new_cache.x
        else:
            # Standard method: For the moment method 1 with automatic threshold
            factor_new_cache = mean_interp_new / signals[:, channel][signals[:, channel] > threshold]
            factor_new_cache = np.mean(factor_new_cache)
            if np.isnan(factor_new_cache):
                factor_new_cache = 0
        factor_new = np.append(factor_new, factor_new_cache)
    return factor_new.reshape(instance.diode_dimension)


def normalization_from_translated_array_v3(list_of_files, instance, method='least_squares', align_lines=True):
    # Load in the data from a list of files in a folder; save position and signal
    position = []
    signals = []
    for file in tqdm(list_of_files):
        # The parsing of the position out of the name and save it
        position.append(instance.pos_parser(file))
        signal = instance.readout(file, instance)['signal']
        signals.append((np.array(signal) - instance.dark))

    # Direction of the measurement: x or y
    x_pos = np.sort(np.array(list(set([i[0] for i in position]))))
    y_pos = np.sort(np.array(list(set([i[1] for i in position]))))
    pos = [x_pos, y_pos]

    # Choose the direction that is considered for normalisation
    if len(x_pos) > len(y_pos):
        sp = 0
    elif len(y_pos) > len(x_pos):
        sp = 1
    else:
        sp = 1

    # If multiple positions in the other directions are given, this version filters for the step with the largest number
    # of translation steps
    if len(pos[1-sp]) > 1:
        pos_choice = []
        for i in pos[1-sp]:
            pos_choice.append(len(np.array(list(set([j[sp] for j in position if j[1-sp] == i]))).sort()))
        choice = x_pos[np.argmax([pos_choice])]
        signals = [sig for i, sig in enumerate(signals) if position[i][1-sp] == choice]
        position = [i for i in position if i[1-sp] == choice]
        pos[sp] = np.array(list(set([i[sp] for i in position]))).sort()
        pos[1-sp] = np.array([choice])

    # Stop normalization if some easy testable conditions are not met (and normalization is not possible)
    if len(pos[0]) == 1 and len(pos[1]) == 1:
        print('This measurement is not usable for this normalization since there is effectively no translation'
              ' - thus, no factor can be calculated!')
        return None
    if instance.diode_dimension[sp] == 1:
        print('This measurement is not usable for this normalization since no diode overlay will exist for the given '
              'translation - thus, no factor can be calculated!')
        return None

    # Some info about steps and step width
    steps = len(pos[sp])
    step_width = np.mean([pos[sp][i+1] - pos[sp][i] for i in range(steps-1)])
    diode_periodicity = instance.diode_size[sp] + instance.diode_spacing[sp]
    if step_width > instance.diode_dimension[sp]*diode_periodicity:
        print('This measurement is not usable for a normalization since no diode overlay will exist - '
              'thus, no factor can be calculated!')
        return None

    print('Normalization is calculated from translation direction', ['x', 'y'][sp], 'with', steps,
          'at a mean step width of', step_width, 'mm for a diode periodicity of', diode_periodicity, 'mm.')

    # Sort the arrays by the position
    indices = np.argsort(np.array(position)[:, sp])
    signals = np.array(signals)[indices]
    position = np.array(position)[indices]

    # Main loop: For each diode line orthogonal to translation region calculate a factor
    factor_new = np.zeros(instance.diode_dimension)
    mean_cache = []
    for line in range(instance.diode_dimension[1-sp]):
        # Recalculate the positions considering the geometry of the diode array
        positions = []
        for i in range(instance.diode_dimension[sp]):
            cache = deepcopy(position)
            cache[:, sp] = cache[:, sp] + i * (instance.diode_size[sp] + instance.diode_spacing[sp]) + instance.diode_offset[1-sp][line]
            positions.append(cache)
        positions = np.array(positions)

        # Try to detect the signal level
        threshold = ski_threshold_otsu(signals[:, line])
        mean_over = np.mean(signals[(signals > threshold)])

        # Group the positions after their recalculation to gain a grid, from which the mean calculation is meaningful
        group_distance = instance.diode_size[sp]
        groups = group(positions[:, :, sp].flatten(), group_distance)

        # Calculate the mean for each grouped position, consider only the diode signals that were close to this position
        mean_new = []
        mean_x_new = []
        groups = np.sort(groups)
        for mean in groups:
            indices = []
            for k, channel in enumerate(np.array(positions)[:, :, sp]):
                index_min = np.argsort(np.abs(channel - mean))[0]
                if np.abs(channel[index_min] - mean) <= group_distance:
                    indices.append(index_min)
                else:
                    indices.append(None)
            cache_new = 0
            j_new = 0
            for i in range(len(indices)):
                if indices[i] is not None and threshold <= signals[indices[i], line, i] <= 1.5 * mean_over:
                    cache_new += signals[indices[i], line, i]
                    j_new += 1
            if j_new > 0:
                mean_new.append(cache_new / j_new)
                mean_x_new.append(mean)

        mean_x_new, mean_new = np.array(mean_x_new), np.array(mean_new)

        if align_lines:
            mean_cache.append([mean_x_new, mean_new])
        # Interpolation
        factor_cache = []
        for channel in range(instance.diode_dimension[sp]):
            restrained_position = (np.min(mean_x_new) <= positions[channel, :, sp]) & (
                        positions[channel, :, sp] <= np.max(mean_x_new))
            mean_interp_new = np.interp(positions[channel, :, sp][restrained_position], mean_x_new, mean_new)
            if isinstance(method, (float, int, np.float64)):
                # Method 1: Threshold for range consideration, for each diode channel mean of the factor between points
                factor_new_cache = mean_interp_new[signals[:, line, channel] > method] / signals[:, line, channel][signals[:, line, channel] > method]
                factor_new_cache = np.mean(factor_new_cache)
                if np.isnan(factor_new_cache):
                    factor_new_cache = 0
            elif method == 'least_squares':
                # Method 2: Optimization with least squares method, recommended
                func_opt_new = lambda a: mean_interp_new - signals[:, line, channel][restrained_position] * a
                factor_new_cache = least_squares(func_opt_new, 1)
                if factor_new_cache.nfev == 1 and factor_new_cache.optimality == 0.0:
                    factor_new_cache = 0
                else:
                    factor_new_cache = factor_new_cache.x[0]
            else:
                # Standard method: For the moment method 1 with automatic threshold
                factor_new_cache = mean_interp_new / signals[signals[:, line, channel] > threshold][:, line, channel]
                factor_new_cache = np.mean(factor_new_cache)
                if np.isnan(factor_new_cache):
                    factor_new_cache = 0

            factor_cache.append(factor_new_cache)
        factor_new[line] = np.array(factor_cache).flatten()

    if align_lines:
        # Calculate the mean of the mean from the different lines
        x_mean = mean_cache[0][0]
        mean_cache = [np.interp(x_mean, mean_cache[i][0], mean_cache[i][1]) for i in range(len(mean_cache))]
        overall_mean = np.mean(mean_cache)
        for line, m in enumerate(mean_cache):
            func_opt_new = lambda a: overall_mean - m * a
            factor_new_cache = least_squares(func_opt_new, 1)
            if factor_new_cache.nfev == 1 and factor_new_cache.optimality == 0.0:
                factor_new_cache = 1
            else:
                factor_new_cache = factor_new_cache.x[0]
            factor_new[line] = factor_new[line] * factor_new_cache
    return factor_new
