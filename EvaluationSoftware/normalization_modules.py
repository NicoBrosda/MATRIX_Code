import os
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from Plot_Methods.plot_standards import *
from tqdm import tqdm


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
    for file in tqdm(list_of_files):
        # The parsing of the position out of the name and save it
        position.append(instance.pos_parser(file))
        signal = instance.readout(file, instance, subtract_background=True)['signal']
        signals.append((np.array(signal)-instance.dark).flatten())

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