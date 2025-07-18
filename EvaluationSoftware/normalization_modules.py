import os
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from Plot_Methods.plot_standards import *
from tqdm import tqdm
from skimage.filters import threshold_otsu as ski_threshold_otsu
from scipy.ndimage import uniform_filter1d
from pybaselines import Baseline


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
        if threshold > np.median(signals[:, line]):
            threshold = np.median(signals[:, line]) * 0.6
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


def normalization_from_translated_array_v4(list_of_files, instance, method='least_squares', align_lines=True,
                                           remove_background=True, factor_limits=(0.9, 1.1), label=''):
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
    diff_new = np.zeros(instance.diode_dimension)
    mean_cache = []

    # results_path = Path("/Users/nico_brosda/Cyrce_Messungen/Results_260325/Homogeneity/Process/") / label
    results_path = Path("/Users/nico_brosda/Cyrce_Messungen/Results_260325/Homogeneity/Process/")

    # -----------------------------------------------------------------------------------------------------------------
    # Plot in loop: Baseline Mean Before / after shift
    # -----------------------------------------------------------------------------------------------------------------
    line_color = sns.color_palette("tab10")
    fig, ax = plt.subplots()

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
        if threshold > np.median(signals[:, line]):
            threshold = np.median(signals[:, line]) * 0.6
        mean_over = np.mean(signals[(signals > threshold)])
        threshold = mean_over * 0.8

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
        diff_cache = []

        diode_cmap = sns.color_palette("coolwarm", as_cmap=True)
        diode_colormapper = lambda diode: color_mapper(diode, 0, instance.diode_dimension[sp])
        diode_color = lambda diode: diode_cmap(diode_colormapper(diode))
        for channel in range(instance.diode_dimension[sp]):
            restrained_position = (np.min(mean_x_new) <= positions[channel, :, sp]) & (
                        positions[channel, :, sp] <= np.max(mean_x_new))
            mean_interp_new = np.interp(positions[channel, :, sp][restrained_position], mean_x_new, mean_new)
            ax.plot(positions[channel, :, sp][restrained_position], mean_interp_new, c=line_color[line], zorder=5)
            ax.plot(positions[channel, :, sp][restrained_position], signals[:, line, channel][restrained_position], c=diode_color(channel), alpha=0.2, zorder=-1)
            diff = 0
            if isinstance(method, (float, int, np.float64)):
                # Method 1: Threshold for range consideration, for each diode channel mean of the factor between points
                factor_new_cache = mean_interp_new[signals[:, line, channel] > method] / signals[:, line, channel][signals[:, line, channel] > method]
                factor_new_cache = np.mean(factor_new_cache)
                if np.isnan(factor_new_cache):
                    factor_new_cache = 0
            elif method == 'least_squares':
                # Method 2: Optimization with least squares method, recommended
                func_opt_new = lambda a: np.abs(mean_interp_new - signals[:, line, channel][restrained_position] * a)
                factor_new_cache = least_squares(func_opt_new, 1)
                if factor_new_cache.nfev == 1 and factor_new_cache.optimality == 0.0:
                    factor_new_cache = 0
                else:
                    diff = np.mean(factor_new_cache.fun)/mean_over
                    factor_new_cache = factor_new_cache.x[0]

                ax.plot(positions[channel, :, sp][restrained_position], signals[:, line, channel][restrained_position]*factor_new_cache, c=diode_color(channel), alpha=0.8)
            else:
                # Standard method: For the moment method 1 with automatic threshold
                factor_new_cache = mean_interp_new / signals[signals[:, line, channel] > threshold][:, line, channel]
                factor_new_cache = np.mean(factor_new_cache)
                if np.isnan(factor_new_cache):
                    factor_new_cache = 0

            factor_cache.append(factor_new_cache)
            diff_cache.append(diff)
        factor_new[line] = np.array(factor_cache).flatten()
        diff_new[line] = np.array(diff_cache).flatten()

    # -----------------------------------------------------------------------------------------------------------------
    ax.set_xlim(ax.get_xlim()), ax.set_ylim(ax.get_ylim())
    y1, y2 = ax.get_ylim()
    if y1 < 0.7 * mean_over:
        y1 = 0.7 * mean_over
    if y2 > 1.3 * mean_over:
        y2 = 1.3 * mean_over
    ax.set_ylim(y1, y2)
    ax.text(*transform_axis_to_data_coordinates(ax, [0.04, 0.93]), label, fontsize=12)
    ax.set_xlabel('Diode channel')
    ax.set_ylabel('Signal Level (ams unit)')
    format_save(results_path, f"1NormProcess_{label}", legend=False, fig=fig)
    # -----------------------------------------------------------------------------------------------------------------

    # Set the mean of each line of the factor to 1
    line_masks = []
    for line in range(instance.diode_dimension[1 - sp]):
        mask = (factor_limits[0] < factor_new[line] ) & (factor_limits[1] > factor_new[line])
        factor_new[line][mask] = factor_new[line][mask] - np.mean(factor_new[line][mask]) + 1
        line_masks.append(mask)

    # The module to fit underlying beam instabilities by an n-th order polynomial
    '''
    def poly_n(x, *args):
        args = np.array(args).flatten()
        print(np.shape(x))
        print(np.shape(args))
        print(args)
        return np.sum([args[i] * x**i for i in range(len(args))], axis=0)

    def cost_func(*args):
        x = np.arange(0, instance.diode_dimension[sp], 1)
        poly = poly_n(x, *args).flatten()
        return np.sum(np.abs([factor_new[i]-poly for i in range(instance.diode_dimension[1-sp])]))
    '''

    if remove_background:
        # --------------------------------------------------------------------------------------------------------------
        # Plot: Comparison Baseline Polynomial orders
        # --------------------------------------------------------------------------------------------------------------
        fig, ax = plt.subplots()
        x = np.arange(0, 64, 1)
        color = sns.color_palette("Spectral", as_cmap=True)
        for i in range(1, 9):
            baseline_fitter = Baseline(x_data=x)
            baseline_data = np.mean(factor_new, axis=0)
            baseline_data[((factor_limits[0] > baseline_data) | (factor_limits[1] < baseline_data))] = 1
            baseline = baseline_fitter.imodpoly(baseline_data, poly_order=i, num_std=0.7)[0]
            '''
            if baseline.nfev == 1 and baseline.optimality == 0.0:
                baseline = [0]
            else:
                baseline = baseline.x
            '''
            ax.plot(x, baseline, label=f"{i}th order polynom", c=color(i/9))

        for line in range(instance.diode_dimension[1-sp]):
            ax.plot(factor_new[line], ls='--', zorder=5, c=line_color[line])
        ax.set_xlim(ax.get_xlim()), ax.set_ylim(ax.get_ylim())
        y1, y2 = ax.get_ylim()
        if y1 < 0.9:
            y1 = 0.9
        if y2 > 1.1:
            y2 = 1.1
        ax.set_ylim(y1, y2)
        ax.text(*transform_axis_to_data_coordinates(ax, [0.04, 0.93]), label, fontsize=12)
        ax.set_xlabel('Diode channel')
        ax.set_ylabel('Signal Homogeneity')
        format_save(results_path, f"2Baseline_Order_{label}", legend=True)
        # --------------------------------------------------------------------------------------------------------------

        i = 7
        x = np.arange(0, 64, 1)
        baseline_fitter = Baseline(x_data=x)
        baseline_data = np.mean(factor_new, axis=0)
        baseline_data[((factor_limits[0] > baseline_data) | (factor_limits[1] < baseline_data))] = 1
        baseline = baseline_fitter.imodpoly(baseline_data, poly_order=i, num_std=0.7)[0]

        # --------------------------------------------------------------------------------------------------------------
        # Plot: Comparison Factor / Baseline Mean
        # --------------------------------------------------------------------------------------------------------------
        fig, ax = plt.subplots()
        x = np.arange(0, 64, 1)
        color = sns.color_palette("Spectral", as_cmap=True)

        ax.plot(x, baseline, label=f"{i}th order polynom", c=color(i / 9))
        ax.axhline(np.mean(baseline), ls='--', c=color(i / 9))

        for line in range(instance.diode_dimension[1 - sp]):
            ax.plot(factor_new[line][line_masks[line]], ls='-', zorder=5, c=line_color[line])
            ax.axhline(np.mean(factor_new[line][line_masks[line]]), ls='--', c=line_color[line])
        ax.set_xlim(ax.get_xlim()), ax.set_ylim(ax.get_ylim())
        y1, y2 = ax.get_ylim()
        if y1 < 0.9:
            y1 = 0.9
        if y2 > 1.1:
            y2 = 1.1
        ax.set_ylim(y1, y2)
        ax.text(*transform_axis_to_data_coordinates(ax, [0.04, 0.93]), label, fontsize=12)
        ax.set_xlabel('Diode channel')
        ax.set_ylabel('Signal Homogeneity')
        format_save(results_path, f"3Baseline_Mean_{label}", legend=True)
        # --------------------------------------------------------------------------------------------------------------

    if align_lines:
        # --------------------------------------------------------------------------------------------------------------
        # Plot: Alignment process
        # --------------------------------------------------------------------------------------------------------------
        fig, ax = plt.subplots()

        # Calculate the mean of the mean from the different lines
        x_mean = mean_cache[0][0]
        mean_cache = [np.interp(x_mean, mean_cache[i][0], mean_cache[i][1]) for i in range(len(mean_cache))]

        overall_mean = np.mean(mean_cache, axis=0)
        ax.plot(x_mean, overall_mean, c='k', label='Overall mean')

        for line, m in enumerate(mean_cache):
            ax.plot(x_mean, m, ls='-', c=line_color[line])
            func_opt_new = lambda a: np.abs(overall_mean - m * a)
            factor_new_cache = least_squares(func_opt_new, 1)
            if factor_new_cache.nfev == 1 and factor_new_cache.optimality == 0.0:
                factor_new_cache = 1
            else:
                factor_new_cache = factor_new_cache.x[0]
            factor_new[line][line_masks[line]] = factor_new[line][line_masks[line]] * factor_new_cache
            ax.plot(x_mean, m*factor_new_cache, ls='--', c=line_color[line], label=f"{(1-factor_new_cache)*100:.2f}% offset from common mean")
        ax.set_xlim(ax.get_xlim()), ax.set_ylim(ax.get_ylim())
        y1, y2 = ax.get_ylim()
        if y1 < 0.7 * np.mean(overall_mean):
            y1 = 0.7 * np.mean(overall_mean)
        if y2 > 1.3 * np.mean(overall_mean):
            y2 = 1.3 * np.mean(overall_mean)
        ax.set_ylim(y1, y2)
        ax.text(*transform_axis_to_data_coordinates(ax, [0.04, 0.93]), label, fontsize=12)
        ax.set_xlabel('Diode channel')
        ax.set_ylabel('Signal Level (ams unit)')
        format_save(results_path, f"4Align procedure_{label}", legend=True, fig=fig)
        # --------------------------------------------------------------------------------------------------------------

    if remove_background:
        for line in range(instance.diode_dimension[1 - sp]):
            factor_new[line][line_masks[line]] = (factor_new[line][line_masks[line]] - baseline[line_masks[line]]
                                                  + np.mean(baseline))

    return factor_new, diff_new


def normalization_from_translated_array_v5(list_of_files, instance, method='least_squares', align_lines=True,
                                           remove_background=True, factor_limits=(0.9, 1.1)):
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
    diff_new = np.zeros(instance.diode_dimension)
    mean_cache = []

    # -----------------------------------------------------------------------------------------------------------------
    # Plot in loop: Baseline Mean Before / after shift
    # -----------------------------------------------------------------------------------------------------------------
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
        if threshold > np.median(signals[:, line]):
            threshold = np.median(signals[:, line]) * 0.6
        mean_over = np.mean(signals[(signals > threshold)])
        threshold = mean_over * 0.8

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
        diff_cache = []

        for channel in range(instance.diode_dimension[sp]):
            restrained_position = (np.min(mean_x_new) <= positions[channel, :, sp]) & (
                        positions[channel, :, sp] <= np.max(mean_x_new))
            mean_interp_new = np.interp(positions[channel, :, sp][restrained_position], mean_x_new, mean_new)
            diff = 0
            if isinstance(method, (float, int, np.float64)):
                # Method 1: Threshold for range consideration, for each diode channel mean of the factor between points
                factor_new_cache = mean_interp_new[signals[:, line, channel] > method] / signals[:, line, channel][signals[:, line, channel] > method]
                factor_new_cache = np.mean(factor_new_cache)
                if np.isnan(factor_new_cache):
                    factor_new_cache = 0
            elif method == 'least_squares':
                # Method 2: Optimization with least squares method, recommended
                func_opt_new = lambda a: np.abs(mean_interp_new - signals[:, line, channel][restrained_position] * a)
                factor_new_cache = least_squares(func_opt_new, 1)
                if factor_new_cache.nfev == 1 and factor_new_cache.optimality == 0.0:
                    factor_new_cache = 0
                else:
                    diff = np.mean(factor_new_cache.fun)/mean_over
                    factor_new_cache = factor_new_cache.x[0]
            else:
                # Standard method: For the moment method 1 with automatic threshold
                factor_new_cache = mean_interp_new / signals[signals[:, line, channel] > threshold][:, line, channel]
                factor_new_cache = np.mean(factor_new_cache)
                if np.isnan(factor_new_cache):
                    factor_new_cache = 0

            factor_cache.append(factor_new_cache)
            diff_cache.append(diff)
        factor_new[line] = np.array(factor_cache).flatten()
        diff_new[line] = np.array(diff_cache).flatten()

    # Set the mean of each line of the factor to 1
    line_masks = []
    for line in range(instance.diode_dimension[1 - sp]):
        mask = (factor_limits[0] < factor_new[line] ) & (factor_limits[1] > factor_new[line])
        factor_new[line][mask] = factor_new[line][mask] - np.mean(factor_new[line][mask]) + 1
        line_masks.append(mask)

    if remove_background:
        i = 7
        x = np.arange(0, 64, 1)
        baseline_fitter = Baseline(x_data=x)
        baseline_data = np.mean(factor_new, axis=0)
        baseline_data[((factor_limits[0] > baseline_data) | (factor_limits[1] < baseline_data))] = 1
        baseline = baseline_fitter.imodpoly(baseline_data, poly_order=i, num_std=0.7)[0]

    if align_lines:
        # Calculate the mean of the mean from the different lines
        x_mean = mean_cache[0][0]
        mean_cache = [np.interp(x_mean, mean_cache[i][0], mean_cache[i][1]) for i in range(len(mean_cache))]

        overall_mean = np.mean(mean_cache, axis=0)

        for line, m in enumerate(mean_cache):
            func_opt_new = lambda a: np.abs(overall_mean - m * a)
            factor_new_cache = least_squares(func_opt_new, 1)
            if factor_new_cache.nfev == 1 and factor_new_cache.optimality == 0.0:
                factor_new_cache = 1
            else:
                factor_new_cache = factor_new_cache.x[0]
            factor_new[line][line_masks[line]] = factor_new[line][line_masks[line]] * factor_new_cache

    if remove_background:
        for line in range(instance.diode_dimension[1 - sp]):
            factor_new[line][line_masks[line]] = (factor_new[line][line_masks[line]] - baseline[line_masks[line]]
                                                  + np.mean(baseline))

    return factor_new, diff_new
