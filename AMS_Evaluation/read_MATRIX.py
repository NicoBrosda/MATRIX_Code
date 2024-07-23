import numpy as np
import pandas as pd
from copy import deepcopy
from AMS_Evaluation.DataAnalysis import threshold_otsu
from Plot_Methods.plot_standards import *
import matplotlib
from mpl_toolkits.mplot3d import axes3d
from FitFuncs import apply_super_resolution
from matplotlib.colors import BoundaryNorm
from scipy.optimize import least_squares


# Function for reading the csv files and create a data array out of it. Note that the return is a pd.Dataframe Object
# containing 64 channels and ca. 5000 rows = samples of each channel.
def read(file_path, fast=False) -> pd.DataFrame:
    if fast:
        data = pd.read_csv(file_path, delimiter=',', usecols=range(66))
    else:
        detect = len(pd.read_csv(file_path, delimiter=',', nrows=1, header=None).columns)
        # print('!!!!!!!!!!!!!', detect)  # Previous 195 fpr e2
        data1 = pd.read_csv(file_path, delimiter=',', nrows=1, header=None, usecols=range(130, detect))
        data2 = pd.read_csv(file_path, delimiter=',', usecols=range(66))
        if len(data2.columns) - len(data1.columns) > 0:
            for i in range(len(data2.columns) - len(data1.columns)):
                data1['Add'+str(i)] = np.NaN
        data1 = data1.set_axis(list(data2.columns), axis=1)
        data = pd.concat([data1, data2])
    return data


# Function to convert the data of each channel into values representing the signal response (as on - off amplitude)
# - standard - or return of separate arrays: signal, signal_std, dark, dark_std, thresholds - advanced_output = True
# The signal and dark amplitudes of each diode are calculated with the threshold_otsu algorithm, designed to find the
# separating value of 2 Gaussian populations.
def read_channels(data, excluded_channel=[], advanced_output=False, varied_beam=True, path_dark=None) -> list or np.ndarray:
    """
    Function to convert the data of each channel into values representing the signal response (as on - off amplitude)
    - standard - or return of separate arrays: signal, signal_std, dark, dark_std, thresholds - advanced_output = True
    The signal and dark amplitudes of each diode are calculated with the threshold_otsu algorithm, designed to find the
    separating value of 2 Gaussian populations.
    :param data: input data, pd.DataFrame or np.NdArray in required DataStructure (channels = rows, rows = samples)
    :param excluded_channel: array with indices of excluded channels, that are added as signal=0 for further processing
    :param advanced_output: advanced return
    :param varied_beam: Defines if the beam was varied on/off or constant on in measurements - standard is on/off
    :return:
    """
    signals = []
    signal_std = []
    if not varied_beam:
        # Load in one or multiple dark, measurements - calculate their mean - subtract from the signal
        dark = []
        if path_dark is not None:
            if not isinstance(path_dark, (tuple, list, np.ndarray)):
                path_dark = [path_dark]
            for file2 in path_dark:
                dark.append(read_channels(read(file2), excluded_channel=excluded_channel, varied_beam=False))
            dark = np.mean(np.array(dark), axis=0)
        else:
            dark = np.zeros(np.shape(data)[1])
        for i, col in enumerate(data):
            if i in excluded_channel:
                print('Yes')
                if advanced_output:
                    signals.append(0)
                    signal_std.append(0)
                else:
                    signals.append(0)
            if i == 0 or i == 1:
                continue
            try:
                # print(np.mean(data[col])-dark[i-2])
                signals.append(np.mean(data[col])-dark[i-2])
                signal_std.append(np.std(data[col]))
            except ValueError:
                print('Yes')
                signals.append(0)
                signal_std.append(0)
        if advanced_output:
            return [np.array(signals)[:len(signals)-len(excluded_channel)],
                    np.array(signal_std)[:len(signals)-len(excluded_channel)]]
        else:
            return np.array(signals)[:len(signals)-len(excluded_channel)]

    else:
        thresholds = []
        signals = []
        signal_std = []
        darks = []
        dark_std = []
        for i, col in enumerate(data):
            if i in excluded_channel:
                if advanced_output:
                    signals.append(0)
                    darks.append(0)
                    signal_std.append(0)
                    dark_std.append(0)
                    thresholds.append(None)
                else:
                    signals.append(0)
            if i == 0 or i == 1:
                continue
            try:
                threshold = threshold_otsu(data[col])
            except ValueError:
                threshold = None
            thresholds.append(threshold)

            if threshold is not None:
                if advanced_output:
                    signals.append(np.mean(data[col][data[col] > threshold]))
                    signal_std.append(np.std(data[col][data[col] > threshold]))
                    darks.append(np.mean(data[col][data[col] < threshold]))
                    dark_std.append(np.std(data[col][data[col] < threshold]))
                else:
                    sig = np.mean(data[col][data[col] > threshold])
                    dar = np.mean(data[col][data[col] < threshold])
                    if sig is not np.nan and dar is not np.nan:
                        signals.append(sig - dar)
                    elif sig is not np.nan and dar is np.nan:
                        signals.append(sig)
                    else:
                        signals.append(0)
            else:
                signals.append(0)
                darks.append(0)
        if advanced_output:
            return [np.array(signals)[:len(signals)-len(excluded_channel)],
                    np.array(signal_std)[:len(signal_std)-len(excluded_channel)],
                    np.array(darks)[:len(darks)-len(excluded_channel)],
                    np.array(dark_std)[:len(dark_std)-len(excluded_channel)],
                    np.array(thresholds)[:len(thresholds)-len(excluded_channel)]]
        else:
            return np.array(signals)[:len(signals)-len(excluded_channel)]


def normalization(paths_of_norm_files, excluded_channel=[]):
    cache = []
    for i, path in enumerate(paths_of_norm_files):
        data = read(path)
        cache.append(read_channels(data, excluded_channel=excluded_channel))

    max_signal = np.amax(np.array(cache), axis=0)
    '''
    max_signal = []
    for i, measurement in enumerate(cache):
        for j, channel in enumerate(measurement[0]):
            if i == 0:
                max_signal.append(channel)
            elif channel > max_signal[j]:
                max_signal[j] =
    '''
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
    # Normalize factor:
    try:
        factor = factor / np.mean(factor)
    except ValueError:
        pass
    return factor


def normalization_new(path_to_folder, list_of_files, excluded_channel=[], scan_direction='y', method='leastsquares',
                      dark_path=None, diode_size=(0.5, 0.5), diode_space=0.08, cache_save=True):
    # Check if factor is already saved and is not needed to be recalculated:
    if os.path.isfile(path_to_folder / 'normalization_factor.npy'):
        try:
            factor = np.load(path_to_folder / 'normalization_factor.npy')
            return factor
        except ValueError:
            pass

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
    if scan_direction == 'x':
        sp = 0
    elif scan_direction == 'y':
        sp = 1
    else:
        sp = 0
    # Load in the data from a list of files in a folder; save position and signal
    path_to_folder = Path(path_to_folder)
    position = []
    signals = []
    for file in list_of_files:
        # The parsing of the position out of the name and save it
        try:
            index3 = file.index('.csv')
            index2 = file.index('_y_')
            index1 = file.index('_x_')
            pos_x = float(file[index1 + 3:index2])
            pos_y = float(file[index2 + 3:index3])
        except ValueError:
            continue
        position.append(np.array([pos_x, pos_y]))

        data = read(path_to_folder / file)
        signal = read_channels(data, excluded_channel=excluded_channel, varied_beam=False, path_dark=dark_path)
        signals.append(np.array(signal))

    # Sort the arrays by the position
    indices = np.argsort(np.array(position)[:, sp])
    signals = np.array(signals)[indices]
    position = np.array(position)[indices]

    # Recalculate the positions considering the size of the diodes and thus the expected real positions
    positions = []
    for i in range(np.shape(position)[0]):
        cache = deepcopy(position)
        cache[:, sp] = cache[:, sp] + (int(np.shape(position)[0]/2) - i) * (diode_size[sp] + diode_space)
        positions.append(cache)
    positions = np.array(positions)

    # Group the positions after their recalculation to gain a grid, from which the mean calculation is meaningful
    group_distance = diode_size[sp]
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
    print(np.shape(positions)[0])
    for channel in range(np.shape(positions)[0]):
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

    if cache_save:
        np.save(path_to_folder / 'normalization_factor.npy', factor)
    return factor


def interpret_map(folder, criteria, save_path='', plot=False, paths_of_norm_files=None, excluded_channel=[],
                  do_normalization=True, convert_param=True, diode_size=(0.5, 0.5), diode_space=0.08, x_stepwidth=0.25,
                  super_resolution=False, contour=True, realistic=False, avoid_sample=True,
                  Z_conversion=lambda Z: Z, varied_beam=True, *args, **kwargs):
    files = os.listdir(Path(folder))
    cache = []
    for file in files:
        if file[-4:] == '.csv' and criteria in file:
            cache.append(file)
    files = cache
    if len(files) == 0:
        print('No according files found in the given directory under the search criteria!')
        return None
    print(len(files), 'files found to construct a map!')

    position = []
    readout = []
    for file in files:
        # The parsing of the position out of the name and save it
        i = 1
        pos = None
        while True:
            try:
                index = file.index('.csv')
                pos = float(file[(index-i):index])
            except ValueError:
                break
            i += 1
        position.append(pos)
        if pos is None:
            continue

        # For each file read the channels (and apply normalization) - saved under same ordering
        data2 = read(Path(folder) / file)
        readout2 = read_channels(data2, excluded_channel, varied_beam=varied_beam)
        # Normalization
        if do_normalization:
            if paths_of_norm_files is not None:
                factor = normalization(paths_of_norm_files, excluded_channel=excluded_channel)
                readout2 = readout2*factor
        readout.append(readout2)

    # print(position, readout)
    position, readout = np.array(position), np.array(readout)
    sorting = np.argsort(position)
    readout = readout[sorting]
    position = position[sorting]
    if not plot:
        return position, readout
    else:
        fig, ax = plt.subplots()
        print('Shape of the readout array', np.shape(readout))
        print(len(readout[0]))
        channels = np.arange(0, np.size(readout[1]), 1)
        X, Y, Z = position, channels, readout.T
        Z = Z[:][::-1]
        print('Shape of the readout array after mirroring', np.shape(Z))
        print(len(Z[0]))
        if super_resolution:
            if len(X) % 2 != 0:
                X = X[1:]
                Z = Z[:, 1:]
            Z = apply_super_resolution(Z)
        elif x_stepwidth < diode_size[0] and avoid_sample:
            X = X[::2]
            Z = Z[:, ::2]
        if convert_param:
            if realistic and not contour:
                X = x_stepwidth * X
                X = np.append(X, (X[-1]+x_stepwidth))
                cache = []
                y = 0
                for i in range(len(Y)*2+1):
                    cache.append(y)
                    if i % 2 == 0:
                        y += diode_size[1]
                    else:
                        y += diode_space
                Y = np.array(cache)
                cache = []
                for row in Z:
                    cache.append(row)
                    cache.append(np.full_like(row, 0))
                Z = np.array(cache)
            else:
                X = x_stepwidth * X  # Defining X as translated position in mm
                # Defining the Y conversion based on the geometry of the diodes
                Y = Y*(diode_size[1]+diode_space)
                # Defining the conversion of amplitude z into a current pA
                Z = Z_conversion(Z)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black", "red", "yellow"])

        if np.max(Z) > 8.5 * np.mean(Z):
            intensity_limits = [0, 1500]
        else:
            intensity_limits = [0, np.max(Z)*0.85]
        intensity_limits = [0, 1600]
        intensity_limits = [0, 8500]

        if 'beamshape' in criteria:
            intensity_limits = [0, np.max(Z)*0.85]
        if super_resolution:
            intensity_limits = np.array(intensity_limits)/2
        # intensity_limits2 = (max(np.min(Z), intensity_limits[0]), min(np.max(Z), intensity_limits[1]))
        intensity_limits2 = intensity_limits
        levels = np.linspace(intensity_limits2[0], intensity_limits2[1], 100)
        print(intensity_limits)
        print(intensity_limits2)
        if not contour:
            norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
            if realistic:
                color_map = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm, shading='flat')
            else:
                color_map = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm, *args, **kwargs)
            norm = matplotlib.colors.Normalize(vmin=intensity_limits2[0], vmax=intensity_limits2[1])
            sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map.cmap)
            sm.set_array([])
            bar = fig.colorbar(sm, ax=ax, extend='max')
        else:
            # color_map = ax.contourf(X, Y, Z, cmap=cmap, extend='neither', levels=levels, *args, **kwargs)
            if np.min(Z) < intensity_limits[0] and np.max(Z) > intensity_limits[1]:
                color_map = ax.contourf(X, Y, Z, cmap=cmap, extend='both', levels=levels)
            elif np.min(Z) < intensity_limits[0]:
                color_map = ax.contourf(X, Y, Z, cmap=cmap, extend='min', levels=levels)
            elif np.max(Z) > intensity_limits[1]:
                color_map = ax.contourf(X, Y, Z, cmap=cmap, extend='max', levels=levels)
            else:
                color_map = ax.contourf(X, Y, Z, cmap=cmap, extend='neither', levels=levels)
            # '''
            norm = matplotlib.colors.Normalize(vmin=intensity_limits2[0], vmax=intensity_limits2[1])
            sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map.cmap)
            sm.set_array([])
            bar = fig.colorbar(sm, ax=ax, extend='max', ticks=color_map.levels)
        if convert_param:
            ax.set_xlabel(r'Position Translation Stage (mm)')
            ax.set_ylabel(r'Position Diode Array (mm)')
            bar.set_label('Measured Amplitude')

        else:
            ax.set_xlabel(r'Position ($\#$Steps)')
            ax.set_ylabel(r'Position ($64 - \#$ Diode Channel)')
            bar.set_label('Measured Amplitude')

        save_name = str(criteria)+'_map'
        if not do_normalization:
            save_name += '_nonorm'
        if not convert_param:
            save_name += '_ams'
        if super_resolution:
            save_name += '_superres'
        if contour:
            save_name += '_contour'
        if realistic:
            save_name += '_realistic'
        if not avoid_sample:
            save_name += '_sampling'
        format_save(save_path=save_path, save_name=save_name)
        plt.show()
        return position, readout


def interpret_2Dmap(folder, criteria, save_path='', plot=False, paths_of_norm_files=None, excluded_channel=[],
                    do_normalization=True, convert_param=True, diode_direction='y', array_len=64, diode_size=(0.5, 0.5),
                    diode_space=0.08, x_stepwidth=0.25, y_stepwidth=0.25, super_resolution=False, contour=True,
                    realistic=False, avoid_sample=True, Z_conversion=lambda Z: Z, varied_beam=True, *args, **kwargs):
    # This part remains the same between 1D map naming and 2D mapping
    files = os.listdir(Path(folder))
    cache = []
    for file in files:
        if file[-4:] == '.csv' and criteria in file:
            cache.append(file)
    files = cache
    if len(files) == 0:
        print('No according files found in the given directory under the search criteria!')
        return None
    print(len(files), 'files found to construct a map!')

    # ToDo: Rewrite the positional readout into a 2D array
    position = []
    readout = []
    for file in files:
        # The parsing of the position out of the name and save it
        try:
            index3 = file.index('.csv')
            index2 = file.index('_y_')
            index1 = file.index('_x_')
            pos_x = float(file[index1+3:index2])
            pos_y = float(file[index2+3:index3])
        except ValueError:
            continue
        position.append(np.array([pos_x, pos_y]))

        # For each file read the channels (and apply normalization) - saved under same ordering
        data2 = read(Path(folder) / file)
        readout2 = read_channels(data2, excluded_channel, varied_beam=varied_beam)

        # ToDo: Check if normalization is still viable or a better method is available (y-shifted measurements)
        # Normalization
        if do_normalization:
            if paths_of_norm_files is not None:
                factor = normalization(paths_of_norm_files, excluded_channel=excluded_channel)
                readout2 = readout2*factor
        readout.append(readout2)

    print(np.shape(position), np.shape(readout))

    distinct_x = sorted(set([i[0] for i in position]))
    distinct_y = sorted(set([i[1] for i in position]))
    if diode_direction == 'y':
        distinct = distinct_y
        print('There are ', len(distinct_x), ' different steps in the x direction (orthogonal to diode array)')
        print('There are ', len(distinct_y), ' different positions in the y direction (direction of diode array)')
        if len(distinct_y) == 1:
            print('Out of 1 y row a standard map is calculated, no overlap in the y direction needs to be considered.')
        else:
            d = [(distinct_y[i+1]-distinct_y[i])/(diode_size[1]+diode_space) for i in range(len(distinct_y)-1)]
            print('Multiple y rows are found. This corresponds a distance between measurement points of ',
                  [round(i) for i in d], ' # diodes - thus leaving an overlap of ',
                  [round(abs(array_len/2-i))+round(array_len/2) for i in d], ' # diodes')
    else:
        distinct = distinct_x
        print('There are ', len(distinct_x), ' different steps in the x direction (direction of the diode array)')
        print('There are ', len(distinct_y), ' different positions in the y direction (orthogonal to diode array)')
        if len(distinct_x) == 1:
            print('Out of 1 x row a standard map is calculated, no overlap in the y direction needs to be considered.')
        else:
            d = [(distinct_x[i+1]-distinct_x[i])/(diode_size[1]+diode_space) for i in range(len(distinct_x)-1)]
            print('Multiple x rows are found. This corresponds a distance between measurement points of ',
                  [round(i) for i in d], ' # diodes - thus leaving an overlap of ',
                  [round(abs(array_len/2-i))+round(array_len/2) for i in d], ' # diodes')
    position, readout = np.array(position), np.array(readout)

    '''
    fig, ax = plt.subplots()
    for pos in position:
        i = 0
        while True:
            if pos[1] == distinct_y[i]:
                break
            else:
                i += 1
        ax.scatter(*pos, marker='x', color=sns.color_palette("tab10")[i % len(sns.color_palette("tab10"))])
    for i, j in enumerate(distinct_y):
        color = sns.color_palette("tab10")[i % len(sns.color_palette("tab10"))]
        x = min(distinct_x) + i
        for k in range(int(array_len/2)):
            rect1 = mpl.patches.Rectangle((x, j-k*(diode_size[1]+diode_space)), *diode_size, edgecolor=color, facecolor='none')
            rect2 = mpl.patches.Rectangle((x, j + k * (diode_size[1] + diode_space)), *diode_size, edgecolor=color, facecolor='none')
            ax.add_patch(rect1)
            ax.add_patch(rect2)
    ax.set_xlabel('Position x direction')
    ax.set_ylabel('Position y direction')
    format_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_19062024/', criteria)
    '''

    # ToDo: Sort the position after x and y - sort the readout accordingly
    sorting = np.argsort(position[:, 0])
    readout = readout[sorting]
    position = position[sorting]
    sorting = np.argsort(position[:, 1], kind='mergesort')
    readout = readout[sorting]
    position = position[sorting]

    # ToDo: Convert the information from diode array direction and - if available - translation steps in this direction
    #  into 1 coherent Z information
    # ToDo: Before - Just plot a map for all Z steps
    if not plot:
        return position, readout
    else:
        # ToDo: Adapt the plotting, maybe with more different options
        for step in distinct:
            if diode_direction == 'y':
                indices = np.where(position[:, 1] == step)[0]
                print(position[:, 1])
                print(indices)
                if len(indices) <= 1:
                    print('For 1 translation step no map is created!')
                    continue
                X = position[:, 0][indices]
                x_stepwidth = X[1]-X[0]
                channels = np.arange(0, np.size(readout[1]), 1)
                Y = channels
                print('Shape of the readout array', np.shape(readout))
                Z = readout[indices].T
                # Z = Z[:][::-1]
                print('Shape of the readout array after mirroring', np.shape(Z))
                print(len(Z[0]))
            else:
                indices = np.where(position[:, 0] == step)[0]
                if len(indices) <= 1:
                    print('For 1 translation step no map is created!')
                    continue
                Y = position[:, 1][indices]
                y_stepwidth = Y[1]-Y[0]
                channels = np.arange(0, np.size(readout[1]), 1)
                X = channels
                print('Shape of the readout array', np.shape(readout))
                Z = readout[indices]
                print('Shape of the readout array after mirroring', np.shape(Z))
                print(len(Z[0]))

            fig, ax = plt.subplots()
            if super_resolution:
                if len(X) % 2 != 0:
                    X = X[1:]
                    Z = Z[:, 1:]
                Z = apply_super_resolution(Z)
            elif x_stepwidth < diode_size[0] and avoid_sample:
                X = X[::2]
                Z = Z[:, ::2]
            if convert_param:
                if realistic and not contour:
                    X = np.append(X, (X[-1]+x_stepwidth))
                    cache = []
                    y = 0
                    for i in range(len(Y)*2+1):
                        cache.append(y)
                        if i % 2 == 0:
                            y += diode_size[1]
                        else:
                            y += diode_space
                    Y = np.array(cache)
                    cache = []
                    for row in Z:
                        cache.append(row)
                        cache.append(np.full_like(row, 0))
                    Z = np.array(cache)
                else:
                    # Defining the Y conversion based on the geometry of the diodes
                    Y = Y*(diode_size[1]+diode_space)
                    # Defining the conversion of amplitude z into a current pA
                    Z = Z_conversion(Z)
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black", "red", "yellow"])

            intensity_limits = [0, min(np.max(Z)*0.8, np.mean(Z)*2)]
            intensity_limits = [0, 200]
            print(np.max(Z), np.mean(Z), np.median(Z), np.std(Z))

            if 'beamshape' in criteria:
                intensity_limits = [0, np.max(Z)*0.85]
            if super_resolution:
                intensity_limits = np.array(intensity_limits)/2
            # intensity_limits2 = (max(np.min(Z), intensity_limits[0]), min(np.max(Z), intensity_limits[1]))
            intensity_limits2 = intensity_limits
            levels = np.linspace(intensity_limits2[0], intensity_limits2[1], 100)
            print(intensity_limits)
            print(intensity_limits2)
            if not contour:
                norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
                if realistic:
                    color_map = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm, shading='flat')
                else:
                    color_map = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm, *args, **kwargs)
                norm = matplotlib.colors.Normalize(vmin=intensity_limits2[0], vmax=intensity_limits2[1])
                sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map.cmap)
                sm.set_array([])
                bar = fig.colorbar(sm, ax=ax, extend='max')
            else:
                # color_map = ax.contourf(X, Y, Z, cmap=cmap, extend='neither', levels=levels, *args, **kwargs)
                if np.min(Z) < intensity_limits[0] and np.max(Z) > intensity_limits[1]:
                    color_map = ax.contourf(X, Y, Z, cmap=cmap, extend='both', levels=levels)
                elif np.min(Z) < intensity_limits[0]:
                    color_map = ax.contourf(X, Y, Z, cmap=cmap, extend='min', levels=levels)
                elif np.max(Z) > intensity_limits[1]:
                    color_map = ax.contourf(X, Y, Z, cmap=cmap, extend='max', levels=levels)
                else:
                    color_map = ax.contourf(X, Y, Z, cmap=cmap, extend='neither', levels=levels)
                # '''
                norm = matplotlib.colors.Normalize(vmin=intensity_limits2[0], vmax=intensity_limits2[1])
                sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map.cmap)
                sm.set_array([])
                bar = fig.colorbar(sm, ax=ax, extend='max', ticks=color_map.levels)
            if convert_param:
                ax.set_xlabel(r'Position Translation Stage (mm)')
                ax.set_ylabel(r'Position Diode Array (mm)')
                bar.set_label('Measured Amplitude')

            else:
                ax.set_xlabel(r'Position ($\#$Steps)')
                ax.set_ylabel(r'Position ($64 - \#$ Diode Channel)')
                bar.set_label('Measured Amplitude')

            save_name = str(criteria)+'_map'
            if not do_normalization:
                save_name += '_nonorm'
            if not convert_param:
                save_name += '_ams'
            if super_resolution:
                save_name += '_superres'
            if contour:
                save_name += '_contour'
            if realistic:
                save_name += '_realistic'
            if not avoid_sample:
                save_name += '_sampling'
            format_save(save_path=save_path, save_name=save_name+'_'+str(step)+'_')
            plt.show()
        return position, readout

