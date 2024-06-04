import numpy as np
import pandas as pd

from AMS_Evaluation.DataAnalysis import threshold_otsu
from Plot_Methods.plot_standards import *
import matplotlib
from mpl_toolkits.mplot3d import axes3d


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
def read_channels(data, excluded_channel=[], advanced_output=False) -> list or np.ndarray:
    """
    Function to convert the data of each channel into values representing the signal response (as on - off amplitude)
    - standard - or return of separate arrays: signal, signal_std, dark, dark_std, thresholds - advanced_output = True
    The signal and dark amplitudes of each diode are calculated with the threshold_otsu algorithm, designed to find the
    separating value of 2 Gaussian populations.
    :param data: input data, pd.DataFrame or np.NdArray in required DataStructure (channels = rows, rows = samples)
    :param excluded_channel: array with indices of excluded channels, that are added as signal=0 for further processing
    :param advanced_output: advanced return
    :return:
    """
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


def lin_test(path):
    pass


def interpret_map(folder, criteria, save_path='', plot=False, paths_of_norm_files=None, excluded_channel=[], do_normalization=True,
                  convert_param=True, diode_size=0.5, diode_space=0.0, x_stepwidth=0.25,
                  Z_conversion=lambda Z: Z, *args, **kwargs):
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
        readout2 = read_channels(data2, excluded_channel)
        # Normalization
        if do_normalization:
            if paths_of_norm_files is not None:
                factor = normalization(paths_of_norm_files, excluded_channel=excluded_channel)
                readout2 = readout2*factor
        readout.append(readout2)

    print(position, readout)
    position, readout = np.array(position), np.array(readout)
    sorting = np.argsort(position)
    readout = readout[sorting]
    position = position[sorting]
    if not plot:
        return position, readout
    else:
        fig, ax = plt.subplots()
        print(np.size(readout))
        print(readout)
        channels = np.arange(0, np.size(readout[1]), 1)
        X, Y, Z = position, channels, readout.T
        Z = Z[:][::-1]
        if convert_param:
            X = x_stepwidth * X  # Defining X as translated position in mm
            # Defining the Y conversion based on the geometry of the diodes
            Y = Y*(diode_size+diode_space)
            # Defining the conversion of amplitude z into a current pA
            Z = Z_conversion(Z)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black", "red", "yellow"])

        if np.max(Z) > 8.5 * np.mean(Z):
            intensity_limits = [0, 1500]
        else:
            intensity_limits = [0, np.max(Z)*0.85]
        intensity_limits = [0, 1600]
        if 'beamshape' in criteria:
            intensity_limits = [0, np.max(Z)*0.85]
        intensity_limits2 = (max(np.min(Z), intensity_limits[0]), min(np.max(Z), intensity_limits[1]))
        levels = np.linspace(intensity_limits2[0], intensity_limits2[1], 100)
        # color_map = ax.contourf(X, Y, Z, cmap=cmap, extend='neither', levels=levels, *args, **kwargs)
        if np.min(Z) < intensity_limits[0] and np.max(Z) > intensity_limits[1]:
            color_map = ax.contourf(X, Y, Z, cmap=cmap, extend='both', levels=levels, *args, **kwargs)
        elif np.min(Z) < intensity_limits[0]:
            color_map = ax.contourf(X, Y, Z, cmap=cmap, extend='min', levels=levels, *args, **kwargs)
        elif np.max(Z) > intensity_limits[1]:
            color_map = ax.contourf(X, Y, Z, cmap=cmap, extend='max', levels=levels, *args, **kwargs)
        else:
            color_map = ax.contourf(X, Y, Z, cmap=cmap, extend='neither', levels=levels, *args, **kwargs)

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
            save_name += '_nonormalization'
        if convert_param:
            save_name += '_real_param'
        format_save(save_path=save_path, save_name=save_name)
        plt.show()
        return position, readout


