import numpy as np

from AMS_Evaluation.DataAnalysis import threshold_otsu
from Plot_Methods.plot_standards import *
import matplotlib
from mpl_toolkits.mplot3d import axes3d

save_path = '/Users/nico_brosda/Desktop/iphc_python_misc/Results/maps/'


def read(file_path, fast=False):
    if fast:
        data = pd.read_csv(file_path, delimiter=',', usecols=range(66))
    else:
        data1 = pd.read_csv(file_path, delimiter=',', nrows=1, header=None, usecols=range(130, 195))
        data2 = pd.read_csv(file_path, delimiter=',', usecols=range(66))
        if len(data2.columns) - len(data1.columns) > 0:
            for i in range(len(data2.columns) - len(data1.columns)):
                data1['Add'+str(i)] = np.NaN
        data1 = data1.set_axis(list(data2.columns), axis=1)
        data = pd.concat([data1, data2])
    return data


def read_channels(data, excluded_channel=[], advanced_output=False) -> list or np.ndarray:
    thresholds = []
    signals = []
    signal_std = []
    darks = []
    dark_std = []

    for i, col in enumerate(data):
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
        return [np.array(signals), np.array(signal_std), np.array(darks), np.array(dark_std), np.array(thresholds)]
    else:
        return np.array(signals)


def normalization(paths_of_norm_files):
    cache = []
    for i, path in enumerate(paths_of_norm_files):
        data = read(path)
        cache.append(read_channels(data))

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
    return factor


def lin_test(path):
    pass


def interpret_map(folder, criteria, plot=False, paths_of_norm_files=None):
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
        readout2 = read_channels(data2)
        # Normalization
        if paths_of_norm_files is not None:
            factor = normalization(paths_of_norm_files)
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
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black", "red", "yellow"])

        if np.max(Z) > 10 * np.mean(Z):
            intensity_limits = [0, 1500]
        else:
            intensity_limits = [0, np.max(Z)*0.85]
        intensity_limits2 = (max(np.min(Z), intensity_limits[0]), min(np.max(Z), intensity_limits[1]))
        levels = np.linspace(intensity_limits2[0], intensity_limits2[1], 100)
        color_map = ax.contourf(X, Y, Z, cmap=cmap, extend='neither', levels=levels)
        if np.min(Z) < intensity_limits[0] and np.max(Z) > intensity_limits[1]:
            color_map = ax.contourf(X, Y, Z, cmap=cmap, extend='both', levels=levels)
        elif np.min(Z) < intensity_limits[0]:
            color_map = ax.contourf(X, Y, Z, cmap=cmap, extend='min', levels=levels)
        elif np.max(Z) > intensity_limits[1]:
            color_map = ax.contourf(X, Y, Z, cmap=cmap, extend='max', levels=levels)
        else:
            color_map = ax.contourf(X, Y, Z, cmap=cmap, extend='neither', levels=levels)

        norm = matplotlib.colors.Normalize(vmin=intensity_limits2[0], vmax=intensity_limits2[1])
        sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map.cmap)
        sm.set_array([])
        bar = fig.colorbar(sm, ax=ax, extend='max', ticks=color_map.levels)
        ax.set_xlabel(r'Position ($\#$Steps)')
        ax.set_ylabel(r'Position ($64 - \#$ Diode Channel)')
        bar.set_label('Measured Amplitude')
        format_save(save_path=save_path, save_name=criteria+'nohomogenisation_map')
        plt.show()
        return position, readout


paths = ['/Users/nico_brosda/Desktop/iphc_python_misc/matrix_27052024/e2_500p_bottom_nA_2.csv',
         '/Users/nico_brosda/Desktop/iphc_python_misc/matrix_27052024/e2_500p_nA_2.csv',
         '/Users/nico_brosda/Desktop/iphc_python_misc/matrix_27052024/e2_500p_top_nA_2.csv']


# for crit in ['screwsmaller_horizontal', 'noscrew', '_screw_', 'screw8_vertical', 'screw8_horizontal_', 'screw8_horizontal2_', 'beamshape_', 'beamshape2_']:
for crit in ['noscrew']:
    out = interpret_map('/Users/nico_brosda/Desktop/iphc_python_misc/matrix_27052024/', crit, plot=True, paths_of_norm_files=None)
