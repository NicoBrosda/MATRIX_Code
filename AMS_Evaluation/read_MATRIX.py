from AMS_Evaluation.DataAnalysis import threshold_otsu
from Plot_Methods.plot_standards import *
import matplotlib

fig, ax = plt.subplots()
ax2 = ax.twinx()


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


data = read('/Users/nico_brosda/Desktop/iphc_python_misc/matrix_27052024/e2_20000p_nA_2.csv')


def read_channels(data) -> list:
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
            signals.append(np.mean(data[col][data[col] > threshold]))
            signal_std.append(np.std(data[col][data[col] > threshold]))
            darks.append(np.mean(data[col][data[col] < threshold]))
            dark_std.append(np.std(data[col][data[col] < threshold]))
        else:
            signals.append(0)
            darks.append(0)

    '''
    # fig, ax = plt.subplots()
    ax.plot(signals)
    ax.plot(darks, c='b')
    # ax.plot(thresholds, c='k')
    # ax2.plot(signal_std, c='r', ls='--')
    # ax2.plot(dark_std, c='b', ls='--')
    # plt.show()
    # '''

    return [np.array(signals), np.array(signal_std), np.array(darks), np.array(dark_std), np.array(thresholds)]


output = read_channels(read('/Users/nico_brosda/Desktop/iphc_python_misc/matrix_27052024/e2_screwsmaller_horizontal_500p_diff_scan_nA_2.0_0_111_1_71.csv'))
print(output[0])


def normalization(paths):
    cache = []
    for i, path in enumerate(paths):
        data = read(path)
        cache.append(read_channels(data))

    max_signal = np.amax(np.array([i[0] for i in cache]), axis=0)
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
            factor.append(signal/mean)
    return factor


def lin_calib(path):
    pass


def interpret_map(folder, criteria):
    files = os.listdir(Path(folder))
    cache = []
    for file in files:
        if file[-4:] == '.csv' and criteria in file:
            cache.append(file)
    files = cache
    print(len(files))

    position = []
    cache = []
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
        # readout2[0] = readout2[0]*factor
        cache.append(readout2)

    return position, cache


paths = ['/Users/nico_brosda/Desktop/iphc_python_misc/matrix_27052024/e2_500p_bottom_nA_2.csv',
         '/Users/nico_brosda/Desktop/iphc_python_misc/matrix_27052024/e2_500p_nA_2.csv',
         '/Users/nico_brosda/Desktop/iphc_python_misc/matrix_27052024/e2_500p_top_nA_2.csv']

factor = normalization(paths)

position, readout = interpret_map('/Users/nico_brosda/Desktop/iphc_python_misc/matrix_27052024/', 'screwsmaller')

Y = np.array(position)
print(position)
X = np.arange(0, 64, 1)
Z = np.array([np.array(i[0], dtype=np.float64) for i in readout], dtype=np.float64)
print(Z[:][3])
print(Z[3])
# Z = np.array([np.append(np.array(i[0]), np.zeros(len(X)-len(Y))) for i in readout])
# Y = np.append(Y, np.zeros(len(X)-len(Y)))

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black", "red", "yellow"])
# X2, Y2, Z2 = interpolate_map_data(X, Y, Z)
# print(np.shape(X2), np.shape(Y2), np.shape(Z2))

intensity_limits = [600, 2600]
intensity_limits2 = (max(np.min(Z), intensity_limits[0]), min(np.max(Z), intensity_limits[1]))
levels = np.linspace(intensity_limits2[0], intensity_limits2[1], 100)
color_map = ax.contourf(X, Y, Z, cmap=cmap, extend='neither', levels=levels)

'''
if np.min(Z) < intensity_limits[0] and np.max(Z) > intensity_limits[1]:
    color_map = ax.contourf(X2, Y2, Z2, cmap=cmap, extend='both', levels=levels)
elif np.min(Z) < intensity_limits[0]:
    color_map = ax.contourf(X2, Y2, Z2, cmap=cmap, extend='min', levels=levels)
elif np.max(Z) > intensity_limits[1]:
    color_map = ax.contourf(X2, Y2, Z2, cmap=cmap, extend='max', levels=levels)
else:
    color_map = ax.contourf(X2, Y2, Z2, cmap=cmap, extend='neither', levels=levels)
'''

# fig.colorbar(color_map, cmap=color_map.cmap)
# '''
norm = matplotlib.colors.Normalize(vmin=intensity_limits2[0], vmax=intensity_limits2[1])
sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map.cmap)
sm.set_array([])
bar = fig.colorbar(sm, ax=ax, extend='max', ticks=color_map.levels)

plt.show()


