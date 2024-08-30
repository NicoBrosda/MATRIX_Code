import numpy as np
import scipy.signal

from read_MATRIX import *


class LineShape:
    def __init__(self, list_of_points, distance_mode=False):
        print('A shape with connection lines between the given points will be initiated! Note that for positioning of '
              'the shape the x-positions can be scaled by setting a reference point later on.')
        self.points = np.array(list_of_points)
        self.distance_mode = distance_mode
        if distance_mode:
            cache = deepcopy(self.points)
            for i, point in enumerate(self.points):
                cache[i, 0] += np.sum(self.points[:i, 0])
            self.points = cache
        else:
            self.sort()

    def sort(self):
        self.points = self.points[np.argsort(self.points[:, 0])]

    def print_shape(self):
        print('The points of the current shape will be converted to a printout')
        for i, point in enumerate(self.points):
            print('Point', i, ':', point)

    def mirror(self):
        cache = []
        distance = 0
        for j in range(np.shape(self.points)[0]):
            cache.append([self.points[0, 0] + distance, self.points[::-1][j, 1]])
            if j < np.shape(self.points)[0]-1:
                distance += self.points[::-1][j, 0] - self.points[::-1][j+1, 0]
        self.points = np.array(cache)

    def add_points(self, list_of_points):
        self.points = np.concatenate((self.points, np.array(list_of_points)), axis=0)
        self.sort()

    def position(self, x_middle=None, reference_x=None, y_diff=0):
        if x_middle is not None:
            if reference_x is None:
                np.mean(self.points[:, 1])
            self.points[:, 0] += (x_middle - reference_x)
        self.points[:, 1] += y_diff

    def add_to_plot(self, y_min=0.2, y_max=0.8, ax=None, *args, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.set_xlim(min(min(self.points[:, 0]), ax.get_xlim()[0]), max(max(self.points[:, 0]), ax.get_xlim()[1]))
        ax.set_ylim(ax.get_ylim())

        y = self.points[:, 1]
        y -= min(y)
        y = y * (y_max-y_min) / max(y)
        y += y_min
        y = [transform_axis_to_data_coordinates(ax, [0.5, j])[1] for j in y]
        y_min = transform_axis_to_data_coordinates(ax, [0.5, y_min])[1]
        fill = ax.fill_between(self.points[:, 0], y, y_min, *args, **kwargs)
        return fill

    def calculate_value(self, x_coordinate):
        def line_segment(x, x1, y1, x2, y2):
            return ((x * (y2-y1)) / (x2-x1)) - ((x1 * (y2-y1)) / (x2 - x1)) + y1
        value = 0
        if self.points[0, 0] < x_coordinate < self.points[-1, 0]:
            distance = x_coordinate - self.points[:, 0]
            index = np.argmin(distance[distance >= 0])
            value = line_segment(x_coordinate, *self.points[index], *self.points[index+1])
        elif self.points[0, 0] >= x_coordinate:
            value = self.points[0, 1]
        else:
            value = self.points[-1, 1]
        return value

    def get_plot_value(self, value, y_min=0.2, y_max=0.8):
        y = self.points[:, 1]
        if value < min(y):
            value = min(y)
        elif value > max(y):
            value = max(y)
        y -= min(y)
        value = (value-min(y)) * (y_max - y_min) / max(y) + y_min
        return transform_axis_to_data_coordinates(ax, [0.5, value])[1]


folder_path = Path('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/matrix_19062024/')
dark_path = folder_path / 'd2_1n_3s_beam_all_without_diffuser_dark.csv'
# Normalization factor - 5s_flat_calib is from the measurements later that day, where the beam was seemingly more stable
for crit in ['5s_flat_calib_', '500p_center_'][0:1]:
    excluded = []
    print('-' * 100)
    print(crit)
    files_norm = os.listdir(folder_path)
    files_norm = array_txt_file_search(files_norm, blacklist=['.png'], searchlist=[crit],
                                  file_suffix='.csv', txt_file=False)
    start = time.time()
    factor = normalization_new(folder_path, files_norm, excluded, 'y', 'least_squares', dark_path, cache_save=True,
                               factor_limits=[0, 3], correction=-0.065)
    end = time.time()
    print(end - start)

for crit in ['trapeze_bragg_0_10s_']:
    excluded = []
    print('-'*100)
    print(crit)
    files = os.listdir(folder_path)
    files = array_txt_file_search(files, blacklist=['.png'], searchlist=[crit],
                                  file_suffix='.csv', txt_file=False)

    print(len(files))

    position = []
    signals = []
    for file in files:
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

        data = read(folder_path / file)
        signal = read_channels(data, excluded_channel=excluded, varied_beam=False, path_dark=dark_path)
        signals.append(np.array(signal))

    indices = np.argsort(np.array(position)[:, 1])
    signals = np.array(signals)[indices]
    position = np.array(position)[indices]

    positions = []
    for i in range(64):
        cache = deepcopy(position)
        cache[:, 1] = cache[:, 1] + (32 - i) * (0.5 + 0.08 - 0.065)
        # ax.plot(position[:, 1]+shift, signals[:, i], c=c)
        positions.append(cache)

    positions = np.array(positions)
    print(position)
    print(np.shape(positions))

    '''
    fig, ax = plt.subplots()
    markers = ['x', 'P', '*', 's', 'd', 'o', '^', 'v']

    for i, pos in enumerate(position):
        c = sns.color_palette("tab10")[i]
        ax.plot(positions[:, i, 1], signals[i]*factor, label=str(pos[1])+r'\,'+'mm', marker=markers[i], ls='', color=c)
        ax.axvline(positions[int(np.shape(positions)[0]/2), i, 1], color=c)

    ax.set_xlabel(r'Calculated real position of diodes (mm)')
    ax.set_ylabel(r'Signal Amplitude')
    ax.set_ylim(-100, ax.get_ylim()[1])
    legend = ax.legend()
    legend.set_title('Position of y-stage')
    format_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_19062024/Bragg/', '_all_', legend=False)
    # '''

    '''
    fig, ax = plt.subplots()
    markers = ['x', 'P', '*', 's', 'd', 'o', '^', 'v']

    for i, pos in enumerate(position):
        if pos[1] > 80:
            c = sns.color_palette("tab10")[i]
            ax.plot(positions[:, i, 1], signals[i] * factor, label=str(pos[1]) + r'\,' + 'mm', marker='x', ls='-',
                    color=c)
            # ax.axvline(positions[int(np.shape(positions)[0] / 2), i, 1], color=c)

    ax.set_xlabel(r'Calculated real position of diodes (mm)')
    ax.set_ylabel(r'Signal Amplitude')
    ax.set_ylim(-100, ax.get_ylim()[1])
    legend = ax.legend()
    legend.set_title('Position of y-stage')
    plt.show()
    # format_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_19062024/Bragg/', '_threefullresolved2_', legend=False)
    #'''

    # '''
    # Plot with each two values in signal array exchanged
    signals = signals * factor
    signals_swapped = deepcopy(signals)
    print(np.shape(signals_swapped))
    signals_swapped[:, 1:-1:2] = signals[:, 2::2]
    signals_swapped[:, 2::2] = signals[:, 1:-2:2]

    # Plot with each two values in signal array exchanged
    fig, ax = plt.subplots()

    for i, pos in enumerate(position):
        if pos[1] > 80:
            c = sns.color_palette("tab10")[i]
            ax.plot(positions[:, i, 1], signals_swapped[i], label=str(pos[1]) + r'\,' + 'mm  swapped channels',
                    marker='x', ls='-', color=c)

    print('x'*50)
    # Calc the mean of the signals
    distance = 0.3
    groups = np.array(group(input_list=positions[:, 5:, 1], group_range=distance))
    groups.sort()
    print(groups)
    sig_mean = np.zeros_like(groups)
    for i, gr in enumerate(groups):
        cache = []
        for j in range(np.shape(positions)[1]):
            if j >= 5:
                # closest value:
                close = np.argmin(np.abs(positions[:, j, 1] - gr))
                if np.abs(positions[close, j, 1] - gr) <= distance and signals_swapped[j, close] > 0:
                    cache.append(signals_swapped[j, close])
        if len(cache) > 0:
            sig_mean[i] = np.mean(cache)

    ax.plot(groups, sig_mean, label='mean signal', marker='+', ls='-', color='k')
    ax.set_xlabel(r'Calculated real position of diodes (mm)')
    ax.set_ylabel(r'Signal Amplitude')
    ax.set_ylim(-100, ax.get_ylim()[1])
    legend = ax.legend()
    legend.set_title('Position of y-stage')
    # plt.show()
    # format_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_19062024/Bragg/', '_swappedvalues_', legend=False)
    # '''
    print('The distance between two pixels is ', np.mean([np.mean([positions[k-1, i, 1] - positions[k, i, 1] for k, elm in enumerate(positions[:, i, 1]) if k > 0]) for i in range(np.shape(positions)[1])]))

    # Test for the geometry plot in matplotlib

    # Parameter to define the middle of the geometry (in data coordinates)
    middle = 92.711
    print('Full material from ', middle-0.966/2-20.503-6.702, ' mm until ', middle-0.966/2-20.503)
    print('Long ascend from ', middle-0.966/2-20.503, ' mm until ', middle-0.966/2)
    print('Free space from ', middle-0.966/2, ' mm until ', middle+0.966/2)
    print('Steep ascend from ', middle+0.966/2, ' mm until ', middle+0.966/2+4.877)
    print('Full material from ', middle+0.966/2+4.877, ' mm until ', middle+0.966/2+4.877+6.951)
    ax.set_xlim(middle-0.966/2-20.503-6.702-2, middle+0.966/2+4.877+6.951+2)

    shape = LineShape([[0, 10], [6.702, 10], [20.503, 0], [0.966, 0], [4.877, 10], [6.951, 10]], distance_mode=True)
    shape.print_shape()
    shape.position(92.711, 0.966 / 2 + 20.503 + 6.702)
    shape.add_to_plot(0.0, 0.5, color='grey', alpha=0.7, zorder=-1, edgecolor='k')
    format_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_19062024/Bragg/', '_trapezandmean_',
                legend=False)

    fig, ax = plt.subplots()
    ax.plot(groups, sig_mean, label='mean signal', marker='+', ls='-', color='k')
    # Find peaks in the signal and select the 2 highest
    peaks = scipy.signal.find_peaks(sig_mean, prominence=0)
    peaks = peaks[0]
    sorting = np.argsort(sig_mean[peaks])[-2:]
    print(sig_mean[peaks][np.argsort(sig_mean[peaks])])
    print(groups[peaks][np.argsort(sig_mean[peaks])])
    print(sorting)
    for i, gr in enumerate(groups[peaks][sorting]):
        c = sns.color_palette("tab10")[i]
        ax.axvline(gr, c='r')
        print('Peak at position', gr, 'with material thickness', shape.calculate_value(gr))
        print(shape.get_plot_value(shape.calculate_value(gr), 0, 0.5))
        ax.axhline(shape.get_plot_value(shape.calculate_value(gr), 0, 0.5), c=c, ls='-', alpha=1, lw=1)
    shape.add_to_plot(0.0, 0.5, color='grey', alpha=0.7, zorder=-1, edgecolor='k')
    ax.set_xlabel(r'Calculated real position of diodes (mm)')
    ax.set_ylabel(r'Signal Amplitude')
    ax.set_ylim(-100, ax.get_ylim()[1])
    format_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_19062024/Bragg/', '_peak_',
                legend=False)