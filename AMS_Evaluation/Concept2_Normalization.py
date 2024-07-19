import numpy as np
import pandas as pd

from read_MATRIX import *
from copy import deepcopy
from tqdm import tqdm

folder_path = Path('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/matrix_19062024/')
dark_path = folder_path / 'd2_1n_3s_beam_all_without_diffuser_dark.csv'
for crit in ['5s_flat_calib_', '500p_center_']:
    excluded = []
    print('-' * 100)
    print(crit)
    files = os.listdir(folder_path)
    files = array_txt_file_search(files, blacklist=['.png'], searchlist=[crit],
                                  file_suffix='.csv', txt_file=False)

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

    print(np.shape(signals))
    print(np.shape(position))

    for signal in signals:
        signal[60] = 0

    maxima = [position[:, 1][np.argmax(signals[:, i])] for i in range(64)]
    positions = []
    for i in range(64):
        shift = np.mean(maxima) - maxima[i]
        cache = deepcopy(position)
        cache[:, 1] = cache[:, 1] + (32 - i) * (0.5 + 0.08)
        # ax.plot(position[:, 1]+shift, signals[:, i], c=c)
        positions.append(cache)


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
        this_group = []
        for j in tqdm(input_list):
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


    print(len(np.array(positions)[:, :, 1].flatten()))
    groups = group(np.array(positions)[:, :, 1].flatten(), 0.2)
    print(len(groups))
    '''
    fig, ax = plt.subplots()
    for i, pos in enumerate(positions):
        ax.plot(pos[:, 0] + i, pos[:, 1], ls='', marker='x')

    for group in groups:
        ax.axhline(np.mean(group), ls='--', zorder=-1)
    plt.show()
    # '''
    print(np.shape(signals))
    print(np.shape(positions))
    print(np.shape(np.array(positions)[:, :, 1].T))
    for mean in groups:
        pass
    '''
        fig, ax = plt.subplots()
        # This defines a colour range for measurement temperatures between 5 and 40 K
        diode_colourmap = sns.color_palette("coolwarm", as_cmap=True)
        diode_colourmapper = lambda x: color_mapper(x, 0, 63)
        diode_colour = lambda x: diode_colourmap(diode_colourmapper(x))

        maxima = [position[:, 1][np.argmax(signals[:, i])] for i in range(64)]

        for i in range(64):
            c = diode_colour(i)
            # shift = np.mean(maxima) - maxima[i]
            ax.plot(position[:, 1] + (32 - i) * (0.5 + 0.08 - 0), signals[:, i], c=c, ls='--')
            # ax.plot(position[:, 1]+shift, signals[:, i], c=c)

        ax.set_xlabel('Y-Position of Diode - Shifted for uniform center (mm)')
        ax.set_ylabel('Measured Amplitude')
        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())
        gradient_arrow(ax, transform_axis_to_data_coordinates(ax, [0.89, 0.91]),
                       transform_axis_to_data_coordinates(ax, [0.89, 0.61]), cmap=diode_colourmap, lw=5)
        ax.text(*transform_axis_to_data_coordinates(ax, [0.78, 0.94]), r'Diode $\#$1', fontsize=15,
                c=diode_colour(0))  # , bbox={'facecolor': freq_colour(1033), 'alpha': 0.2, 'pad': 2})
        ax.text(*transform_axis_to_data_coordinates(ax, [0.78, 0.55]), r'Diode $\#$64', fontsize=15,
                c=diode_colour(63))  # , bbox={'facecolor': freq_colour(32033), 'alpha': 0.2, 'pad': 2})
        plt.show()
        '''


