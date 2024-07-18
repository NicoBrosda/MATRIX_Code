from read_MATRIX import *
from copy import deepcopy

folder_path = Path('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/matrix_19062024/')
for crit in ['5s_flat_calib_', '500p_center_']:
    excluded = []
    print('-'*100)
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
        signal = read_channels(data, excluded_channel=excluded, varied_beam=False)
        signals.append(np.array(signal))

    indices = np.argsort(np.array(position)[:, 1])
    signals = np.array(signals)[indices]
    position = np.array(position)[indices]

    print(np.shape(signals))
    print(np.shape(position))

    maxima = [position[:, 1][np.argmax(signals[:, i])] for i in range(64)]
    positions = []
    for i in range(64):
        shift = np.mean(maxima) - maxima[i]
        cache = deepcopy(position)
        cache[:, 1] = cache[:, 1] + (32 - i) * (0.5 + 0.08)
        # ax.plot(position[:, 1]+shift, signals[:, i], c=c)
        positions.append(cache)

    '''
    fig, ax = plt.subplots()
    for i, pos in enumerate(positions):
        ax.plot(pos[:, 0] + i, pos[:, 1], ls='', marker='x')
    plt.show()
    '''

    fig, ax = plt.subplots()
    # This defines a colour range for measurement temperatures between 5 and 40 K
    diode_colourmap = sns.color_palette("coolwarm", as_cmap=True)
    diode_colourmapper = lambda x: color_mapper(x, 0, 63)
    diode_colour = lambda x: diode_colourmap(diode_colourmapper(x))

    maxima = [position[:, 1][np.argmax(signals[:, i])] for i in range(64)]
    for signal in signals:
        signal[60] /= 60

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
    fig, ax = plt.subplots()

    # This defines a colour range for measurement temperatures between 5 and 40 K
    pos_colourmap = sns.color_palette("crest_r", as_cmap=True)
    pos_colourmapper = lambda x: color_mapper(x, min(position[:, 1]), max(position[:, 1]))
    pos_colour = lambda x: pos_colourmap(pos_colourmapper(x))

    for i, signal in enumerate(signals):
        # Data plotting
        y_posi = position[i, 1]
        c = pos_colour(y_posi)
        signal[60] /= 60
        ax.plot(signal, ls='-', c=c)

    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    gradient_arrow(ax, transform_axis_to_data_coordinates(ax, [0.1, 0.95]),
                   transform_axis_to_data_coordinates(ax, [0.4, 0.95]), cmap=pos_colourmap, lw=5)
    ax.text(*transform_axis_to_data_coordinates(ax, [0.05, 0.88]), r'y='+str(min(position[:, 1])), fontsize=15,
            c=pos_colour(min(position[:, 1])))  # , bbox={'facecolor': freq_colour(1033), 'alpha': 0.2, 'pad': 2})
    ax.text(*transform_axis_to_data_coordinates(ax, [0.3, 0.88]), r'y='+str(max(position[:, 1])), fontsize=15,
            c=pos_colour(max(position[:, 1])))  # , bbox={'facecolor': freq_colour(32033), 'alpha': 0.2, 'pad': 2})

    ax.set_xlabel(r'Channel ($\#$ of Diode)')
    ax.set_ylabel('Measured amplitude')
    format_save(Path('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_19062024/FlatCalib/'), crit+'_Channels',
                legend=False)
    # '''

    '''
    fig, ax = plt.subplots()

    # This defines a colour range for measurement temperatures between 5 and 40 K
    diode_colourmap = sns.color_palette("coolwarm", as_cmap=True)
    diode_colourmapper = lambda x: color_mapper(x, 0, 63)
    diode_colour = lambda x: diode_colourmap(diode_colourmapper(x))

    maxima = [position[:, 1][np.argmax(signals[:, i])] for i in range(64)]

    for i in range(64):
        c = diode_colour(i)
        shift = np.mean(maxima) - maxima[i]
        ax.plot(position[:, 1]+(32-i)*(0.5+0.08), signals[:, i], c=c, ls='--')
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

    format_save(Path('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_19062024/FlatCalib/'), crit,
                legend=False)
    '''
