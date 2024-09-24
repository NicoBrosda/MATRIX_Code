from AMS_Evaluation.read_MATRIX import *

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
    format_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_19062024/Bragg/', '_threefullresolved2_', legend=False)
    #'''

    # Plot with each two values in signal array exchanged
    fig, ax = plt.subplots()
    signals = signals * factor
    signals_swapped = deepcopy(signals)
    print(np.shape(signals_swapped))
    signals_swapped[:, 1:-1:2] = signals[:, 2::2]
    signals_swapped[:, 2::2] = signals[:, 1:-2:2]

    for i, pos in enumerate(position):
        if pos[1] > 86:
            c = sns.color_palette("tab10")[i]
            ax.plot(positions[:, i, 1], signals_swapped[i], label='Swapped channels', marker='x', ls='-',
                    color=sns.color_palette("tab10")[0])
            ax.plot(positions[:, i, 1], signals[i], label='original signal', marker='+', ls='', alpha=1,
                    color=sns.color_palette("tab10")[1], zorder=-1)

    ax.set_xlabel(r'Calculated real position of diodes (mm)')
    ax.set_ylabel(r'Signal Amplitude')
    ax.set_ylim(-100, ax.get_ylim()[1])
    legend = ax.legend()
    legend.set_title('Position of y-stage')
    format_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_19062024/Bragg/', '_exampleswapped_',
                legend=False)


# Plot with each two values in signal array exchanged
    fig, ax = plt.subplots()

    for i, pos in enumerate(position):
        if pos[1] > 80:
            c = sns.color_palette("tab10")[i]
            ax.plot(positions[:, i, 1], signals_swapped[i], label=str(pos[1]) + r'\,' + 'mm  swapped channels',
                    marker='x', ls='-', color=c)

    ax.set_xlabel(r'Calculated real position of diodes (mm)')
    ax.set_ylabel(r'Signal Amplitude')
    ax.set_ylim(-100, ax.get_ylim()[1])
    legend = ax.legend()
    legend.set_title('Position of y-stage')
    format_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_19062024/Bragg/', '_swappedvalues_',
                legend=False)