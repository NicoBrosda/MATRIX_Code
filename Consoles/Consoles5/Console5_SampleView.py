import os

import matplotlib.pyplot as plt
import numpy as np

from EvaluationSoftware.main import *

result_path = Path('/Users/nico_brosda/Cyrce_Messungen/Results_221024/')
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_230924/')
# folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_19062024/')


dark = ['voltage_scan_no_beam_nA_1.9000000000000006_x_20.0_y_70.0.csv']
signal = ['voltage_scan_beam_2nA_nA_1.9000000000000006_x_20.0_y_70.0.csv']
signal = ['d2_1n_3s_beam_all_without_diffuser_mesures_nA_2.0_x_21.5_y_91.0.csv']
# signal = ['d2_1n_5s_flat_mesures_nA_2.0_x_20.0_y_85.0.csv']

signal = ['uniformity_scan_y_200um_2nA__nA_1.8_x_20.0_y_64.5.csv']
# signal = ['BraggPeak_200um_2nA_nA_1.8_x_22.0_y_70.0.csv']
signal = ['BeamScan_0um_2nA_nA_1.8_x_22.5_y_55.0.csv']

files = array_txt_file_search(os.listdir(folder_path), blacklist=['.png'], searchlist=signal, txt_file=False, file_suffix='.csv')

print(len(files))

for file in files[0:1]:
    print(file)
    path_to_data_file = folder_path / file
    el = 128
    columns_used = el
    if columns_used > 128:
        columns_used = 128
    detect = len(pd.read_csv(path_to_data_file, delimiter=',', nrows=1, header=None).columns)
    data1 = pd.read_csv(path_to_data_file, delimiter=',', nrows=1, header=None, usecols=range(130, detect))
    data2 = pd.read_csv(path_to_data_file, delimiter=',', usecols=range(columns_used+2))
    if len(data2.columns) - len(data1.columns) > 0:
        for i in range(len(data2.columns) - len(data1.columns)):
            data1['Add' + str(i)] = np.NaN
    data1 = data1.set_axis(list(data2.columns), axis=1)
    data = pd.concat([data1, data2])

    fig, ax = plt.subplots()
    diode_cmap = sns.color_palette("coolwarm", as_cmap=True)
    diode_colormapper = lambda diode: color_mapper(diode, 0, 128)
    diode_color = lambda diode: diode_cmap(diode_colormapper(diode))

    cache = []
    for i, col in enumerate(data):
        if i < 2:
            continue
        cache.append(np.mean(data[col]))

    cache = np.array(cache)
    ax.plot(data['Ch'+str(np.argmax(cache)+2)])
    ax.plot(data['Ch'+str(np.argmin(cache)+2)])


    ax.set_xlabel('Sample = Time (ms)')
    ax.set_ylabel('Signal (a.u.)')
    ax.set_title(r'Without diffuser')
    ax.set_xlim(0, 500)
    # gradient_arrow(ax, transform_axis_to_data_coordinates(ax, [0.11, 0.92]), transform_axis_to_data_coordinates(ax, [0.11, 0.79]), cmap=diode_cmap, lw=10, zorder=5)
    # ax.text(*transform_axis_to_data_coordinates(ax, [0.035, 0.94]), r'Diode $\#$1', fontsize=13, c=diode_color(0), zorder=3)  # , bbox={'facecolor': freq_colour(1033), 'alpha': 0.2, 'pad': 2})
    # ax.text(*transform_axis_to_data_coordinates(ax, [0.02, 0.71]), r'Diode $\#$' + str(128), fontsize=13, c=diode_color(128), zorder=3)  # , bbox={'facecolor': freq_colour(32033), 'alpha': 0.2, 'pad': 2})
    format_save(result_path / '50Hz/', 'withoutdiff', legend=False)