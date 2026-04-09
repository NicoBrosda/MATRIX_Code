from datetime import datetime

import numpy as np

from EvaluationSoftware.main import *
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

mapping = Path('../../Files/mapping.xlsx')
data = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:])-1 for k in data['direction_2']]
readout, position_parser, voltage_parser, current_parser = (
    lambda x, y: ams_2line_fast_avg(x, y, channel_assignment=channel_assignment),
    standard_position,
    standard_voltage,
    current5
)

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_171225/')
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/ResultsPEEK_161225/')
save_format = '.png'

A = Analyzer((2, 64), (0.4, 0.4), (0.1, 0.1), readout=readout,
                 diode_offset=[[0, - 0.25], np.zeros(64)], position_parser=position_parser,
                 voltage_parser=voltage_parser, current_parser=current_parser)
dark_path = folder_path
dark = ['dark_current.csv']
# A.set_dark_measurement(dark_path, dark)
norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_260325/')
norm_array1 = ['exp7_norm1,9V_']
norm = norm_array1
norm_func = lambda list_of_files, instance, method='least_squares': normalization_from_translated_array_v3(
    list_of_files, instance, method, align_lines=True)
A.normalization(norm_path, norm, normalization_module=norm_func)
A.scale = 'nano'

measurements = array_txt_file_search(os.listdir(folder_path), searchlist=['exp2_', 'exp4_'], file_suffix='.csv', txt_file=False, blacklist=['.png'])

dark0 = array_txt_file_search(os.listdir(folder_path), searchlist=['dark_current.csv'], file_suffix='.csv', txt_file=False, blacklist=['.png'])
dark_end = array_txt_file_search(os.listdir(folder_path), searchlist=['exp10_cyrce_irr__nA_1.0_x_120.0_y_15.8'], file_suffix='.csv', txt_file=False, blacklist=['.png'])

def time_parser(input_string, parser=None):
    time_string = input_string[input_string.rindex('_')+1:input_string.rindex('.csv')]
    if parser is None:
        return int(time_string)
    else:
        return parser(int(time_string))

measurements = sorted(
    measurements,
    key=lambda s: int(s.rsplit("_", 1)[1].replace(".csv", ""))
)
time_0 = time_parser(measurements[0], parser=datetime.fromtimestamp)
times = np.array([(time_parser(i, parser=datetime.fromtimestamp)-time_0).total_seconds() / 60 for i in measurements])

try:
    cache = np.load(results_path / 'long_term_cache.npy')
except FileNotFoundError:
    cache = []
    for measurement in tqdm(measurements):
        cache.append(A.readout(folder_path / measurement, A)['signal'])
    cache = np.array(cache)
    np.save(results_path / 'long_term_cache.npy', cache)

signal_level = 0.05

cache = np.loadtxt(results_path / 'long_term_data.txt', delimiter=',')
times = np.loadtxt(results_path / 'long_term_times.txt', delimiter=',') * 60

try:
    dark_cache = np.load(results_path / 'long_term_dark_cache.npy')
    dark_times = np.load(results_path / 'dark_times.npy')
except FileNotFoundError:
    dark_cache = []
    dark_times = []
    for i, measurement in tqdm(enumerate(measurements)):
        signal = A.readout(folder_path / measurement, A)['signal']
        if A.signal_conversion(np.mean(signal)) < signal_level:
            dark_cache.append(signal)
            dark_times.append(times[i])
    dark_cache = np.array(dark_cache)
    dark_times = np.array(dark_times)

    dark0 = A.readout(folder_path / dark0[0], A)['signal']
    print(np.shape(dark0))
    dark_cache = np.concatenate(([dark0], dark_cache), axis=0)
    dark_times = np.concatenate((np.array([0]), dark_times))

    dark_end = A.readout(folder_path / dark_end[0], A)['signal']
    dark_cache = np.concatenate((dark_cache, [dark_end]), axis=0)
    dark_times = np.append(dark_times, [times[-1]], axis=0)

    np.save(results_path / 'long_term_dark_cache.npy', dark_cache)
    np.save(results_path / 'dark_times.npy', dark_times)

cache2 = np.empty((dark_cache.shape[0], 128), dtype=dark_cache.dtype)
cache2[:, 0::2] = dark_cache[:, 1, :]
cache2[:, 1::2] = dark_cache[:, 0, :]
dark_cache = cache2

dark_cache = A.signal_conversion(dark_cache)

np.savetxt(results_path / 'long_term_dark_data.txt', dark_cache, delimiter=',', fmt='%.6f')
np.savetxt(results_path / 'long_term_dark_times.txt', dark_times, delimiter=',', fmt='%.3f')

fig, ax = plt.subplots()
ax.plot(times/60, cache[:, 0], marker='x', ls='', color='r')
for dt in dark_times:
    ax.axvline(x=dt/60, ls='--', color='k')
ax.set_xlabel('Irradiation Time (min)')
ax.set_ylabel(f'Measurement signal ({scale_dict[A.scale][-1]}A)')
format_save(results_path / 'DarkCurrentsAnalyse/', "SelectionCheck", save_format=save_format, legend=False)

param_list = dark_times
param_cmap = sns.color_palette("crest_r", as_cmap=True)
param_colormapper = lambda param: color_mapper(param, np.min(param_list), np.max(param_list))
param_color = lambda param: param_cmap(param_colormapper(param))
fig, ax = plt.subplots()
ax2 = ax.twinx()
for i, ds in enumerate(dark_cache):
    if i == 0:
        ax.plot(dark_times[i] / 60, np.mean(ds), marker='x', ls='', color=param_color(dark_times[i]), label='Array mean dark current')
        ax2.plot(dark_times[i] / 60, np.std(ds), marker='o', ls='', color=param_color(dark_times[i]), label='Array std dark current')
    ax.plot(dark_times[i]/60, np.mean(ds), marker='x', ls='', color=param_color(dark_times[i]))
    ax2.plot(dark_times[i]/60, np.std(ds), marker='o', ls='', color=param_color(dark_times[i]))

ax.set_xlabel('Irradiation Time (min)')
ax.set_ylabel(f'Array mean of dark current ({scale_dict[A.scale][-1]}A)')
ax2.set_ylabel(f'Array Std of dark current ({scale_dict[A.scale][-1]}A)')

format_save(results_path / 'DarkCurrentsAnalyse/', "DarkCurrentArrayDevelopment", save_format=save_format, legend=True)