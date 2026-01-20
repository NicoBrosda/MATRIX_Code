from copy import deepcopy

from EvaluationSoftware.main import *
from EvaluationSoftware.readout_modules import ams_channel_assignment_readout
from EvaluationSoftware.normalization_modules import normalization_from_translated_array

mapping = Path('../../Files/mapping.xlsx')
data = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:])-1 for k in data['direction_2']]
readout, position_parser, voltage_parser, current_parser = (
    lambda x, y: ams_2line_readout(x, y, channel_assignment=channel_assignment),
    standard_position,
    standard_voltage,
    current5
)

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_peek_121125/')
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/ResultsPEEK_121125/Maps/')

new_measurements = []
# '''
new_measurements += [f'calibration_1.9_24MeV_P0{int(i+1)}_' for i in range(9)]
new_measurements += [f'calibration_1.9_24MeV_P{int(i+1)}_' for i in range(9, 13)]

dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_peek_121125/')

dark_paths_array1 = ['/Users/nico_brosda/Cyrce_Messungen/matrix_peek_121125/test__nA_1.9_x_108.0_y_28.25.csv',
                     '/Users/nico_brosda/Cyrce_Messungen/matrix_peek_121125/test__nA_1.9_x_107.5_y_28.25.csv']

norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_260325/')
norm_array1 = ['exp7_norm1,9V_']
# norm_array1 = ['2exp76_normday2_P0_']

cache = []
data_wheel_200 = pd.read_csv('../../Files/energies_after_wheel_diffusor200.txt', sep='\t', header=4,
                         names=['position', 'thickness', 'energies'])
param_cmap = sns.color_palette("crest_r", as_cmap=True)
comp_list = data_wheel_200['energies'].to_numpy()[:-1]
param_colormapper_200 = lambda param: color_mapper(param, np.min(comp_list), np.max(comp_list))
param_color = lambda param: param_cmap(param_colormapper_200(param))
param_unit = 'MeV'

for k, crit in enumerate(new_measurements[0:]):
    print('-'*50)
    print(crit)
    print('-'*50)

    A = Analyzer((2, 64), (0.4, 0.4), (0.1, 0.1), readout=readout,
                 diode_offset=[[0, - 0.25], np.zeros(64)], position_parser=position_parser,
                 voltage_parser=voltage_parser, current_parser=current_parser)
    dark = dark_paths_array1
    A.set_measurement(folder_path, crit)
    A.load_measurement()
    cache_save_raw = deepcopy(A.measurement_data)
    A.set_dark_measurement(dark_path, dark)
    norm = norm_array1
    norm_func = lambda list_of_files, instance, method='least_squares': normalization_from_translated_array_v3(
        list_of_files, instance, method, align_lines=True)
    A.normalization(norm_path, norm, normalization_module=norm_func)
    A.update_measurement()
    A.create_map(inverse=[True, False])
    intensity_limits = [0, np.max(A.maps[0]['z'])]

    # A.plot_map(results_path / 'maps/', pixel=True, intensity_limits=intensity_limits)
    A.plot_map(results_path / 'maps/', pixel='fill', intensity_limits=intensity_limits, imshow=True)
    # A.plot_map(results_path / 'maps/', pixel=False, intensity_limits=intensity_limits)
    # A.plot_map(results_path / 'maps/', pixel='fill', intensity_limits=intensity_limits, imshow=True)
    # A.plot_map(results_path / 'maps/', pixel=True, intensity_limits=intensity_limits, imshow=True)

    fig, ax = plt.subplots()
    ax.plot(A.maps[0]['y'], A.maps[0]['z'][:, 0], color=param_color(comp_list[k]))
    ax.set_xlabel('Position y Experiment (mm)')
    ax.set_ylabel('Signal Current (pA)')
    ax.set_xlim(10, 50)
    format_save(results_path / 'SignalCurves/', f'{crit}')

    cache.append([A.maps[0]['y'], A.maps[0]['z'][:, 0], A.maps[0]['z'][:, 1]])

    '''
    for i, image_map in enumerate(A.maps):
        A.maps[i]['z'] = simple_zero_replace(image_map['z'])
    # A.plot_map(results_path / 'maps_plus/', pixel=True, intensity_limits=intensity_limits)
    A.plot_map(results_path / 'maps_plus/', pixel='fill', intensity_limits=intensity_limits)
    # A.plot_map(results_path / 'maps_plus/', pixel=False, intensity_limits=intensity_limits)
    # A.plot_map(results_path / 'maps_plus/', pixel='fill', intensity_limits=intensity_limits, imshow=True)
    # A.plot_map(results_path / 'maps_plus/', pixel=True, intensity_limits=intensity_limits, imshow=True)
    '''

    continue

    A.measurement_data = cache_save_raw
    A.create_map(inverse=[True, False])
    print(np.max(np.max(A.maps[0]['z'])))
    # A.plot_map(results_path / 'raw/', pixel=True, intensity_limits=intensity_limits)
    A.plot_map(results_path / 'raw/', pixel='fill')
    # A.plot_map(results_path / 'raw/', pixel=False, intensity_limits=intensity_limits)
    # A.plot_map(results_path / 'raw/', pixel='fill', intensity_limits=intensity_limits, imshow=True)

    A.update_measurement(factor=False)
    A.create_map(inverse=[True, False])
    print(np.max(np.max(A.maps[0]['z'])))

    # A.plot_map(results_path / 'no_norm/', pixel=True, intensity_limits=intensity_limits)
    A.plot_map(results_path / 'no_norm/', pixel='fill', intensity_limits=intensity_limits)
    # A.plot_map(results_path / 'no_norm/', pixel=False, intensity_limits=intensity_limits)
    # A.plot_map(results_path / 'no_norm/', pixel='fill', intensity_limits=intensity_limits, imshow=True)

fig, ax = plt.subplots()

for k, item in enumerate(cache):
    ax.plot(item[0], item[1], color=param_color(comp_list[k]))
    ax.plot(item[0], item[2], color=param_color(comp_list[k]), ls='--')

ax.set_xlabel('Position y Experiment (mm)')
ax.set_ylabel('Signal Current (pA)')
ax.set_xlim(15, 50)
format_save(results_path / 'SignalCurves/', f'OverviewLineComp')

fig, ax = plt.subplots()

for k, item in enumerate(cache):
    ax.plot(item[0], (item[1]+item[2])/2, color=param_color(comp_list[k]))

ax.set_xlabel('Position y Experiment (mm)')
ax.set_ylabel('Signal Current (pA)')
ax.set_xlim(15, 50)
ax.set_ylim(ax.get_ylim())

gradient_arrow(ax, transform_axis_to_data_coordinates(ax, [0.1, 0.925]),
                       transform_axis_to_data_coordinates(ax, [0.1, 0.795]),
                   cmap=param_cmap, lw=10, zorder=5)
ax.text(*transform_axis_to_data_coordinates(ax, [0.035, 0.94]),
        f'{comp_list[k]: .2f}$\\,${param_unit}', fontsize=13, c=param_color(comp_list[k]),
        zorder=3, bbox={'facecolor': 'w', 'alpha': 0.8, 'pad': 2, 'edgecolor': 'w'})
ax.text(*transform_axis_to_data_coordinates(ax, [0.025, 0.71]),
        f'{np.max(comp_list): .2f}$\\,${param_unit}', fontsize=13, c=param_color(np.max(comp_list)),
        zorder=3, bbox={'facecolor': 'w', 'alpha': 0.8, 'pad': 2, 'edgecolor': 'w'})

format_save(results_path / 'SignalCurves/', f'Overview')

