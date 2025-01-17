from EvaluationSoftware.main import *
from EvaluationSoftware.helper_modules import rename_files

# To get the mapping of the 2d array correct I have to translate the mapping of the contacts
# The given info by The-Duc are for the 1 direction of mapping, but we have by standard the other mapping
# Therefore, I need the correct channel assignment between these two to replace the channels in the 2D pixel placement
mapping = Path('../../Files/mapping.xlsx')
direction1 = pd.read_excel(mapping, header=1)
direction1 = np.array([int(k[-3:]) for k in direction1['direction_1']])
direction2 = pd.read_excel(mapping, header=1)
direction2 = np.array([int(k[-3:]) for k in direction2['direction_2']])

mapping = Path('../../Files/Mapping_SmallMatrix2.xlsx')
data2 = pd.read_excel(mapping, header=None)
mapping_map = data2.to_numpy().flatten()
translated_mapping_small2 = np.array([direction2[np.argwhere(direction1 == i)[0][0]]-1 for i in mapping_map])

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_161224/')
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Results_161224/')

dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_161224/')

readout, position_parser, voltage_parser, current_parser = lambda x, y: ams_2D_assignment_readout(x, y, channel_assignment=translated_mapping_small2), standard_position, standard_voltage, current3
A = Analyzer((11, 11), 0.4, 0.1, readout=readout, voltage_parser=voltage_parser, current_parser=current_parser)

dark = ['exp1_dark_voltage_']
A.set_dark_measurement(dark_path, dark)

'''
norm_path = folder_path
norm = '8_2DSmall_yscan_'
norm_func = lambda list_of_files, instance, method='least_squares': normalization_from_translated_array_v3(
    list_of_files, instance, method, align_lines=True)
A.normalization(norm_path, norm, normalization_module=norm_func)
# '''

A.set_measurement(folder_path, 'exp2_voltage_')
# A.plot_for_parameter('voltage', True, [True, False], results_path / 'no_norm/', pixel='fill')

# '''
norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_211124/')
norm = '8_2DSmall_yscan_'
norm_func = lambda list_of_files, instance, method='least_squares': normalization_from_translated_array_v3(
    list_of_files, instance, method, align_lines=True)
A.normalization(norm_path, norm, normalization_module=norm_func)
A.norm_factor[3][5] = 1
A.norm_factor[4][5] = 1
# '''
A.set_measurement(folder_path, 'exp2_voltage_')
# A.plot_for_parameter('voltage', True, [True, False], results_path / 'maps/', pixel='fill')

measurements = (['exp3_bragg_p01_']+[f'exp4_bragg_p0{i}_' for i in range(2, 10)]+
                [f'exp4_bragg_p{i}_' for i in range(10, 21)]+['exp6_star_p01_', 'exp7_star_p06_', 'exp8_star_p06_',
                                                              'exp10_star_p01_', 'exp13_star_p15_', 'exp14_star_p09_'])

'''
# Line Scan image by image
results_path2 = results_path / '8_2DSmall_yscan_'

A.set_measurement(folder_path, norm)
measurement_files = np.array([str(i)[len(str(folder_path))+1:] for i in A.measurement_files], dtype=str)
measurement_files = measurement_files[np.argsort([standard_position(i)[1] for i in measurement_files])]

for file in tqdm(measurement_files[0:], colour='blue'):
    A.set_measurement(folder_path, file)
    A.load_measurement()
    A.create_map(inverse=[True, False])
    intensity_limits = [0, np.max(A.maps[0]['z'])]
    A.plot_map(results_path2, pixel='fill', intensity_limits=intensity_limits)

# '''

for k, crit in enumerate(measurements[0:]):
    print('-'*50)
    print(crit)
    print('-'*50)

    A = Analyzer((11, 11), 0.4, 0.1, readout=readout, voltage_parser=voltage_parser, current_parser=current_parser)
    A.set_measurement(folder_path, crit)
    if 'bragg' in crit:
        dark = ['exp1_dark_voltage_scan_nA_1.9_x_22.0_y_67.75']
    elif 'star' in crit:
        dark = ['exp1_dark_voltage_scan_nA_1.1_x_22.0_y_67.75']
    A.set_dark_measurement(dark_path, dark)
    A.normalization(norm_path, norm, normalization_module=norm_func)
    A.norm_factor[3][5] = 1
    A.norm_factor[4][5] = 1
    A.load_measurement()
    A.create_map(inverse=[True, False])

    intensity_limits = [0, np.max(A.maps[0]['z'])]

    A.plot_map(results_path / 'maps/', pixel=True, intensity_limits=intensity_limits)
    A.plot_map(results_path / 'maps/', pixel='fill', intensity_limits=intensity_limits)
    A.plot_map(results_path / 'maps/', pixel=False, intensity_limits=intensity_limits)
    A.plot_map(results_path / 'maps/', pixel='fill', intensity_limits=intensity_limits, imshow=True)
    A.plot_map(results_path / 'maps/', pixel=True, intensity_limits=intensity_limits, imshow=True)


    A = Analyzer((11, 11), 0.4, 0.1, readout=readout)
    A.set_measurement(folder_path, crit)
    A.load_measurement()
    A.create_map(inverse=[True, False])

    A.plot_map(results_path / 'raw/', pixel=True, intensity_limits=intensity_limits)
    A.plot_map(results_path / 'raw/', pixel='fill', intensity_limits=intensity_limits)
    A.plot_map(results_path / 'raw/', pixel=False, intensity_limits=intensity_limits)
    A.plot_map(results_path / 'raw/', pixel='fill', intensity_limits=intensity_limits, imshow=True)

    A.set_dark_measurement(dark_path, dark)
    A.update_measurement(factor=False)
    A.create_map(inverse=[True, False])

    A.plot_map(results_path / 'no_norm/', pixel=True, intensity_limits=intensity_limits)
    A.plot_map(results_path / 'no_norm/', pixel='fill', intensity_limits=intensity_limits)
    A.plot_map(results_path / 'no_norm/', pixel=False, intensity_limits=intensity_limits)
    A.plot_map(results_path / 'no_norm/', pixel='fill', intensity_limits=intensity_limits, imshow=True)
# '''