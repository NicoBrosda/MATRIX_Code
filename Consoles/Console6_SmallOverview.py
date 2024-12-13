from EvaluationSoftware.main import *
from EvaluationSoftware.helper_modules import rename_files

# To get the mapping of the 2d array correct I have to translate the mapping of the contacts
# The given info by The-Duc are for the 1 direction of mapping, but we have by standard the other mapping
# Therefore, I need the correct channel assignment between these two to replace the channels in the 2D pixel placement
mapping = Path('../Files/mapping.xlsx')
direction1 = pd.read_excel(mapping, header=1)
direction1 = np.array([int(k[-3:]) for k in direction1['direction_1']])
direction2 = pd.read_excel(mapping, header=1)
direction2 = np.array([int(k[-3:]) for k in direction2['direction_2']])

mapping = Path('../Files/Mapping_SmallMatrix2.xlsx')
data2 = pd.read_excel(mapping, header=None)
mapping_map = data2.to_numpy().flatten()
translated_mapping = np.array([direction2[np.argwhere(direction1 == i)[0][0]]-1 for i in mapping_map])

readout, position_parser = lambda x, y: ams_2D_assignment_readout(x, y, channel_assignment=translated_mapping), standard_position

A = Analyzer((11, 11), 0.4, 0.1, readout=readout)

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_211124/')
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Results_221124/2DSmall/')

dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_211124/')
matrix_dark = ['12_2DSmall_miscshape_']
A.set_dark_measurement(dark_path, matrix_dark)

norm_path = folder_path
norm = '8_2DSmall_yscan_'
norm_func = lambda list_of_files, instance, method='least_squares': normalization_from_translated_array_v3(
        list_of_files, instance, method, align_lines=True)
A.normalization(norm_path, norm, normalization_module=norm_func)

measurements = ['9_2DSmall_miscshape_xyscan_', '10_2DSmall_miscshape_xyscan_',
                '14_2DSmall_misc_xyscan_', '15_2DSmall_misc_xyscan_', '16_2DSmall_foot_xyscan_',
                '17_2DSmall_foot_xyscan_', '18_2DSmall_foot_xyscan_', '20_2DSmall_iphc_crhea_xyscan_']

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
for k, crit in enumerate(measurements[4 :]):
    print('-'*50)
    print(crit)
    print('-'*50)

    A = Analyzer((11, 11), 0.4, 0.1, readout=readout)
    dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_211124/')
    matrix_dark = ['12_2DSmall_miscshape_']

    A.set_measurement(folder_path, crit)
    A.set_dark_measurement(dark_path, matrix_dark)
    norm_path = folder_path
    norm = '8_2DSmall_yscan_'
    norm_func = lambda list_of_files, instance, method='least_squares': normalization_from_translated_array_v3(
        list_of_files, instance, method, align_lines=True)
    A.normalization(norm_path, norm, normalization_module=norm_func)
    A.load_measurement()
    print('Here')
    A.create_map(inverse=[True, False])
    print('Here')

    intensity_limits = [0, np.max(A.maps[0]['z'])]

    A.plot_map(results_path / 'maps/', pixel=True, intensity_limits=intensity_limits)
    A.plot_map(results_path / 'maps/', pixel='fill', intensity_limits=intensity_limits)
    A.plot_map(results_path / 'maps/', pixel=False, intensity_limits=intensity_limits)
    A.plot_map(results_path / 'maps/', pixel='fill', intensity_limits=intensity_limits, imshow=True)
    A.plot_map(results_path / 'maps/', pixel=True, intensity_limits=intensity_limits, imshow=True)

    for i, image_map in enumerate(A.maps):
        A.maps[i]['z'] = simple_zero_replace(image_map['z'])
    A.plot_map(results_path / 'maps_plus/', pixel=True, intensity_limits=intensity_limits)
    A.plot_map(results_path / 'maps_plus/', pixel='fill', intensity_limits=intensity_limits)
    A.plot_map(results_path / 'maps_plus/', pixel=False, intensity_limits=intensity_limits)
    A.plot_map(results_path / 'maps_plus/', pixel='fill', intensity_limits=intensity_limits, imshow=True)
    A.plot_map(results_path / 'maps_plus/', pixel=True, intensity_limits=intensity_limits, imshow=True)

    if 'foot' in crit:
        fig, ax = plt.subplots()
        ax.hist(A.maps[0]['z'].flatten(), bins=500)
        ax.set_xlabel('Signal (a.u.)')
        format_save(results_path / 'XRay/', 'Hist_' + crit, legend=False, fig=fig, axes=[ax])
        c_map = "Greys_r"
        c_map = sns.color_palette(c_map, as_cmap=True)
        intensity_limitsXRay = [5000, np.max(A.maps[0]['z'])]
        A.plot_map(results_path / 'XRay/', pixel='fill', intensity_limits=intensity_limitsXRay, cmap=c_map)

    A = Analyzer((11, 11), 0.4, 0.1, readout=readout)
    A.set_measurement(folder_path, crit)
    A.load_measurement()
    A.create_map(inverse=[True, False])

    A.plot_map(results_path / 'raw/', pixel=True, intensity_limits=intensity_limits)
    A.plot_map(results_path / 'raw/', pixel='fill', intensity_limits=intensity_limits)
    A.plot_map(results_path / 'raw/', pixel=False, intensity_limits=intensity_limits)
    A.plot_map(results_path / 'raw/', pixel='fill', intensity_limits=intensity_limits, imshow=True)

    A.set_dark_measurement(dark_path, matrix_dark)
    A.update_measurement(factor=False)
    A.create_map(inverse=[True, False])

    A.plot_map(results_path / 'no_norm/', pixel=True, intensity_limits=intensity_limits)
    A.plot_map(results_path / 'no_norm/', pixel='fill', intensity_limits=intensity_limits)
    A.plot_map(results_path / 'no_norm/', pixel=False, intensity_limits=intensity_limits)
    A.plot_map(results_path / 'no_norm/', pixel='fill', intensity_limits=intensity_limits, imshow=True)
# '''
