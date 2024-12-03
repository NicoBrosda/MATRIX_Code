from EvaluationSoftware.main import *
from EvaluationSoftware.readout_modules import ams_channel_assignment_readout
from EvaluationSoftware.normalization_modules import normalization_from_translated_array

mapping = Path('../Files/mapping.xlsx')
data = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:])-1 for k in data['direction_2']]
readout, position_parser = lambda x, y: ams_2line_readout(x, y, channel_assignment=channel_assignment), standard_position

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Results_221024/2LineMaps/XRay/')

new_measurements = ['_MouseFoot_', '_MouseFoot2_']
live_scan_array1 = [str(round(i+1, 0))+'_live1_' for i in range(9)]
# new_measurements_array_matrix = ['']

dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')

dark_paths_array1 = ['2Line_DarkVoltageScan_200_ um_0_nA_nA_1.9_x_22.0_y_66.625.csv']

norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
norm_array1 = ['2Line_YScan_']

c_map = "Greys_r"
c_map = sns.color_palette(c_map, as_cmap=True)
for k, crit in enumerate(new_measurements[0:]):
    print('-'*50)
    print(crit)
    print('-'*50)

    A = Analyzer((2, 64), (0.4, 0.4), (0.1, 0.1), readout=readout, diode_offset=[[0, - 0.25], np.zeros(64)])
    dark = dark_paths_array1
    A.set_measurement(folder_path, crit)
    A.set_dark_measurement(dark_path, dark)
    norm = norm_array1
    norm_func = lambda list_of_files, instance, method='least_squares': normalization_from_translated_array_v3(
        list_of_files, instance, method, align_lines=True)
    A.normalization(norm_path, norm, normalization_module=norm_func)
    A.load_measurement()
    A.create_map(inverse=[True, False])
    intensity_limits = [4500, np.max(A.maps[0]['z'])]

    for i, image_map in enumerate(A.maps):
        A.maps[i]['z'] = simple_zero_replace(image_map['z'])

    fig, ax = plt.subplots()
    ax.hist(A.maps[0]['z'].flatten(), bins=500)
    ax.set_xlabel('Signal (a.u.)')
    format_save(results_path / crit, 'Hist_'+crit, legend=False, fig=fig, axes=[ax])

    save_format = '.png'
    dpi = 300
    plotsize_a = (latex_textwidth, latex_textwidth / 1.2419)
    plotsize_a = fullsize_plot

    A.plot_map(results_path / crit, pixel='fill', cmap=c_map, save_format=save_format, dpi=dpi, plot_size=plotsize_a,
               intensity_limits=intensity_limits)
    A.plot_map(results_path / crit, pixel=False, cmap=c_map, save_format=save_format, dpi=dpi, plot_size=plotsize_a,
               intensity_limits=intensity_limits)
    A.plot_map(results_path / crit, pixel='fill', cmap=c_map, save_format=save_format, dpi=dpi, plot_size=plotsize_a,
               intensity_limits=intensity_limits, imshow=True)



