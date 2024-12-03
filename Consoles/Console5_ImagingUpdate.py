from EvaluationSoftware.main import *
from EvaluationSoftware.readout_modules import ams_channel_assignment_readout
from EvaluationSoftware.normalization_modules import normalization_from_translated_array

mapping = Path('../Files/mapping.xlsx')
data = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:])-1 for k in data['direction_2']]
readout, position_parser = lambda x, y: ams_2line_readout(x, y, channel_assignment=channel_assignment), standard_position

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Results_221024/ImageUpdate/')
results_path2 = Path('/Users/nico_brosda/Cyrce_Messungen/Results_221024/ImageUpdate/DPI/')
results_path3 = Path('/Users/nico_brosda/Cyrce_Messungen/Results_221024/ImageUpdate/ColorMap/')

new_measurements = ['_GafComp200_', '_GafComp400_', '_GafComp40_', '_GafCompLogo_', '_GafCompMisc_', '_GafCompPEEK_',
                    '_MouseFoot_', '_MouseFoot2_', '2Line_Beam_']
new_measurements = ['_GafComp200_', '_GafCompLogo_', '_GafCompMisc_', '_MouseFoot2_', '2Line_Beam_', '_GafCompPEEK_']
new_measurements = ['_GafCompMisc_']

dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')

dark_paths_array1 = ['2Line_DarkVoltageScan_200_ um_0_nA_nA_1.9_x_22.0_y_66.625.csv']

norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
norm_array1 = ['2Line_YScan_']

txt_posi = [0.03, 0.93]
for k, crit in enumerate(new_measurements[0:]):
    print('-'*50)
    print(crit)
    print('-'*50)

    A = Analyzer((2, 64), (0.4, 0.4), (0.1, 0.1), readout=readout, diode_offset=[[0, - 0.25], np.zeros(64)])
    dark = dark_paths_array1

    A.set_measurement(folder_path, crit)
    A.set_dark_measurement(dark_path, dark)
    norm = norm_array1
    norm_func = lambda list_of_files, instance, method='least_squares': normalization_from_translated_array_v3(list_of_files, instance, method, align_lines=True)
    A.normalization(norm_path, norm, normalization_module=norm_func)
    A.load_measurement()
    A.create_map(inverse=[True, False])

    for i, image_map in enumerate(A.maps):
        A.maps[i]['z'] = simple_zero_replace(image_map['z'])
    # A.maps[0] = overlap_treatment(A.maps[0], A, True, super_res=True)

    # '''
    if 'MouseFoot' in crit:
        intensity_limits = [4500, np.max(A.maps[0]['z'])]
        c_map = "Greys_r"
        c_map = sns.color_palette(c_map, as_cmap=True)
    else:
        intensity_limits = [0, np.max(A.maps[0]['z'])]
        c_map = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black", "red", "yellow"])

    A.name = 'PColormesh_Whitespace'
    A.plot_map(results_path / (str(crit)+'/'), pixel=True, intensity_limits=intensity_limits, insert_txt=[txt_posi, A.name, 15], cmap=c_map)
    A.name = 'PColormesh_Filled'
    A.plot_map(results_path / (str(crit)+'/'), pixel='fill', intensity_limits=intensity_limits, insert_txt=[txt_posi, A.name, 15], cmap=c_map)
    A.name = 'Contourf'
    A.plot_map(results_path / (str(crit)+'/'), pixel=False, intensity_limits=intensity_limits, insert_txt=[txt_posi, A.name, 15], cmap=c_map)

    for interpolation in ['none', 'antialiased', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning',
                          'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc',
                          'lanczos', 'blackman']:
        if interpolation in ['none', 'antialiased', 'nearest', 'bilinear']:
            A.name = 'Imshow_Whitespace_' + interpolation
            A.plot_map(results_path / (str(crit) + '/'), pixel=True, imshow=interpolation,
                       intensity_limits=intensity_limits, insert_txt=[txt_posi, A.name, 15], cmap=c_map)
        A.name = 'Imshow_Filled_' + interpolation
        A.plot_map(results_path / (str(crit) + '/'), pixel='Fill', imshow=interpolation,
                   intensity_limits=intensity_limits, insert_txt=[txt_posi, A.name, 15], cmap=c_map)

    for dpi in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1200]:
        A.name = str(dpi) + 'PColormesh_Filled'
        A.plot_map(results_path2, pixel='fill', dpi=dpi, intensity_limits=intensity_limits, insert_txt=[txt_posi, A.name, 15], cmap=c_map)
        A.name = str(dpi) + 'PColormesh_Whitespace'
        A.plot_map(results_path2, pixel=True, dpi=dpi, intensity_limits=intensity_limits, insert_txt=[txt_posi, A.name, 15], cmap=c_map)
        for interpolation in ['none', 'antialiased', 'bilinear']:
            A.name = str(dpi) + 'Imshow_Whitespace_' + interpolation
            A.plot_map(results_path2, pixel=True, imshow=interpolation, dpi=dpi, cmap=c_map,
                       intensity_limits=intensity_limits, insert_txt=[txt_posi, A.name, 15])
            A.name = str(dpi) + 'Imshow_Filled_' + interpolation
            A.plot_map(results_path2, pixel='Fill', imshow=interpolation, dpi=dpi, cmap=c_map,
                       intensity_limits=intensity_limits, insert_txt=[txt_posi, A.name, 15])

    # '''
    # cmap testing
    for cmap in tqdm(["rocket", "rocket_r", "mako", "mako_r", "flare", "flare_r", "crest", "crest_r", "magma", "magma_r",
                 "viridis", "viridis_r", "plasma", "plasma_r", "inferno", "inferno_r", "cividis", "cividis_r", "Greys",
                 "Greys_r", "Spectral", "Spectral_r", "jet", "jet_r", "self_design01"]):
        if "self_design01" in cmap:
            c_map = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black", "red", "yellow"])
        elif "jet" in cmap:
            c_map = matplotlib.colormaps[cmap]
        else:
            c_map = sns.color_palette(cmap, as_cmap=True)
        print(cmap)

        plotsize_adapt = fullsize_plot
        A.name = 'PColormesh_' + cmap
        A.plot_map(results_path3 / (str(crit) + '/'), pixel='Fill', intensity_limits=intensity_limits, cmap=c_map,
                   plot_size=plotsize_adapt, insert_txt=[txt_posi, A.name, 15])
        A.name = 'Contour_Cmap_' + cmap
        A.plot_map(results_path3 / (str(crit) + '/'), pixel=False, intensity_limits=intensity_limits, cmap=c_map,
                   plot_size=plotsize_adapt, insert_txt=[txt_posi, A.name, 15])
        A.name = 'Imshow_Cmap_' + 'antialiased_' + cmap
        A.plot_map(results_path3 / (str(crit) + '/'), pixel='Fill', imshow='antialiased',
                   intensity_limits=intensity_limits, cmap=c_map, insert_txt=[txt_posi, A.name, 15])
        A.name = 'Imshow_Cmap_' + 'lanczos_' + cmap
        A.plot_map(results_path3 / (str(crit) + '/'), pixel='Fill', imshow='lanczos',
                   intensity_limits=intensity_limits, cmap=c_map, insert_txt=[txt_posi, A.name, 15])

    # '''
