import numpy as np

from EvaluationSoftware.main import *
import time
import scipy as sp
from Concept8GafMeasurementComparison import GafImage
from Concept8GafCompTests import align_and_compare_images, resample_image

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black", "red", "yellow"])
cmap_r = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "red"])
cmap_b = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "blue"])
cmap2 = sns.color_palette('viridis', as_cmap=True)

mapping = Path('../../Files/mapping.xlsx')
data = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:]) - 1 for k in data['direction_2']]
readout, position_parser = lambda x, y: ams_2line_readout(x, y,
                                                          channel_assignment=channel_assignment), standard_position

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Results_221024/GafComp/')

new_measurements = ['_GafComp200_', '_GafComp400_', '_GafComp40_', '_GafCompLogo_', '_GafCompMisc_', '_GafCompPEEK_',
                    '_MouseFoot_', '_MouseFoot2_', '2Line_Beam_']
Gaf_path = Path('/Users/nico_brosda/Cyrce_Messungen/Gafchromic_221024/')
Gaf_map = ['gafchromic_matrix211024_003.bmp', 'gafchromic_matrix211024_004.bmp', 'gafchromic_matrix211024_005.bmp',
           'gafchromic_matrix211024_006.bmp', 'gafchromic_matrix211024_007.bmp', 'gafchromic_matrix211024_008.bmp',
           'gafchromic_matrix211024_009.bmp', 'gafchromic_matrix211024_009.bmp', 'gafchromic_matrix211024_010.bmp']
live_scan_array1 = [str(round(i + 1, 0)) + '_live1_' for i in range(9)]
# new_measurements_array_matrix = ['']

dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')

dark_paths_array1 = ['2Line_DarkVoltageScan_200_ um_0_nA_nA_1.9_x_22.0_y_66.625.csv']

norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
norm_array1 = ['2Line_YScan_']

for k, crit in enumerate(new_measurements[0:]):
    print('-' * 50)
    print(crit)
    print('-' * 50)

    A = Analyzer((2, 64), (0.4, 0.4), (0.1, 0.1), readout=readout, diode_offset=[[0, - 0.25], np.zeros(64)])
    dark = dark_paths_array1
    A.set_measurement(folder_path, crit)
    A.set_dark_measurement(dark_path, dark)
    norm = norm_array1
    norm_func = lambda list_of_files, instance, method='least_squares': normalization_from_translated_array_v3(
        list_of_files, instance, method, align_lines=True)
    A.normalization(norm_path, norm, normalization_module=norm_func)
    A.load_measurement()
    A.create_map(inverse=[False, False])
    for i, image_map in enumerate(A.maps):
        A.maps[i]['z'] = simple_zero_replace(image_map['z'])

    print(np.shape(A.maps[0]['z']))
    print(A.maps[0]['x'][1] - A.maps[0]['x'][0])
    print(A.maps[0]['y'][1] - A.maps[0]['y'][0])


    A.maps[0]['x'], A.maps[0]['y'], A.maps[0]['z'] = homogenize_pixel_size(
        [A.maps[0]['x'], A.maps[0]['y'], np.abs(A.maps[0]['z']) / np.max(A.maps[0]['z'])])

    print(np.shape(A.maps[0]['z']))
    print(A.maps[0]['x'][1] - A.maps[0]['x'][0])

    OriginalGaf = GafImage(Gaf_path / Gaf_map[k])
    OriginalGaf.load_image()
    if 'matrix211024_006.bmp' in Gaf_map[k]:
        OriginalGaf.image = OriginalGaf.image[::-1]
        OriginalGaf.image = OriginalGaf.image[:, ::-1]
    OriginalGaf.transform_to_normed(max_n=1e5)
    print('-' * 50)

    # Image down sampling (global to save time):
    low_pixel_size = A.maps[0]['x'][1] - A.maps[0]['x'][0]
    DownSampGaf = GafImage(Gaf_path / Gaf_map[k])
    quick_load = ((Gaf_path / Gaf_map[k]).parent / 'QuickLoads') / (Gaf_map[k][:-4]+'.npy')
    print(quick_load)
    if os.path.isfile(quick_load):
        DownSampGaf.load_image(quick=True)
        down_samp = DownSampGaf.image
    else:
        down_samp = resample_image(OriginalGaf.image, OriginalGaf.pixel_size, low_pixel_size)
        DownSampGaf.image = down_samp
        DownSampGaf.save_image(quick_load.parent)

    if 'PEEK' in crit:
        max_shape = (max(down_samp.shape[0], A.maps[0]['z'].shape[0]), max(down_samp.shape[1], A.maps[0]['z'].shape[1]))
        bounds = [(-30, 30), (-max_shape[0] // 5, max_shape[0] // 5), (-max_shape[1] // 5, max_shape[1] // 5)]
    else:
        bounds = (-3, 3)

    diff, score, addition = align_and_compare_images(A.maps[0]['z'], OriginalGaf.image, low_pixel_size, OriginalGaf.pixel_size,
                                                        optimize_alignment=True, bounds=bounds, ev_max_iter=500,
                                                        ev_pop_size=10,
                                                        optimization_method='evolutionary',
                                                        image_down_sampled=down_samp)

    matrix_image = addition[-1]
    gafchromic_image = addition[-2]
    image_shape = np.shape(matrix_image)
    print(f'Image has shape {image_shape} corresponding to size (x, y) ({image_shape[1]*low_pixel_size}, {image_shape[0]*low_pixel_size}) mm.')
    if np.shape(gafchromic_image) == image_shape:
        print(f'Images align in their size')
    else:
        print(f'Images do not align in their size, trafo image has shape {image_shape}, while down sampled Gafchromic '
              f'image has shape {np.shape(gafchromic_image)}')
        break

    # ------------------------------------------------------------------------------------------------------------------
    # Plot: Cross sections with r/b cs lines and alpha map overlay
    # ------------------------------------------------------------------------------------------------------------------

    fig, (ax1, ax2) = plt.subplots(1, 2)
    # Plot both measurement signal onto one axis - still have to see if same colour map or different one should be used
    color_map1 = ax1.imshow(matrix_image, vmin=0, vmax=1, cmap=cmap,
                            extent=(0, np.shape(matrix_image)[1] * low_pixel_size, 0,
                                    np.shape(matrix_image)[0] * low_pixel_size))
    color_map2 = ax1.imshow(gafchromic_image, vmin=0, vmax=1, cmap=cmap, alpha=0.5,
                           extent=(0, np.shape(gafchromic_image)[1] * low_pixel_size, 0, np.shape(gafchromic_image)[0] * low_pixel_size))

    # Creation of colorbar
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map1.cmap)
    sm.set_array([])
    bar = fig.colorbar(sm, ax=ax1, extend='max')
    bar.set_label('Normed detector response')

    # Axis labeling
    ax1.set_xlabel('Position x (mm)')
    ax1.set_ylabel('Position y (mm)')

    # Creation of 2 intersection lines
    csx_positions = [None, None, None, 19.5, 30, 23, 22, 22, 26]
    csy_positions = [22.5, 22.5, 25, 28, 20, None, 22, 22, 24]
    if csx_positions[k] is not None:
        middle_x = image_shape[1] // 2
        middle_x = int(csx_positions[k] // low_pixel_size)
        ax1.axvline(middle_x*low_pixel_size, color='b', ls='-', lw=3)
        ax1.arrow(middle_x*low_pixel_size, 3, 0, 1, color='b', ls='-', lw=3, length_includes_head=True,
              head_width=1, head_length=1.)

    xs = np.arange(0, matrix_image.shape[0]) * low_pixel_size
    ys = np.arange(0, matrix_image.shape[1]) * low_pixel_size

    if csy_positions[k] is not None:
        middle_y = image_shape[0] // 2
        y_pos = image_shape[0] - int(csy_positions[k] // low_pixel_size)
        y_pos_plot = int(csy_positions[k] // low_pixel_size)
        ax1.axhline(y_pos_plot*low_pixel_size, color='r', ls='-', lw=3)
        ax1.arrow(3, y_pos_plot*low_pixel_size, 1, 0, color='r', ls='-', lw=3, length_includes_head=True,
          head_width=1, head_length=1.)

    # ----------  ----------  ----------
    # Ax2: The diagrams of the cross-section data

    if csx_positions[k] is not None:
        ax2.plot(xs, matrix_image[::-1, middle_x], color='b', ls='-', label='MATRIX response cross-section x', marker='x')
        ax2.plot(xs, gafchromic_image[::-1, middle_x], color='b', ls=':', label='Gafchromic response cross-section x', marker='o')

    if csy_positions[k] is not None:
        ax2.plot(ys, matrix_image[y_pos, :], color='r', ls='-', label='MATRIX response cross-section y', marker='x')
        ax2.plot(ys, gafchromic_image[y_pos, :], color='r', ls=':', label='Gafchromic response cross-section y', marker='o')

    # Axis labeling
    ax2.set_xlabel('Axis Position (mm)')
    ax2.set_ylabel('Normed detector response')
    ax2.legend(loc=3)

    # ----------  ----------  ----------
    # Plot saving
    plot_size = (latex_textwidth * 2, latex_textwidth / 1.2419)
    name = A.name + '_CrossSectionVersion1_'
    format_save(results_path / A.name, save_name=name, save=True, legend=False, fig=fig, plot_size=plot_size)

    # ------------------------------------------------------------------------------------------------------------------
    # Plot: Cross sections with styled cs lines and color_channel map overlay
    # ------------------------------------------------------------------------------------------------------------------

    fig, (ax1, ax2) = plt.subplots(1, 2)
    # Plot both measurement signal onto one axis - still have to see if same colour map or different one should be used
    color_map1 = ax1.imshow(matrix_image, vmin=0, vmax=1, cmap=cmap_b,
                            extent=(0, np.shape(matrix_image)[1] * low_pixel_size, 0,
                                    np.shape(matrix_image)[0] * low_pixel_size))
    color_map2 = ax1.imshow(gafchromic_image, vmin=0, vmax=1, cmap=cmap_r, alpha=0.5,
                            extent=(0, np.shape(gafchromic_image)[1] * low_pixel_size, 0,
                                    np.shape(gafchromic_image)[0] * low_pixel_size))

    # Creation of colorbar
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map2.cmap)
    sm.set_array([])
    bar = fig.colorbar(sm, ax=ax1, extend='max')
    bar.set_label('Normed detector response')

    # Axis labeling
    ax1.set_xlabel('Position x (mm)')
    ax1.set_ylabel('Position y (mm)')

    # Creation of 2 intersection lines
    if csx_positions[k] is not None:
        ax1.axvline(middle_x*low_pixel_size, color='k', ls='--', lw=3)
        ax1.arrow(middle_x*low_pixel_size, 3, 0, 1, color='k', ls='-', lw=3, length_includes_head=True,
            head_width=1, head_length=1.)
    if csy_positions[k] is not None:
        ax1.axhline(y_pos_plot*low_pixel_size, color='k', ls=':', lw=3)
        ax1.arrow(3, y_pos_plot * low_pixel_size, 1, 0, color='k', ls='-', lw=3, length_includes_head=True,
          head_width=1, head_length=1.)

    # ----------  ----------  ----------
    # Ax2: The diagrams of the cross-section data
    if csx_positions[k] is not None:
        ax2.plot(xs, matrix_image[::-1, middle_x], color='b', ls='--', label='MATRIX response cross-section x', marker='x')
        ax2.plot(xs, gafchromic_image[::-1, middle_x], color='r', ls=':', label='Gafchromic response cross-section x', marker='o')

    if csy_positions[k] is not None:
        ax2.plot(ys, matrix_image[y_pos, :], color='b', ls='--', label='MATRIX response cross-section y', marker='x')
        ax2.plot(ys, gafchromic_image[y_pos, :], color='r', ls=':', label='Gafchromic response cross-section y', marker='o')

    # Axis labeling
    ax2.set_xlabel('Axis Position (mm)')
    ax2.set_ylabel('Normed detector response')
    ax2.legend(loc=3)

    # ----------  ----------  ----------
    # Plot saving
    plot_size = (latex_textwidth * 2, latex_textwidth / 1.2419)
    name = A.name + '_CrossSectionVersion2_'
    format_save(results_path / A.name, save_name=name, save=True, legend=False, fig=fig, plot_size=plot_size)

