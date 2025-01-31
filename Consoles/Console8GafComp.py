from EvaluationSoftware.main import *
import time
import scipy as sp
from Concept8GafMeasurementComparison import GafImage
from Concept8GafCompTests import align_and_compare_images, resample_image

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black", "red", "yellow"])
cmap2 = sns.color_palette('viridis', as_cmap=True)

mapping = Path('../Files/mapping.xlsx')
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

    A.maps[0]['x'], A.maps[0]['y'], A.maps[0]['z'] = homogenize_pixel_size(
        [A.maps[0]['x'], A.maps[0]['y'], np.abs(A.maps[0]['z']) / np.max(A.maps[0]['z'])])

    Test = GafImage(Gaf_path / Gaf_map[k])
    Test.load_image()
    if 'matrix211024_006.bmp' in Gaf_map[k]:
        Test.image = Test.image[::-1]
        Test.image = Test.image[:, ::-1]
    Test.transform_to_normed(max_n=1e5)
    print('-' * 50)
    '''
    print(len(A.maps[0]['x']))
    print(A.maps[0]['x'])
    print(len(A.maps[0]['y']))
    print(A.maps[0]['y'])
    print(np.shape((A.maps[0]['z'])))

    def interp2D(input_coords, data_points, interp_coords):
        cache = []
        for row in data_points.T:
            cache.append(np.interp(interp_coords[0], input_coords[0], row))

        cache = np.array(cache)
        print(np.shape(cache))
        print(cache)

        cache2 = []
        for col in cache.T:
            cache2.append(np.interp(interp_coords[1], input_coords[1], col))

        cache2 = np.array(cache2)
        print(np.shape(cache2))
        print(cache2)

        return cache2

    minimal_ext_x = min(np.shape(Test.image)[1] * Test.pixel_size, -np.min(A.maps[0]['x']) + np.max(A.maps[0]['x']))
    minimal_ext_y = min(np.shape(Test.image)[0] * Test.pixel_size, - np.min(A.maps[0]['y']) + np.max(A.maps[0]['y']))
    newx = np.linspace(0, minimal_ext_x, int(minimal_ext_x/Test.pixel_size))
    newy = np.linspace(0, minimal_ext_y, int(minimal_ext_y/Test.pixel_size))

    A2 = interp2D((A.maps[0]['x']-np.min(A.maps[0]['x']), A.maps[0]['y']-np.min(A.maps[0]['y'])), A.maps[0]['z'].T, (newx, newy))

    print(np.shape(A2))
    print('_'*50)
    print('Extent of Gaf is: ', (np.shape(Test.image)[1] * Test.pixel_size, np.shape(Test.image)[0] * Test.pixel_size))
    print('Extent of Map is: ', (-np.min(A.maps[0]['x']) + np.max(A.maps[0]['x']), - np.min(A.maps[0]['y']) + np.max(A.maps[0]['y'])))


    def build_image_subtraction(image1, image2):
        # WLOG: Image1 is the smaller one (one dimension of the images has to match)
        if np.sum(np.shape(image1)) > np.sum(np.shape(image2)):
            image1, image2 = image2, image1

        # Take a certain range out of image2 around the center

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.imshow(A.maps[0]['z'], cmap=cmap, vmin=0, vmax=1, extent=(np.min(A.maps[0]['x']), np.max(A.maps[0]['x']), np.min(A.maps[0]['y']), np.max(A.maps[0]['y'])))
    ax2.imshow(A2.T, cmap=cmap, vmin=0, vmax=1, extent=(0, minimal_ext_x, 0, minimal_ext_y))

    ax3.imshow(Test.image, cmap=cmap, vmin=0, vmax=1, extent=(0, np.shape(Test.image)[1] * Test.pixel_size, 0, np.shape(Test.image)[0] * Test.pixel_size))
    plt.show()
    '''
    # Image down sampling (global to save time):
    low_pixel_size = A.maps[0]['x'][1] - A.maps[0]['x'][0]
    down_samp = resample_image(Test.image, Test.pixel_size, low_pixel_size)

    diff, score, addition = align_and_compare_images(A.maps[0]['z'], Test.image, A.maps[0]['x'][1] - A.maps[0]['x'][0],
                                             Test.pixel_size, center_position=[0, 0], optimize_alignment=False,
                                           image_down_sampled=down_samp)
    diff2, score2, addition2 = align_and_compare_images(A.maps[0]['z'], Test.image, low_pixel_size, Test.pixel_size,
                                               optimize_alignment=True, bounds=(-3, 3), ev_max_iter=500, ev_pop_size=10,
                                               optimization_method='evolutionary', image_down_sampled=down_samp)

    # -------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------

    fig, (ax1, ax2) = plt.subplots(1, 2)
    color_map = ax1.imshow(diff2, vmin=-1, vmax=1,
                           extent=(0, np.shape(diff2)[1] * low_pixel_size, 0, np.shape(diff2)[0] * low_pixel_size))
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map.cmap)
    sm.set_array([])
    bar = fig.colorbar(sm, ax=ax1, extend='max')
    ax1.set_xlabel('Position x (mm)')
    ax1.set_ylabel('Position y (mm)')
    ax1.set_title(f'Alignment Score = {score2: .4f}')
    bar.set_label('Difference between images')

    ax2.hist(diff2.flatten(), bins=100)
    ax2.set_xlabel('Differences between normed maps')
    plot_size = (latex_textwidth * 3, latex_textwidth / 1.5 / 1.2419)
    name = A.name + '_HistogramNotNormed_'
    format_save(results_path / A.name, save_name=name, save=True, legend=False, fig=fig)

    # -------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------

    diff, diff2 = np.abs(diff), np.abs(diff2)
    # print("Alignment Score high:", score)
    print("Alignment Score low:", score2)

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
    # -------------------------------------------------------------------------------------------------------------------
    # Ax1: Plot of original map normed to 1
    # -------------------------------------------------------------------------------------------------------------------
    color_map1 = ax1.imshow(A.maps[0]['z'], cmap=cmap, vmin=0, vmax=1, extent=(
    np.min(A.maps[0]['x']), np.max(A.maps[0]['x']), np.min(A.maps[0]['y']), np.max(A.maps[0]['y'])))
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map1.cmap)
    sm.set_array([])
    bar1 = fig.colorbar(sm, ax=ax1, extend='max')
    ax1.set_xlabel('Position x (mm)')
    ax1.set_ylabel('Position y (mm)')
    bar1.set_label('Normed Signal')
    # -------------------------------------------------------------------------------------------------------------------
    # Ax2: Plot of GafImage transformed
    # -------------------------------------------------------------------------------------------------------------------
    color_map2 = ax2.imshow(Test.image, cmap=cmap, vmin=0, vmax=1, extent=(
    0, np.shape(Test.image)[1] * Test.pixel_size, 0, np.shape(Test.image)[0] * Test.pixel_size))
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map2.cmap)
    sm.set_array([])
    bar2 = fig.colorbar(sm, ax=ax2, extend='max')
    ax2.set_xlabel('Position x (mm)')
    ax2.set_ylabel('Position y (mm)')
    bar2.set_label('Gafchromic response')
    # -------------------------------------------------------------------------------------------------------------------
    # Ax3: Plot of GafImage transformed
    # -------------------------------------------------------------------------------------------------------------------
    ax3.hist(A.maps[0]['z'].flatten(), bins=100)
    ax3.set_xlabel('Normed signal')
    # -------------------------------------------------------------------------------------------------------------------
    # Ax4: Plot of GafImage transformed
    # -------------------------------------------------------------------------------------------------------------------
    ax4.hist(Test.image.flatten(), bins=100)
    ax4.set_xlabel('Gafchromic response')
    # -------------------------------------------------------------------------------------------------------------------
    # Ax5: Plot of GafImage transformed
    # -------------------------------------------------------------------------------------------------------------------
    print(np.min(diff), np.max(diff))
    color_map5 = ax5.imshow(diff, vmin=0, vmax=1,
                            extent=(0, np.shape(diff)[1] * low_pixel_size, 0, np.shape(diff)[0] * low_pixel_size))
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map5.cmap)
    sm.set_array([])
    bar5 = fig.colorbar(sm, ax=ax5, extend='max')
    ax5.set_title(f'Alignment Score = {score: .4f}')
    ax5.set_xlabel('Position x (mm)')
    ax5.set_ylabel('Position y (mm)')
    bar5.set_label('Difference between images')
    # -------------------------------------------------------------------------------------------------------------------
    # Ax6: Plot of GafImage transformed
    # -------------------------------------------------------------------------------------------------------------------
    color_map6 = ax6.imshow(diff2, vmin=0, vmax=1,
                            extent=(0, np.shape(diff2)[1] * low_pixel_size, 0, np.shape(diff2)[0] * low_pixel_size))
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map6.cmap)
    sm.set_array([])
    bar6 = fig.colorbar(sm, ax=ax6, extend='max')
    ax6.set_xlabel('Position x (mm)')
    ax6.set_ylabel('Position y (mm)')
    ax6.set_title(f'Alignment Score = {score2: .4f}')
    bar6.set_label('Difference between images')

    plot_size = (latex_textwidth * 1.5, latex_textwidth * 2.5 / 1.2419)
    name = A.name + '_overview_'
    format_save(results_path / A.name, save_name=name, save=True, legend=False, fig=fig, plot_size=plot_size)
    # plt.show()

    # -------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------

    fig, ax = plt.subplots()
    color_map = ax.imshow(diff2, vmin=0, vmax=1,
                          extent=(0, np.shape(diff2)[1] * low_pixel_size, 0, np.shape(diff2)[0] * low_pixel_size))
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map.cmap)
    sm.set_array([])
    bar = fig.colorbar(sm, ax=ax, extend='max')
    ax.set_xlabel('Position x (mm)')
    ax.set_ylabel('Position y (mm)')
    ax.set_title(f'Alignment Score = {score2: .4f}')
    bar.set_label('Difference between images')
    name = A.name + '_DiffMap_'
    format_save(results_path / A.name, save_name=name, save=True, legend=False, fig=fig)

    # -------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------

    fig, ax = plt.subplots()
    color_map = ax.imshow(diff2, vmin=0, vmax=0.3,
                          extent=(0, np.shape(diff2)[1] * low_pixel_size, 0, np.shape(diff2)[0] * low_pixel_size))
    norm = matplotlib.colors.Normalize(vmin=0, vmax=0.3)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map.cmap)
    sm.set_array([])
    bar = fig.colorbar(sm, ax=ax, extend='max')
    ax.set_xlabel('Position x (mm)')
    ax.set_ylabel('Position y (mm)')
    ax.set_title(f'Alignment Score = {score2: .4f}')
    bar.set_label('Difference between images')
    name = A.name + '_DiffMapZoom_'
    format_save(results_path / A.name, save_name=name, save=True, legend=False, fig=fig)

    # -------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------

    fig, ax = plt.subplots()
    color_map = ax.imshow(diff, vmin=0, vmax=1,
                          extent=(0, np.shape(diff)[1] * low_pixel_size, 0, np.shape(diff)[0] * low_pixel_size))
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map.cmap)
    sm.set_array([])
    bar = fig.colorbar(sm, ax=ax, extend='max')
    ax.set_xlabel('Position x (mm)')
    ax.set_ylabel('Position y (mm)')
    ax.set_title(f'Alignment Score = {score: .4f}')
    bar.set_label('Difference between images')
    name = A.name + '_NotAligned_'
    format_save(results_path / A.name, save_name=name, save=True, legend=False, fig=fig)

    # -------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------

    fig, (ax1, ax2) = plt.subplots(1, 2)
    color_map = ax1.imshow(diff2, vmin=0, vmax=1,
                           extent=(0, np.shape(diff2)[1] * low_pixel_size, 0, np.shape(diff2)[0] * low_pixel_size))
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map.cmap)
    sm.set_array([])
    bar = fig.colorbar(sm, ax=ax1, extend='max')
    ax1.set_xlabel('Position x (mm)')
    ax1.set_ylabel('Position y (mm)')
    ax1.set_title(f'Alignment Score = {score2: .4f}')
    bar.set_label('Difference between images')

    ax2.hist(diff2.flatten(), bins=100)
    ax2.set_xlabel('Differences between normed maps')
    plot_size = (latex_textwidth * 3, latex_textwidth / 1.5 / 1.2419)
    name = A.name + '_HistogramNormed_'
    format_save(results_path / A.name, save_name=name, save=True, legend=False, fig=fig)




