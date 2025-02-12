from copy import deepcopy

from EvaluationSoftware.main import *
import time
import scipy as sp
from Concept8GafMeasurementComparison import GafImage

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black", "red", "yellow"])
cmap2 = sns.color_palette('viridis', as_cmap=True)
color_cycle = sns.color_palette("tab10")

mapping = Path('../../Files/mapping.xlsx')
data = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:]) - 1 for k in data['direction_2']]
readout, position_parser = lambda x, y: ams_2line_readout(x, y,
                                                          channel_assignment=channel_assignment), standard_position

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Results_221024/GafComp/')
run_name = 'GafMaxN_Bare'

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
    '''
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
    '''

    Test = GafImage(Gaf_path / Gaf_map[k])
    Test.load_image()
    if 'matrix211024_006.bmp' in Gaf_map[k]:
        Test.image = Test.image[::-1]
        Test.image = Test.image[:, ::-1]

    image_cache = deepcopy(Test.image)

    print('-'*50)

    # Part I: Bare max_n
    max_n = [1, 1e2, 1e4, 1e5, 1e6, 1e7]
    cache = []
    cache2 = []
    for n in max_n:
        Test.image = deepcopy(image_cache)
        low, high = Test.transform_to_normed(max_n=n)
        cache.append(Test.image)
        cache2.append([low, high])
        print('-' * 50)

    start = time.time()
    # Plot of images depending on basing
    fig, axes = plt.subplots(3, 2)
    for i, ax in enumerate(axes.flatten()):
        image = cache[i]
        n = max_n[i]
        color_map = ax.imshow(image, vmin=0, vmax=1, cmap=cmap,
                                extent=(0, np.shape(image)[1] * Test.pixel_size, 0, np.shape(image)[0] * Test.pixel_size))
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map.cmap)
        sm.set_array([])
        bar = fig.colorbar(sm, ax=ax, extend='max')
        ax.set_title(f'Points for response max mean {n:.1e}')
        ax.set_xlabel('Position x (mm)')
        ax.set_ylabel('Position y (mm)')
        bar.set_label('Gafchromic response (normed to 1)')
    print('Plot Images: ', time.time()-start)

    plot_size = (latex_textwidth * 1.5, latex_textwidth * 2.5 / 1.2419)
    name = run_name + '_GafImageComp_'
    format_save(results_path / A.name, save_name=name, save=True, legend=False, fig=fig, plot_size=plot_size)
    # plt.show()

    # ----------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------
    start = time.time()
    # Plot of image hist depending on image rebase
    plot_size = (latex_textwidth * 1.5, latex_textwidth * 2.5 / 1.2419)
    fig, axes = plt.subplots(3, 2, figsize=plot_size)
    fig.tight_layout()
    for i, ax in enumerate(axes.flatten()):
        image = cache[i]
        n = max_n[i]
        color_map = ax.hist(image.flatten(), bins=100, color='k')
        ax.axvline(0, color='red')
        ax.axvline(1, color='red')
        ax.set_xlim(-0.05, 1.1)
        ax.set_title(f'Points for response max mean {n:.1e}')
        if i >= 4:
            ax.set_xlabel('Gafchromic response (normed to 1)')

    plot_size = (latex_textwidth * 1.5, latex_textwidth * 2.5 / 1.2419)
    name = run_name + '_GafHistComp_'
    format_save(results_path / A.name, save_name=name, save=True, legend=False, fig=fig, plot_size=plot_size)
    # plt.show()
    print('Plot Hists: ', time.time()-start)

    # ----------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------
    start = time.time()
    # Plot of image hist depending on image rebase

    fig, ax = plt.subplots()

    image = image_cache
    for i, n in enumerate(max_n):
        ax.axvline(cache2[i][0]+cache2[i][1], color=color_cycle[i], label=f'Max n = {n:.1e}')
        ax.axvline(cache2[i][1], color=color_cycle[i])

    ax.hist(image.flatten(), bins=100, color='k')
    ax.set_xlabel('Gafchromic response (normed to 1)')

    name = run_name + '_GafHistBefore_'
    format_save(results_path / A.name, save_name=name, save=True, legend=True, fig=fig)
    # plt.show()
    print('Plot HistBefore: ', time.time() - start)
