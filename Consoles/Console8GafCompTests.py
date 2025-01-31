from Consoles.Concept8GafCompTests import func_calls
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
run_name = '_OptMethod_'

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

    print('-'*50)

    # Image down sampling (global to save time):
    low_pixel_size = A.maps[0]['x'][1] - A.maps[0]['x'][0]
    down_samp = resample_image(Test.image, Test.pixel_size, low_pixel_size)

    print('-'*50)

    start, start_score, addition = align_and_compare_images(A.maps[0]['z'], Test.image, A.maps[0]['x'][1] - A.maps[0]['x'][0],
                                           Test.pixel_size, image_down_sampled=down_samp, center_position=[0, 0],
                                           optimize_alignment=False)
    # Part I: Variation of penalty:
    '''
    rotation = [1, 2, 3, 4, 5, 6]
    results_grad = []
    results_evol = []

    for rot in rotation:
        diff, score = align_and_compare_images(A.maps[0]['z'], Test.image, A.maps[0]['x'][1] - A.maps[0]['x'][0],
                                               Test.pixel_size, image_down_sampled=down_samp,
                                               center_position=[0, 0],
                                               optimize_alignment=True, optimization_method='gradient',
                                               bounds=(-rot, rot))
        diff2, score2 = align_and_compare_images(A.maps[0]['z'], Test.image, low_pixel_size, Test.pixel_size,
                                                 image_down_sampled=down_samp, optimize_alignment=True,
                                                 optimization_method='evolutionary', bounds=(-rot, rot))
        results_grad.append([np.abs(diff), score])
        results_evol.append(([np.abs(diff2), score2]))


    fig, axes = plt.subplots(3, 2)
    for i, ax in enumerate(axes.flatten()):
        diff = results_grad[i][0]
        score = results_grad[i][1]
        color_map5 = ax.imshow(diff, vmin=0, vmax=1,
                                extent=(0, np.shape(diff)[1] * low_pixel_size, 0, np.shape(diff)[0] * low_pixel_size))
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map5.cmap)
        sm.set_array([])
        bar = fig.colorbar(sm, ax=ax, extend='max')
        ax.set_title(f'Max Rotation {rotation[i]}, Score {score: .4f}')
        ax.set_xlabel('Position x (mm)')
        ax.set_ylabel('Position y (mm)')
        bar.set_label('Difference between images')

    plot_size = (latex_textwidth * 1.5, latex_textwidth * 2.5 / 1.2419)
    name = A.name + run_name +'GridGrad_overview_'
    format_save(results_path / A.name, save_name=name, save=True, legend=False, fig=fig, plot_size=plot_size)
    # plt.show()

    # ----------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    fig, axes = plt.subplots(3, 2)
    for i, ax in enumerate(axes.flatten()):
        diff = results_evol[i][0]
        score = results_evol[i][1]
        color_map5 = ax.imshow(diff, vmin=0, vmax=1,
                               extent=(0, np.shape(diff)[1] * low_pixel_size, 0, np.shape(diff)[0] * low_pixel_size))
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map5.cmap)
        sm.set_array([])
        bar = fig.colorbar(sm, ax=ax, extend='max')
        ax.set_title(f'Max Rotation {rotation[i]}, Score {score: .4f}')
        ax.set_xlabel('Position x (mm)')
        ax.set_ylabel('Position y (mm)')
        bar.set_label('Difference between images')

    plot_size = (latex_textwidth * 1.5, latex_textwidth * 2.5 / 1.2419)
    name = A.name + run_name + 'Evol_overview_'
    format_save(results_path / A.name, save_name=name, save=True, legend=False, fig=fig, plot_size=plot_size)
    # plt.show()
    # '''

    maxiter = [50, 100, 200, 400, 1000, 5000]
    popsize = [1, 5, 10, 25, 100, 500]
    grad_n = [3, 5, 10, 15, 20, 30]
    results_gradn = []
    results_evol_max = []
    results_evol_popsize = []

    print('Evol Max N', ':'*50)

    for i in maxiter:
        diff, score, addition = align_and_compare_images(A.maps[0]['z'], Test.image, low_pixel_size, Test.pixel_size,
                                                 image_down_sampled=down_samp, optimize_alignment=True,
                                                 optimization_method='evolutionary', ev_max_iter=i)
        results_evol_max.append(([np.abs(diff), score, addition[2]]))
        print(f'Optimization {start_score:.3f} -> {score:.3f} with rotation {addition[0]} and center shift {addition[1]} consuming {addition[2]} metric function calls.')
        func_calls = 0

    print('Evol Popsize', ':'*50)

    for i in popsize:
        diff, score, addition = align_and_compare_images(A.maps[0]['z'], Test.image, low_pixel_size, Test.pixel_size,
                                                 image_down_sampled=down_samp, optimize_alignment=True,
                                                 optimization_method='evolutionary', ev_pop_size=i)
        results_evol_popsize.append(([np.abs(diff), score, addition[2]]))
        print(f'Optimization {start_score:.3f} -> {score:.3f} with rotation {addition[0]} and center shift {addition[1]} consuming {addition[2]} metric function calls.')
        func_calls = 0


    print('Grad max n', ':'*50)

    for i in grad_n:
        diff, score, addition = align_and_compare_images(A.maps[0]['z'], Test.image, low_pixel_size, Test.pixel_size,
                                                 image_down_sampled=down_samp, optimize_alignment=True,
                                                 optimization_method='gradient', grid_n=i)
        results_gradn.append(([np.abs(diff), score, addition[2]]))
        print(f'Optimization {start_score:.3f} -> {score:.3f} with rotation {addition[0]} and center shift {addition[1]} consuming {addition[2]} metric function calls.')
        func_calls = 0

    fig, axes = plt.subplots(3, 2)
    for i, ax in enumerate(axes.flatten()):
        diff = results_gradn[i][0]
        score = results_gradn[i][1]
        color_map5 = ax.imshow(diff, vmin=0, vmax=1,
                               extent=(0, np.shape(diff)[1] * low_pixel_size, 0, np.shape(diff)[0] * low_pixel_size))
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map5.cmap)
        sm.set_array([])
        bar = fig.colorbar(sm, ax=ax, extend='max')
        ax.set_title(f'{grad_n[i]} start points, FuncCalls {results_gradn[i][2]} \n Score {start_score:.7f} -to- {score: .7f}')
        ax.set_xlabel('Position x (mm)')
        ax.set_ylabel('Position y (mm)')
        bar.set_label('Difference between images')

    plot_size = (latex_textwidth * 1.5, latex_textwidth * 2.5 / 1.2419)
    name = A.name + run_name + 'GridGrad_overview_'
    format_save(results_path / A.name, save_name=name, save=True, legend=False, fig=fig, plot_size=plot_size)
    # plt.show()

    fig, axes = plt.subplots(3, 2)
    for i, ax in enumerate(axes.flatten()):
        diff = results_evol_max[i][0]
        score = results_evol_max[i][1]
        color_map5 = ax.imshow(diff, vmin=0, vmax=1,
                               extent=(0, np.shape(diff)[1] * low_pixel_size, 0, np.shape(diff)[0] * low_pixel_size))
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map5.cmap)
        sm.set_array([])
        bar = fig.colorbar(sm, ax=ax, extend='max')
        ax.set_title(
            f'{maxiter[i]} max iterations, FuncCalls {results_evol_max[i][2]} \n Score {start_score:.7f} -to- {score: .7f}')
        ax.set_xlabel('Position x (mm)')
        ax.set_ylabel('Position y (mm)')
        bar.set_label('Difference between images')

    plot_size = (latex_textwidth * 1.5, latex_textwidth * 2.5 / 1.2419)
    name = A.name + run_name + 'MaxiterEvol_overview_'
    format_save(results_path / A.name, save_name=name, save=True, legend=False, fig=fig, plot_size=plot_size)
    # plt.show()

    fig, axes = plt.subplots(3, 2)
    for i, ax in enumerate(axes.flatten()):
        diff = results_evol_popsize[i][0]
        score = results_evol_popsize[i][1]
        color_map5 = ax.imshow(diff, vmin=0, vmax=1,
                               extent=(0, np.shape(diff)[1] * low_pixel_size, 0, np.shape(diff)[0] * low_pixel_size))
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map5.cmap)
        sm.set_array([])
        bar = fig.colorbar(sm, ax=ax, extend='max')
        ax.set_title(
            f'{popsize[i]} Popsize, FuncCalls {results_evol_popsize[i][2]} \n Score {start_score:.7f} -to- {score: .7f}')
        ax.set_xlabel('Position x (mm)')
        ax.set_ylabel('Position y (mm)')
        bar.set_label('Difference between images')

    plot_size = (latex_textwidth * 1.5, latex_textwidth * 2.5 / 1.2419)
    name = A.name + run_name + 'PopsizeEvol_overview_'
    format_save(results_path / A.name, save_name=name, save=True, legend=False, fig=fig, plot_size=plot_size)
    # plt.show()