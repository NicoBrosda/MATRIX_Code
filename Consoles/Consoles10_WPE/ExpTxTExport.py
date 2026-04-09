from EvaluationSoftware.main import *
from EvaluationSoftware.helper_modules import rename_files
import contextlib
import io

# To get the mapping of the 2d array correct I have to translate the mapping of the contacts
# The given info by The-Duc are for the 1 direction of mapping, but we have by standard the other mapping
# Therefore, I need the correct channel assignment between these two to replace the channels in the 2D pixel placement
mapping = Path('../../Files/mapping.xlsx')
direction1 = pd.read_excel(mapping, header=1)
direction1 = np.array([int(k[-3:]) for k in direction1['direction_1']])
direction2 = pd.read_excel(mapping, header=1)
direction2 = np.array([int(k[-3:]) for k in direction2['direction_2']])

mapping_small2 = Path('../../Files/Mapping_SmallMatrix2.xlsx')
mapping_big2 = Path('../../Files/Mapping_MatrixArray.xlsx')

data_small2 = pd.read_excel(mapping_small2, header=None)
mapping_map = data_small2.to_numpy().flatten()
translated_mapping_small2 = np.array([direction2[np.argwhere(direction1 == i)[0][0]] - 1 for i in mapping_map])

data_big = pd.read_excel(mapping_big2, header=None)
mapping_map = data_big.to_numpy().flatten()
translated_mapping_big = np.array([direction2[np.argwhere(direction1 == i)[0][0]] - 1 for i in mapping_map])

folder_path = Path('/Users/nico_brosda/Desktop/matrix_210525/Nouveau dossier/')
folder_image1 = Path('/Users/nico_brosda/Desktop/matrix_210525/exp52_eye_image/')
folder_image2 = Path('/Users/nico_brosda/Desktop/matrix_210525/exp53_eye_image/')
folder_image3 = Path('/Users/nico_brosda/Desktop/matrix_210525/exp56_phantom_9_steps/')
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/ResultsWPE_210525/')

dark_path = Path('/Users/nico_brosda/Desktop/matrix_210525/Nouveau dossier/')
dark_small = ['exp1_dark__nA_1.9_x_100.0_y_50.0.csv']
dark_big = ['exp55_nA_1.9_x_100.0_y_50.0.csv']

position_parser, voltage_parser, current_parser = WPE, standard_voltage, current3
readout_small = lambda x, y: ams_2D_assignment_readout_WPE(x, y, channel_assignment=translated_mapping_small2)
readout_big = lambda x, y: ams_2D_assignment_readout_WPE(x, y, channel_assignment=translated_mapping_big)

# A = Analyzer((11, 11), 0.4, 0.1, readout=readout, voltage_parser=voltage_parser, current_parser=current_parser)
# A.set_dark_measurement(dark_path, dark)

measurements = [f'exp{i}' for i in range(10, 55)] + [f'exp{i}' for i in range(57, 66)]

# Time structure
# '''
for k, crit in enumerate(measurements[0:]):
    if k <= len(range(10, 53)):
        readout = readout_small
        dark = dark_small
        mapping = translated_mapping_small2
    else:
        readout = readout_big
        dark = dark_big
        mapping = translated_mapping_big

    print('-' * 50)
    print(crit)
    print('-' * 50)
    param_cmap = sns.cubehelix_palette(as_cmap=True)
    param_cmap = sns.color_palette("Spectral", as_cmap=True)

    param_colormapper = lambda param: color_mapper(param, 0, 128)
    param_color = lambda param: param_cmap(param_colormapper(param))
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()

    A = Analyzer((11, 11), 0.4, 0.1, readout=readout, voltage_parser=voltage_parser, current_parser=current_parser,
                 position_parser=position_parser)
    A.set_dark_measurement(dark_path, dark)
    A.set_measurement(folder_path, crit)
    norm_func = lambda list_of_files, instance, method='least_squares': normalization_from_translated_array_v3(
        list_of_files, instance, method, align_lines=True)
    norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_211124/')
    norm = '8_2DSmall_yscan_'
    A.normalization(norm_path, norm, normalization_module=norm_func)

    try:
        FrameReadoutConfig.bunch_size = 100000
        cache = ams_2D_assignment_frame(A.measurement_files[0], instance=A, channel_assignment=translated_mapping_small2)
    except IndexError:
        continue

    analyzer = A
    print(A.scale)
    break
    print(np.shape(cache))
    cache = np.reshape(cache, (len(cache), 11, 11))
    cache -= analyzer.dark
    cache *= analyzer.norm_factor
    cache = analyzer.signal_conversion(cache)
    cache = np.reshape(cache, (len(cache), 121))
    np.savetxt(f"/Users/nico_brosda/Cyrce_Messungen/ResultsWPE_210525/ExpTxTs/{crit}.txt", cache, fmt="%.5e",)