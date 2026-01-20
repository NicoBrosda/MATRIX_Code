import numpy as np
import matplotlib.pyplot as plt
import numpy as np

from EvaluationSoftware.main import *
import cv2

def quick_norm_return(array_name):
    # Get the correct mapping for the matrix array
    mapping = Path('../../Files/mapping.xlsx')
    direction1 = pd.read_excel(mapping, header=1)
    direction1 = np.array([int(k[-3:]) for k in direction1['direction_1']])
    direction2 = pd.read_excel(mapping, header=1)
    direction2 = np.array([int(k[-3:]) for k in direction2['direction_2']])
    if 'BigMatrix' == array_name:
        mapping = Path('../../Files/Mapping_MatrixArray.xlsx')
        dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_111024/')
        matrix_dark = ['2DLarge_dark_200_um_0_nA__nA_1.9_x_21.0_y_70.35.csv']
        norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_111024/')
        norm = ['2DLarge_YScan_']
        data2 = pd.read_excel(mapping, header=None)
        mapping_map = data2.to_numpy().flatten()
        translated_mapping = np.array([direction2[np.argwhere(direction1 == i)[0][0]] - 1 for i in mapping_map])
        readout, position_parser = lambda x, y: ams_2D_assignment_readout(x, y, channel_assignment=translated_mapping), standard_position
        A = Analyzer((11, 11), 0.8, 0.2, readout=readout)
    elif 'BigMatrix2' == array_name:
        mapping = Path('../../Files/Mapping_BigMatrix_2.xlsx')
        dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
        matrix_dark = ['2DLarge_DarkVoltage_200_ um_0_nA_nA_1.9_x_44.0_y_66.625.csv']
        norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
        norm = ['2DLarge_YTranslation_']
        data2 = pd.read_excel(mapping, header=None)
        mapping_map = data2.to_numpy().flatten()
        translated_mapping = np.array([direction2[np.argwhere(direction1 == i)[0][0]] - 1 for i in mapping_map])
        readout, position_parser = lambda x, y: ams_2D_assignment_readout(x, y, channel_assignment=translated_mapping), standard_position
        A = Analyzer((11, 11), 0.8, 0.2, readout=readout)
    elif 'SmallMatrix' == array_name:
        mapping = Path('../../Files/Mapping_BigMatrix_2.xlsx')
        dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_211124/')
        matrix_dark = ['12_2DSmall_miscshape_']
        norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_211124/')
        norm = '8_2DSmall_yscan_'
        data2 = pd.read_excel(mapping, header=None)
        mapping_map = data2.to_numpy().flatten()
        translated_mapping = np.array([direction2[np.argwhere(direction1 == i)[0][0]] - 1 for i in mapping_map])
        readout, position_parser = lambda x, y: ams_2D_assignment_readout(x, y, channel_assignment=translated_mapping), standard_position
        A = Analyzer((11, 11), 0.4, 0.1, readout=readout)
    elif '2Line' == array_name:
        dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
        matrix_dark = ['2Line_DarkVoltageScan_200_ um_0_nA_nA_1.9_x_22.0_y_66.625.csv']
        norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
        norm = ['2Line_YScan_']
        readout = lambda x, y: ams_2line_readout(x, y, channel_assignment=direction2-1)
        A = Analyzer((2, 64), (0.4, 0.4), (0.1, 0.1), readout=readout,
                     diode_offset=[[0, - 0.25], np.zeros(64)], position_parser=standard_position)
    else:
        return None

    A.set_dark_measurement(dark_path, matrix_dark)
    norm_func = lambda list_of_files, instance, method='least_squares': normalization_from_translated_array_v5(
        list_of_files, instance, method, align_lines=True)
    A.normalization(norm_path, norm, normalization_module=norm_func)
    return A


for array_name in ['BigMatrix', 'BigMatrix2', 'SmallMatrix', '2Line']:
    print('-'*50)
    print(array_name)
    print('_'*50)
    norm_factor = quick_norm_return(array_name).norm_factor.T[::-1, :]
    print(norm_factor)

    # Plot
    cmap = sns.color_palette("flare_r", as_cmap=True)

    # Adjust figure size dynamically (bigger for longer arrays)
    ny, nx = norm_factor.shape
    aspect_ratio = nx / ny
    print(aspect_ratio)
    fig_width = min(max(4 * aspect_ratio, 6), 12)
    fig_height = min(max(4 / aspect_ratio, 4), 10)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    v_min = 0.95
    v_max = 1.05
    # Show the image
    if aspect_ratio != 1:
        aspect = 'auto'
    else:
        aspect = 'equal'
    im = ax.imshow(norm_factor, cmap=cmap, aspect=aspect, vmin=v_min, vmax=v_max)

    # Annotate only if it's not too large
    if ny * nx <= 150:  # avoid clutter for long 1×128 arrays
        for i in range(ny):
            for j in range(nx):
                ax.text(j, i, f"{norm_factor[i, j]:.3f}",
                        ha='center', va='center', color='white', fontsize=7)

    # Titles and labels
    ax.set_title(f"{array_name} Norm Factor", fontsize=14)
    # ax.set_xlabel("X pixel index")
    # ax.set_ylabel("Y pixel index")

    # Set tick marks dynamically
    ax.set_xticks(np.arange(nx))
    ax.set_yticks(np.arange(ny))
    ax.set_xticklabels(np.arange(1, nx + 1))
    ax.set_yticklabels(np.arange(1, ny + 1))

    # Reduce tick density for long arrays
    if nx > 20:
        ax.set_xticks(np.arange(0, nx, nx // 10))
    if ny > 20:
        ax.set_yticks(np.arange(0, ny, ny // 10))

    # Colorbar
    plt.colorbar(im, ax=ax, label='Normalization factor')

    # Keep "image-style" orientation
    ax.invert_yaxis()

    plt.tight_layout()
    format_save(Path('/Users/nico_brosda/Cyrce_Messungen/ResultsUV/NormFactors_v5/'),
                f'Norm_factor{array_name}_{v_min}_{v_max}', save_format='.pdf',
                plot_size=(fig_width, fig_height))
