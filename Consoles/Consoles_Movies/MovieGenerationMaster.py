import numpy as np

from EvaluationSoftware.movie_modules import *

def quick_movie_wrap(folder_path):
    # Get the correct mapping for the matrix array
    mapping = Path('../../Files/mapping.xlsx')
    direction1 = pd.read_excel(mapping, header=1)
    direction1 = np.array([int(k[-3:]) for k in direction1['direction_1']])
    direction2 = pd.read_excel(mapping, header=1)
    direction2 = np.array([int(k[-3:]) for k in direction2['direction_2']])
    if '_111024' in folder_path.name:
        mapping = Path('../../Files/Mapping_MatrixArray.xlsx')
        dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
        matrix_dark = ['2DLarge_dark_200_um_0_nA__nA_1.9_x_21.0_y_70.35.csv']
        norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
        norm = ['2DLarge_YScan_']
    elif '_221024' in folder_path.name:
        mapping = Path('../../Files/Mapping_BigMatrix_2.xlsx')
        dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
        matrix_dark = ['2DLarge_DarkVoltage_200_ um_0_nA_nA_1.9_x_44.0_y_66.625.csv']
        norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
        norm = ['2DLarge_YTranslation_']
    elif '_161224' in folder_path.name:
        mapping = Path('../../Files/Mapping_BigMatrix_2.xlsx')
        dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_161224/')
        matrix_dark = ['exp16_dark_big_matrix_2_nA_1.8_x_28.25_y_0.0', 'exp18_dark_big_matrix_2_nA_1.8_x_28.25_y_0.0',
                     'exp18_dark_big_matrix_2_nA_1.1_x_28.25_y_0.0', 'exp18_dark_big_matrix_2_nA_1.1_x_28.25_y_0.0',
                     'exp18_dark_big_matrix_2_nA_1.8_x_28.25_y_0.0']
        norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
        norm = ['2DLarge_YTranslation_']

    data2 = pd.read_excel(mapping, header=None)
    mapping_map = data2.to_numpy().flatten()
    translated_mapping = np.array([direction2[np.argwhere(direction1 == i)[0][0]]-1 for i in mapping_map])

    # Define the Analyzer instance and assign dark current / normalization
    readout, position_parser = lambda x, y: ams_2D_assignment_readout(x, y, channel_assignment=translated_mapping), standard_position
    A = Analyzer((11, 11), 0.8, 0.2, readout=readout)
    A.set_dark_measurement(dark_path, matrix_dark)
    norm_func = lambda list_of_files, instance, method='least_squares': normalization_from_translated_array_v3(
        list_of_files, instance, method, align_lines=True)
    A.normalization(norm_path, norm, normalization_module=norm_func)
    if '161224' in folder_path.name:
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(A.norm_factor.T[::-1, :], cmap='viridis')

        # Annotate each pixel with its channel number
        for i in range(A.norm_factor.T[::-1, :].shape[0]):
            for j in range(A.norm_factor.T[::-1, :].shape[1]):
                ax.text(j, i, f"{A.norm_factor.T[::-1, :][i, j]:.2f}",
                        ha='center', va='center', color='white', fontsize=8)

        # Axis and layout
        ax.set_title("Diode → Measurement Channel Assignment", fontsize=14)
        ax.set_xlabel("X pixel index")
        ax.set_ylabel("Y pixel index")
        ax.set_xticks(np.arange(11))
        ax.set_yticks(np.arange(11))
        ax.set_xticklabels(np.arange(1, 12))
        ax.set_yticklabels(np.arange(1, 12))
        ax.invert_yaxis()  # Optional: if you want image-like coordinates

        # Add colorbar
        plt.colorbar(im, ax=ax, label='Channel number')

        plt.tight_layout()
        plt.show()
        # format_save(Path('../../Files/'), f'{mapping.name[:-5]}', save_format='.pdf')

        A.norm_factor[8, 3] = 1
        A.norm_factor[7, 3] = 1

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(A.norm_factor.T[::-1, :], cmap='viridis')

        # Annotate each pixel with its channel number
        for i in range(A.norm_factor.T[::-1, :].shape[0]):
            for j in range(A.norm_factor.T[::-1, :].shape[1]):
                ax.text(j, i, f"{A.norm_factor.T[::-1, :][i, j]:.2f}",
                        ha='center', va='center', color='white', fontsize=8)

        # Axis and layout
        ax.set_title("Diode → Measurement Channel Assignment", fontsize=14)
        ax.set_xlabel("X pixel index")
        ax.set_ylabel("Y pixel index")
        ax.set_xticks(np.arange(11))
        ax.set_yticks(np.arange(11))
        ax.set_xticklabels(np.arange(1, 12))
        ax.set_yticklabels(np.arange(1, 12))
        ax.invert_yaxis()  # Optional: if you want image-like coordinates

        # Add colorbar
        plt.colorbar(im, ax=ax, label='Channel number')

        plt.tight_layout()
        plt.show()
        # A.norm_factor = np.ones_like(A.norm_factor)
    readout = lambda x, y, channel_assignment=translated_mapping, keep_first_row=True, frame_bunch_size=1: (
        ams_2D_assignment_frame(x, y, channel_assignment=channel_assignment, keep_first_row=keep_first_row,
                                frame_bunch_size=frame_bunch_size))
    A.readout = readout

    return A


'''
# ---------------------------------------------------------------------------------------------------------------------
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_111024/')
analyzer = quick_movie_wrap(folder_path)
analyzer.scale = 'nano'
crit = '2DLarge_movieScan_'
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black", "red", "yellow"])
movie_res = 4
detector_status = True
breaks = 'proportional'
output_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/ResultsMovies/MatrixArray/')
return_info = False
frame_select = [1305, 1, 30]
zero_frame = 1305
length_scale = 0.3

generate_movie(analyzer, folder_path, crit, output_path, frame_select=frame_select, zero_frame=zero_frame, cmap=cmap,
               length_scale=length_scale, movie_res=movie_res, detector_status=detector_status, breaks=breaks,
               return_info=return_info)
# '''


'''
# ---------------------------------------------------------------------------------------------------------------------
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
crit = '2DLarge_MovieBeamChanges2_'
output_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/ResultsMovies/{folder_path.name}/{crit}/')
analyzer = quick_movie_wrap(folder_path)
analyzer.scale = 'nano'
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black", "red", "yellow"])
movie_res = 8
detector_status = True
breaks = 1
return_info = False
frame_select = [1050, 1, 20]
zero_frame = None
length_scale = 5

generate_movie(analyzer, folder_path, crit, output_path, frame_select=frame_select, zero_frame=zero_frame, cmap=cmap,
               length_scale=length_scale, movie_res=movie_res, detector_status=detector_status, breaks=breaks,
               return_info=return_info)
# '''

'''
# ---------------------------------------------------------------------------------------------------------------------
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
crit = '2DLarge_MovieCurrentIncrease_'
output_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/ResultsMovies/{folder_path.name}/{crit}/')
analyzer = quick_movie_wrap(folder_path)
analyzer.scale = 'nano'
cmap = plt.cm.magma
movie_res = 8
detector_status = False
breaks = 'proportional'
return_info = False
frame_select = None
zero_frame = None
length_scale = 1/30

generate_movie(analyzer, folder_path, crit, output_path, frame_select=frame_select, zero_frame=zero_frame, cmap=cmap,
               length_scale=length_scale, movie_res=movie_res, detector_status=detector_status, breaks=breaks,
               return_info=return_info)
# '''

'''
# ---------------------------------------------------------------------------------------------------------------------
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
crit = '2DLarge_MovieLogoMove_'
output_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/ResultsMovies/{folder_path.name}/{crit}/')
analyzer = quick_movie_wrap(folder_path)
analyzer.scale = 'nano'
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black", "red", "yellow"])
movie_res = 8
detector_status = False
breaks = 'proportional'
return_info = False
frame_select = None
zero_frame = None
length_scale = 1/30

generate_movie(analyzer, folder_path, crit, output_path, frame_select=frame_select, zero_frame=zero_frame, cmap=cmap,
               length_scale=length_scale, movie_res=movie_res, detector_status=detector_status, breaks=breaks,
               return_info=return_info)
# '''

# '''
# ---------------------------------------------------------------------------------------------------------------------
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_161224/')
for crit in ['exp17_', 'exp20_', 'exp21_', 'exp22_', 'exp23_']:
    if crit != 'exp23_':
        continue
    output_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/ResultsMovies/{folder_path.name}/{crit}/')
    analyzer = quick_movie_wrap(folder_path)
    analyzer.scale = 'nano'
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black", "red", "yellow"])
    movie_res = 8
    detector_status = False
    breaks = 'proportional'
    return_info = False
    frame_select = None
    zero_frame = None
    length_scale = 1/50

    generate_movie(analyzer, folder_path, crit, output_path, frame_select=frame_select, zero_frame=zero_frame, cmap=cmap,
                   length_scale=length_scale, movie_res=movie_res, detector_status=detector_status, breaks=breaks,
                   return_info=return_info)

    # '''