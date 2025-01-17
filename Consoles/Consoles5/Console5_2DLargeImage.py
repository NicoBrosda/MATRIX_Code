import numpy as np

from EvaluationSoftware.main import *
from EvaluationSoftware.helper_modules import rename_files

# To get the mapping of the 2d array correct I have to translate the mapping of the contacts
# The given info by The-Duc are for the 1 direction of mapping, but we have by standard the other mapping
# Therefore, I need the correct channel assignment between these two to replace the channels in the 2D pixel placement

mapping = Path('../../Files/mapping.xlsx')
direction1 = pd.read_excel(mapping, header=1)
direction1 = np.array([int(k[-3:]) for k in direction1['direction_1']])
direction2 = pd.read_excel(mapping, header=1)
direction2 = np.array([int(k[-3:]) for k in direction2['direction_2']])

mapping = Path('../../Files/Mapping_BigMatrix_2.xlsx')
data2 = pd.read_excel(mapping, header=None)
mapping_map = data2.to_numpy().flatten()
translated_mapping = np.array([direction2[np.argwhere(direction1 == i)[0][0]]-1 for i in mapping_map])

readout, position_parser = lambda x, y: ams_2D_assignment_readout(x, y, channel_assignment=translated_mapping), standard_position

A = Analyzer((11, 11), 0.8, 0.2, readout=readout)

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Results_221024/2DLarge/')

dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
matrix_dark = ['2DLarge_DarkVoltage_200_ um_0_nA_nA_1.9_x_44.0_y_66.625.csv']
A.set_dark_measurement(dark_path, matrix_dark)

norm_path = folder_path
norm = ['2DLarge_YTranslation_']
norm_func = lambda list_of_files, instance, method='least_squares': normalization_from_translated_array_v3(
        list_of_files, instance, method, align_lines=True)
A.normalization(folder_path, ['2DLarge_YTranslation_'], normalization_module=norm_func)
A.normalization(norm_path, norm, normalization_module=norm_func)

measurements = ['2DLarge_YTranslation_']

# Line Scan image by image
results_path2 = results_path / 'YTranslation/'
A.set_measurement(folder_path, measurements[0])
measurement_files = np.array([str(i)[len(str(folder_path))+1:] for i in A.measurement_files], dtype=str)
measurement_files = measurement_files[np.argsort([standard_position(i)[1] for i in measurement_files])]

for file in tqdm(measurement_files[0:], colour='blue'):
    break
    A.set_measurement(folder_path, file)
    A.load_measurement()
    A.create_map(inverse=[True, False])
    intensity_limits = [0, np.max(A.maps[0]['z'])]
    A.plot_map(results_path2, pixel='fill', intensity_limits=intensity_limits)

A.set_measurement(folder_path, measurements)
A.load_measurement()
A.create_map(inverse=[True, False])
intensity_limits = [0, np.max(A.maps[0]['z'])]
A.plot_map(results_path, pixel='fill', intensity_limits=intensity_limits)