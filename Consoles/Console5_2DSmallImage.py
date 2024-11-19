import numpy as np

from EvaluationSoftware.main import *
from EvaluationSoftware.helper_modules import rename_files

# To get the mapping of the 2d array correct I have to translate the mapping of the contacts
# The given info by The-Duc are for the 1 direction of mapping, but we have by standard the other mapping
# Therefore, I need the correct channel assignment between these two to replace the channels in the 2D pixel placement

mapping = Path('../Files/mapping.xlsx')
direction1 = pd.read_excel(mapping, header=1)
direction1 = np.array([int(k[-3:]) for k in direction1['direction_1']])
direction2 = pd.read_excel(mapping, header=1)
direction2 = np.array([int(k[-3:]) for k in direction2['direction_2']])

mapping = Path('../Files/Mapping_SmallMatrix1.xlsx')
data2 = pd.read_excel(mapping, header=None)
mapping_map = data2.to_numpy().flatten()
translated_mapping = np.array([direction2[np.argwhere(direction1 == i)[0][0]]-1 for i in mapping_map])

readout, position_parser = lambda x, y: ams_2D_assignment_readout(x, y, channel_assignment=translated_mapping), standard_position

A = Analyzer((11, 11), 0.8, 0.2, readout=readout)

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_211024/')
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Results_221024/2DSmall/')

dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_211024/')
matrix_dark = ['2D_Mini_Dark_VoltageLinearity_200_um_0_nA_nA_1.9_x_22.0_y_71.25.csv']
A.set_dark_measurement(dark_path, matrix_dark)

norm_path = folder_path
norm = '2D_Mini_YScan_'
norm_func = lambda list_of_files, instance, method='least_squares': normalization_from_translated_array_v3(
        list_of_files, instance, method, align_lines=True)
A.normalization(norm_path, norm, normalization_module=norm_func)

measurements = ['2D_Mini_XYScanMisc_200_um_2,0_nA_nA_1.9_']
measurements = ['2D_Mini_YScanAfter_200_ um_2,0_nA_nA_1.9_x_22.0_y_82.75.csv']
measurements = ['2D_Mini_FullXYScan_200_um_2,0_nA_nA_1.9']

for crit in measurements:
    A.set_measurement(folder_path, crit)
    A.load_measurement()
    A.create_map(inverse=[True, False])
    intensity_limits = [0, np.max(A.maps[0]['z'])]

    A.plot_map(results_path / 'maps/', pixel=True,
               intensity_limits=intensity_limits)
    A.plot_map(results_path / 'maps/', pixel='fill',
               intensity_limits=intensity_limits)
    A.plot_map(results_path / 'maps/', pixel=False,
               intensity_limits=intensity_limits)
    A.plot_map(results_path / 'maps/', pixel='fill',
               intensity_limits=intensity_limits, imshow=True)
    A.plot_map(results_path / 'maps/', pixel=True,
               intensity_limits=intensity_limits, imshow=True)

