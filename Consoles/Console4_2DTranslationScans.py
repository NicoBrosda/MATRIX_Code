import numpy as np

from EvaluationSoftware.main import *

# To get the mapping of the 2d array correct I have to translate the mapping of the contacts
# The given info by The-Duc are for the 1 direction of mapping, but we have by standard the other mapping
# Therefore, I need the correct channel assignment between these two to replace the channels in the 2D pixel placement

mapping = Path('../Files/mapping.xlsx')
direction1 = pd.read_excel(mapping, header=1)
direction1 = np.array([int(k[-3:]) for k in direction1['direction_1']])
direction2 = pd.read_excel(mapping, header=1)
direction2 = np.array([int(k[-3:]) for k in direction2['direction_2']])

mapping = Path('../Files/Mapping_MatrixArray.xlsx')
data2 = pd.read_excel(mapping, header=None)
mapping_map = data2.to_numpy().flatten()
print(mapping_map)
translated_mapping = np.array([direction2[np.argwhere(direction1 == i)[0][0]]-1 for i in mapping_map])
print(translated_mapping)

readout, position_parser = lambda x, y: ams_2D_assignment_readout(x, y, channel_assignment=translated_mapping), standard_position

A = Analyzer((11, 11), 0.8, 0.2, readout=readout)

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_111024/')
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Results_111024/MatrixArray/')

dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_111024/')
matrix_dark = ['2DLarge_dark_200_um_0_nA__nA_1.9_x_21.0_y_70.35.csv']

for i in range(0, 43):
    # crit = '2DLarge_YScan_200_um_2_nA__nA_1.9_x_{x_pos:.1f}_y_{y_pos:.2f}'.format(x_pos=21.0, y_pos=i+0.35)
    crit = '2DLarge_XScan_200_um_2_nA__nA_1.9_x_{x_pos:.1f}_y_{y_pos:.2f}'.format(x_pos=i, y_pos=70 + 0.35)
    files = os.listdir(folder_path)
    # for i in range(len(array_txt_file_search(files, [crit], txt_file=False, file_suffix='.csv'))):
        # print(i)
        # A.set_measurement(folder_path, '_'+str(i+1)+'_'+crit)
    A.set_measurement(folder_path, crit)
    A.set_dark_measurement(dark_path, matrix_dark)

    A.load_measurement()
    A.create_map(inverse=[True, False])
    intensity_limits = [0, 32000]
    # A.plot_map(results_path / 'raw/', pixel=True, intensity_limits=intensity_limits)
    A.plot_map(results_path / 'no_norm/', pixel='fill', intensity_limits=intensity_limits)
    # A.plot_map(results_path / 'no_norm/', pixel=False, intensity_limits=intensity_limits)
