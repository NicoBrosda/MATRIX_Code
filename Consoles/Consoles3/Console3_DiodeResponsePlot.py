import numpy as np
import scipy.signal

from EvaluationSoftware.main import *
from EvaluationSoftware.readout_modules import ams_channel_assignment_readout
from EvaluationSoftware.normalization_modules import normalization_from_translated_array

mapping = Path('../../Files/mapping.xlsx')
data = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:])-1 for k in data['direction_2']]

readout, position_parser = lambda x, y: ams_channel_assignment_readout(x, y, channel_assignment=channel_assignment), standard_position

A = Analyzer((1, 128), 0.5, 0.0, readout=readout)


folder_path = Path('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/matrix_230924/')

dark_paths_array1 = ['voltage_scan_no_beam_nA_1.8000000000000005_x_20.0_y_70.0.csv',
                     'd2_1n_5s_flat_calib_nA_1.8000000000000007_x_20.0_y_70.0.csv']

dark_paths_array3_1V = ['Array3_VoltageScan_dark_nA_1.0_x_0.0_y_40.0.csv']

dark_paths_array3 = ['Array3_VoltageScan_dark_nA_1.8_x_0.0_y_40.0.csv']

norm_array1 = ['Normalization2']
norm_array1 = ['uniformity_scan_']

y_scans = ['uniformity_scan_', 'BraggYScan']

norm_array3 = ['Array3_DiffuserYScan']

norm_arrays = norm_array1+norm_array3+y_scans
norm_arrays = norm_array3

for k, crit in enumerate(norm_arrays):
    print('-'*50)
    print(crit)
    print('-'*50)

    # Correct sizing of the arrays
    if 'Array3' in crit:
        A.diode_size = (0.25, 0.5)
        A.diode_size = (0.17, 0.4)
        A.diode_spacing = (0.08, 0.1)

    else:
        A.diode_size = (0.5, 0.5)
        A.diode_size = (0.4, 0.4)
        A.diode_spacing = (0.1, 0.1)

    # Filtering for correct files - Logo would be found in Array3_Logo...
    if crit == 'Logo':
        A.set_measurement(folder_path, crit, blacklist=['png', 'Array3'])
    else:
        A.set_measurement(folder_path, crit)

    # Dark Subtraction - correct file assignment
    if crit == 'Array3_Logo':
        dark = dark_paths_array3
    elif 'Array3' in crit:
        dark = dark_paths_array3_1V
    else:
        dark = dark_paths_array1

    A.set_dark_measurement(folder_path, dark)

    # A.plot_diodes('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/Norm/AfterNorm/', direction='y', plotting_range=None)

    if 'Array3' in crit:
        norm = norm_array3
    else:
        norm = norm_array1

    A.normalization(folder_path, norm, normalization_module=normalization_from_translated_array, blacklist=['.png', '_x_0.0_y'])

    A.load_measurement(readout_module=readout, position_parser=position_parser)

    A.plot_diodes('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/Norm/AfterNorm/', direction='y', plotting_range=None)
