import numpy as np
import scipy.signal

from EvaluationSoftware.main import *
from EvaluationSoftware.readout_modules import ams_channel_assignment_readout
from EvaluationSoftware.normalization_modules import normalization_from_translated_array

mapping = Path('../Files/mapping.xlsx')
data = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:])-1 for k in data['direction_2']]

readout, position_parser = lambda x, y, subtract_background=True: ams_channel_assignment_readout(x, y, subtract_background, channel_assignment=channel_assignment), standard_position

A = Analyzer((1, 128), 0.5, 0.0, readout=readout)


folder_path = Path('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/matrix_230924/')
# new_measurements = ['round_aperture_2_3scans', 'Logo', 'scan_round_aperture_200um']
# new_measurements = ['Array3_Logo', 'Array3_BeamShape', 'BraggPeak', 'MiscShape', 'round_aperture_2_3scans', 'Logo', 'scan_round_aperture_200um', 'BeamScan']
new_measurements = ['BeamScan']

dark_paths_array1 = ['voltage_scan_no_beam_nA_1.8000000000000005_x_20.0_y_70.0.csv',
                     'd2_1n_5s_flat_calib_nA_1.8000000000000007_x_20.0_y_70.0.csv']

dark_paths_array3_1V = ['Array3_VoltageScan_dark_nA_1.0_x_0.0_y_40.0.csv']

dark_paths_array3 = ['Array3_VoltageScan_dark_nA_1.8_x_0.0_y_40.0.csv']

norm_array1 = ['Normalization2']

norm_array3 = ['Array3_DiffuserYScan']

for k, crit in enumerate(new_measurements):
    print('-'*50)
    print(crit)
    print('-'*50)

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

    # Normalization - correct assignment
    if 'Array3' in crit:
        norm = norm_array3
    else:
        norm = norm_array1

    A.normalization(folder_path, norm, normalization_module=normalization_from_translated_array)

    A.load_measurement(readout_module=readout)

    A.create_map(inverse=[True, False])

    intensity_limits = None

    A.plot_map('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/',
               contour=True, intensity_limits=intensity_limits)
    A.plot_map('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/',
               contour=False, intensity_limits=intensity_limits)