import numpy as np
import scipy.signal

from EvaluationSoftware.main import *
from EvaluationSoftware.readout_modules import ams_channel_assignment_readout

mapping = Path('../Files/mapping.xlsx')
data = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:])-1 for k in data['direction_2']]

'''
channel_assignment = {}
j = 0
for i in range(1, 129):
    if i <= 32:
        channel_assignment[str(i)] = 63 - 2*j
    elif i <= 64:
        channel_assignment[str(i)] = 64 - 2*j
    elif i <= 96:
        channel_assignment[str(i)] = 65 + j
    elif i <= 128:
        channel_assignment[str(i)] = 127 - 2*j
    j += 1
    if i == 32 or i == 64 or i == 96:
        j = 0
# '''

# channel_assignment = [i*2+64 for i in range(32)]
print(channel_assignment)

readout, position_parser = lambda x, y, z=True: ams_channel_assignment_readout(x, y, z, channel_assignment=channel_assignment), standard_position

A = Analyzer((1, 128), 0.5, 0.0, readout=readout)


folder_path = Path('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/matrix_230924/')
new_measurements = ['round_aperture_2_3scans', 'Logo', 'scan_round_aperture_200um']


for k, crit in enumerate(new_measurements):
    # A.set_dark_measurement(folder_path, 'd2_1n_3s_beam_all_without_diffuser_dark.csv')
    # A.normalization(folder_path, '5s_flat_calib_', normalization_module=normalization_from_translated_array)
    A.set_measurement(folder_path, crit)
    # continue
    A.load_measurement(readout_module=readout)

    A.create_map(inverse=[False, False])
    intensity_limits = None

    A.plot_map('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/',
               contour=True, intensity_limits=intensity_limits)
    A.plot_map('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/',
               contour=False, intensity_limits=intensity_limits)