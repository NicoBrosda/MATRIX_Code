from EvaluationSoftware.main import *
from EvaluationSoftware.readout_modules import ams_channel_assignment_readout
from EvaluationSoftware.normalization_modules import normalization_from_translated_array

mapping = Path('../Files/mapping.xlsx')
data = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:])-1 for k in data['direction_2']]
readout, position_parser = lambda x, y: ams_2line_readout(x, y, channel_assignment=channel_assignment), standard_position

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Results_221024/2LineMaps/M0/')

new_measurements = ['_GafComp200_', '_GafComp400_', '_GafComp40_', '_GafCompLogo_', '_GafCompMisc_', '_GafCompPEEK_', '_MouseFoot_', '_MouseFoot2_', '2Line_Beam_']
live_scan_array1 = [str(round(i+1, 0))+'_live1_' for i in range(9)]
# new_measurements_array_matrix = ['']

dark_path = Path('/Users/nico_brosda//Cyrce_Messungen/matrix_221024/')

dark_paths_array1 = ['2Line_DarkVoltageScan_200_ um_0_nA_nA_1.9_x_22.0_y_66.625.csv']

norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
norm_array1 = ['_YScan_']

for k, crit in enumerate(new_measurements[0:]):
    print('-'*50)
    print(crit)
    print('-'*50)

    A = Analyzer((1, 128), (0.4, 0.4), (0.1, 0.1), readout=readout)  # , diode_offset=[[0, 0.25], np.zeros(128)])
    A.dark = np.zeros((1, 64))
    A.norm_factor = np.ones((1, 64))
    dark = dark_paths_array1

    A.set_measurement(folder_path, crit)
    A.load_measurement()
    A.diode_dimension = (1, 64)
    A.create_map(inverse=[True, False])
    print(np.shape(A.maps[0]['z']))
    intensity_limits = None
    A.plot_map(results_path / 'raw/', pixel=True,
               intensity_limits=intensity_limits)
    A.plot_map(results_path / 'raw/', pixel='fill',
               intensity_limits=intensity_limits)
    A.plot_map(results_path / 'raw/', pixel=False,
               intensity_limits=intensity_limits)

    A.diode_dimension = (1, 128)
    A.set_dark_measurement(dark_path, dark)
    A.diode_dimension = (1, 64)
    A.update_measurement(factor=False)
    A.create_map(inverse=[True, False])
    intensity_limits = None
    A.plot_map(results_path / 'no_norm/', pixel=True,
               intensity_limits=intensity_limits)
    A.plot_map(results_path / 'no_norm/', pixel='fill',
               intensity_limits=intensity_limits)
    A.plot_map(results_path / 'no_norm/', pixel=False,
               intensity_limits=intensity_limits)

    continue
    norm = norm_array1
    A.normalization(norm_path, norm, normalization_module=normalization_from_translated_array_v2)
    A.update_measurement(dark=False)
    A.create_map(inverse=[True, False])
    intensity_limits = None
    A.plot_map(results_path / 'maps/', pixel=True,
               intensity_limits=intensity_limits)
    A.plot_map(results_path / 'maps/', pixel='fill',
               intensity_limits=intensity_limits)
    A.plot_map(results_path / 'maps/', pixel=False,
               intensity_limits=intensity_limits)
