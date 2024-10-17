from EvaluationSoftware.main import *
from EvaluationSoftware.readout_modules import ams_channel_assignment_readout
from EvaluationSoftware.normalization_modules import normalization_from_translated_array

mapping = Path('../Files/mapping.xlsx')
data = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:])-1 for k in data['direction_2']]

readout, position_parser = lambda x, y: ams_channel_assignment_readout(x, y, channel_assignment=channel_assignment), standard_position

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_111024/')
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Results_111024/')

new_measurements_array1 = ['Start_QuickScan_', 'BeamScan1_', 'BeamScan02_', 'DiffBeamScan03_', 'DiffBeamScan04_', 'DiffBeamScan05_']
live_scan_array1 = [str(round(i+1, 0))+'_live1_' for i in range(9)]
# new_measurements_array_matrix = ['']

dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_230924/')
dark_path = Path('/Users/nico_brosda//Cyrce_Messungen/matrix_111024/')

dark_paths_array1 = ['Dark_QuickYScan_0_um_2_nA_.csv']
# dark_paths_array1 = ['voltage_scan_no_beam_nA_1.9000000000000006_x_20.0_y_70.0.csv']

dark_paths_array_matrix = ['Array3_VoltageScan_dark_nA_1.8_x_0.0_y_40.0.csv']

norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_230924/')
norm_array1 = ['Normalization2']
norm_array1 = ['uniformity_scan_']

norm_array3 = ['Array3_DiffuserYScan']

new_measurements = new_measurements_array1 + live_scan_array1
new_measurements = ['DiffBeamScan05_']
for k, crit in enumerate(new_measurements):
    print('-'*50)
    print(crit)
    print('-'*50)

    A = Analyzer((1, 128), 0.5, 0.0, readout=readout)

    # Correct sizing of the arrays
    A.diode_size = (0.5, 0.5)
    A.diode_size = (0.4, 0.4)
    A.diode_spacing = (0.1, 0.1)

    dark = dark_paths_array1

    A.set_measurement(folder_path, crit)
    A.load_measurement()
    A.create_map(inverse=[True, False])
    intensity_limits = None
    A.plot_map(results_path / 'raw/', pixel=True,
               intensity_limits=intensity_limits)
    A.plot_map(results_path / 'raw/', pixel='fill',
               intensity_limits=intensity_limits)
    A.plot_map(results_path / 'raw/', pixel=False,
               intensity_limits=intensity_limits)

    A.set_dark_measurement(dark_path, dark)
    A.update_measurement(factor=False)
    A.create_map(inverse=[True, False])
    intensity_limits = None
    A.plot_map(results_path / 'no_norm/', pixel=True,
               intensity_limits=intensity_limits)
    A.plot_map(results_path / 'no_norm/', pixel='fill',
               intensity_limits=intensity_limits)
    A.plot_map(results_path / 'no_norm/', pixel=False,
               intensity_limits=intensity_limits)


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
