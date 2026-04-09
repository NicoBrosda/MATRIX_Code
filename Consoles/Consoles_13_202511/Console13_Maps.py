from Consoles.Consoles9.Console9_EnergyExtrapolation import new_measurements
from EvaluationSoftware.main import *

mapping = Path('../../Files/mapping.xlsx')
data = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:])-1 for k in data['direction_2']]
readout, position_parser, voltage_parser, current_parser = (
    lambda x, y: ams_2line_fast_avg(x, y, channel_assignment=channel_assignment),
    standard_position,
    standard_voltage,
    current5
)
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_171225/')
results_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/ResultsPEEK_161225/')
# new_measurements = [f'exp{6}_', f'exp{7}_', f'exp{9}_', f'exp{10}_']
new_measurements = [f'exp{9}_', f'exp{10}_']

dark_path = folder_path

dark_paths_array1 = ['dark_current']

norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_260325/')
norm_array1 = ['exp7_norm1,9V_']
# norm_array1 = ['2exp76_normday2_P0_']
files = []
for k, crit in enumerate(new_measurements[0:]):
    print('-'*50)
    print(crit)
    print('-'*50)

    A = Analyzer((2, 64), (0.4, 0.4), (0.1, 0.1), readout=readout,
                 diode_offset=[[0, - 0.25], np.zeros(64)], position_parser=position_parser,
                 voltage_parser=voltage_parser, current_parser=current_parser)
    dark = dark_paths_array1
    A.set_measurement(folder_path, crit)
    files += A.measurement_files

A.measurement_files = files
A.name = 'full_map'
A.load_measurement()
cache_save_raw = deepcopy(A.measurement_data)
A.set_dark_measurement(dark_path, dark)
norm = norm_array1
norm_func = lambda list_of_files, instance, method='least_squares': normalization_from_translated_array_v3(
    list_of_files, instance, method, align_lines=True)
A.normalization(norm_path, norm, normalization_module=norm_func)
A.update_measurement()
A.create_map(inverse=[True, False])
intensity_limits = [0, np.max(A.maps[0]['z'])]
print(np.max(np.max(A.maps[0]['z'])))

A.maps[0] = overlap_treatment(A.maps[0], A, True)
# A.plot_map(results_path / 'maps/', pixel=True, intensity_limits=intensity_limits)
A.plot_map(results_path / 'maps/', pixel='fill', intensity_limits=intensity_limits, save_format='.png')

compensation = np.load(results_path / 'compensation_factor.npy')
correction_factor = np.load(results_path / 'correction_factor.npy')

def deinterleave_to_2x64(a):
    a = np.asarray(a)
    if a.shape[-1] != 128:
        raise ValueError(f"Expected last axis = 128, got {a.shape}")
    even = a[..., 0::2]  # g1
    odd  = a[..., 1::2]  # g0
    return np.stack((odd, even), axis=-2)

compensation = deinterleave_to_2x64(compensation)
correction_factor = deinterleave_to_2x64(correction_factor)

norm_factor = deepcopy(A.norm_factor)
A.norm_factor = norm_factor / compensation
A.name = 'full_map_compensated'
A.load_measurement()
A.create_map(inverse=[True, False])
intensity_limits = [0, np.max(A.maps[0]['z'])]
A.maps[0] = overlap_treatment(A.maps[0], A, True)
A.plot_map(results_path / 'maps/', pixel='fill', intensity_limits=intensity_limits, save_format='.png')

A.norm_factor = norm_factor / correction_factor
A.name = 'full_map_corrected'
A.load_measurement()
A.create_map(inverse=[True, False])
intensity_limits = [0, np.max(A.maps[0]['z'])]
A.maps[0] = overlap_treatment(A.maps[0], A, True)
A.plot_map(results_path / 'maps/', pixel='fill', intensity_limits=intensity_limits, save_format='.png')