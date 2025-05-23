from EvaluationSoftware.standard_processes import *
from EvaluationSoftware.parameter_parsing_modules import *

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_irradiation_111224/')
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Results_irradiation_111224/')

crit = ['exp1_', 'exp2_', 'exp3_', 'exp4_', 'exp5_', 'exp6_']
dark_crit = ['exp8_']

# Small matrix
mapping = Path('../../Files/mapping.xlsx')
data = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:])-1 for k in data['direction_2']]
print(channel_assignment)
readout, position_parser, voltage_parser, current_parser = lambda x, y: ams_channel_assignment_readout(x, y, channel_assignment=channel_assignment), standard_position, standard_voltage, lambda c: current3(c, conversion_factor=0.688)
A = Analyzer((1, 128), (0.4, 0.4), (0.1, 0.1), readout, position_parser, voltage_parser, current_parser)

standard_layout = lambda axes: standard_layout(axes, second_axis=True)
linearity(folder_path, results_path, crit, dark_crit, A)

# signal_comparison_voltage(folder_path, results_path, ['test_nA_', 'exp7_', 'exp8_'], None, A, names=['before linearity', 'after with cup', 'after without cup'], )

'''
signal_comparison_channel(folder_path, results_path, ['exp12_25nA_12_12_2024_15_16_36', 'exp12_25nA_12_12_2024_15_20_36', 'exp12_25nA_12_12_2024_15_24_37', 'exp12_25nA_12_12_2024_15_28_37', 'exp12_25nA_12_12_2024_15_32_38', 'exp12_25nA_12_12_2024_15_36_38', 'exp12_25nA_12_12_2024_15_40_39', 'exp12_25nA_12_12_2024_15_44_39', 'exp12_25nA_12_12_2024_15_48_40', 'exp12_25nA_12_12_2024_15_52_40', 'exp12_25nA_12_12_2024_15_56_41', 'exp12_25nA_12_12_2024_16_00_41', 'exp12_25nA_12_12_2024_16_04_42'],
                  'exp8_dark_without_cup_nA_1.9_x_0_y_0', A, names=['15:16', '+4 min', '+8 min', '+12 min', '+16 min', '+20 min', '+24 min', '+28 min', '+32 min', '+36 min', '+40 min', '+44 min', '+48 min', '+52 min', '+56 min', '+60 min'])

signal_comparison_channel(folder_path, results_path, ['exp6_25nA_nA_1.9_x_0_y_0', 'exp12_25nA_12_12_2024_15_16_36', 'exp12_25nA_12_12_2024_15_20_36', 'exp12_25nA_12_12_2024_15_24_37', 'exp12_25nA_12_12_2024_15_28_37', 'exp12_25nA_12_12_2024_15_32_38', 'exp12_25nA_12_12_2024_15_36_38', 'exp12_25nA_12_12_2024_15_40_39', 'exp12_25nA_12_12_2024_15_44_39', 'exp12_25nA_12_12_2024_15_48_40', 'exp12_25nA_12_12_2024_15_52_40', 'exp12_25nA_12_12_2024_15_56_41', 'exp12_25nA_12_12_2024_16_00_41', 'exp12_25nA_12_12_2024_16_04_42'],
                  'exp8_dark_without_cup_nA_1.9_x_0_y_0', A, names=['Day before', '15:16', '+4 min', '+8 min', '+12 min', '+16 min', '+20 min', '+24 min', '+28 min', '+32 min', '+36 min', '+40 min', '+44 min', '+48 min', '+52 min', '+56 min', '+60 min'], mark_n=[0])

signal_comparison_channel(folder_path, results_path, ['exp6_25nA_nA_1.9_x_0_y_0', 'exp12_25nA_12_12_2024_15_16_36', 'exp12_25nA_12_12_2024_15_20_36', 'exp12_25nA_12_12_2024_15_24_37', 'exp12_25nA_12_12_2024_15_28_37', 'exp12_25nA_12_12_2024_15_32_38', 'exp12_25nA_12_12_2024_15_36_38', 'exp12_25nA_12_12_2024_15_40_39', 'exp12_25nA_12_12_2024_15_44_39', 'exp12_25nA_12_12_2024_15_48_40', 'exp12_25nA_12_12_2024_15_52_40', 'exp12_25nA_12_12_2024_15_56_41', 'exp12_25nA_12_12_2024_16_00_41', 'exp12_25nA_12_12_2024_16_04_42'],
                  'exp8_dark_without_cup_nA_1.9_x_0_y_0', A, names=['Day before', '15:16', '+4 min', '+8 min', '+12 min', '+16 min', '+20 min', '+24 min', '+28 min', '+32 min', '+36 min', '+40 min', '+44 min', '+48 min', '+52 min', '+56 min', '+60 min'], normed=True, mark_n=[0])
# '''