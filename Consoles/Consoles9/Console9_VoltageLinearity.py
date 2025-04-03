from EvaluationSoftware.standard_processes import *
from EvaluationSoftware.parameter_parsing_modules import *

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_260325/')
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Results_260325/Checks/')

crit = ['exp2_', 'exp3_', 'exp4_', 'exp5_']
dark_crit = ['exp1_dark_']

# Small matrix
mapping = Path('../../Files/mapping.xlsx')
data = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:])-1 for k in data['direction_2']]
readout, position_parser, voltage_parser, current_parser = (
    lambda x, y: ams_2line_readout(x, y, channel_assignment=channel_assignment),
    standard_position,
    standard_voltage,
    current5
)
A = Analyzer((2, 64), (0.4, 0.4), (0.1, 0.1), readout=readout,
             diode_offset=[[0, - 0.25], np.zeros(64)], position_parser=position_parser,
             voltage_parser=voltage_parser, current_parser=current_parser)

linearity(folder_path, results_path, crit, dark_crit, A)


''' Dark signal comp for different voltages also
signal_comparison_channel(folder_path, results_path,
                  ['exp1_dark_0nA_400um_nA_1.1_x_20.0_y_68.0', 'exp64_darkEnd_0.5nA_400um_nA_1.1_x_20.0_y_68.0', '2exp67_Dark2_0.0nA_0um_nA_1.1_x_20.0_y_68.0', '2exp138_DarkEnd_0nA_200um_nA_1.1_x_20.0_y_68.0'],
                  None, A,
                          names=['Start day1', 'End day1', 'Start day2', 'End day2'],
                          save_plot=True)

signal_comparison_channel(folder_path, results_path,
                  ['exp1_dark_0nA_400um_nA_1.5_x_20.0_y_68.0', 'exp64_darkEnd_0.5nA_400um_nA_1.5_x_20.0_y_68.0', '2exp67_Dark2_0.0nA_0um_nA_1.5_x_20.0_y_68.0', '2exp138_DarkEnd_0nA_200um_nA_1.5_x_20.0_y_68.0'],
                  None, A,
                          names=['Start day1', 'End day1', 'Start day2', 'End day2'],
                          save_plot=True)

signal_comparison_channel(folder_path, results_path,
                  ['exp1_dark_0nA_400um_nA_1.9_x_20.0_y_68.0', 'exp64_darkEnd_0.5nA_400um_nA_1.9_x_20.0_y_68.0', '2exp67_Dark2_0.0nA_0um_nA_1.9_x_20.0_y_68.0', '2exp138_DarkEnd_0nA_200um_nA_1.9_x_20.0_y_68.0'],
                  None, A,
                          names=['Start day1', 'End day1', 'Start day2', 'End day2'],
                          save_plot=True)

signal_comparison_voltage(folder_path, results_path,
                          ['exp1_', 'exp64_', 'exp67_', 'exp138_'],
                          None, A,
                          names=['Start day1', 'End day1', 'Start day2', 'End day2'] )

# '''