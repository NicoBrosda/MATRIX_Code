from EvaluationSoftware.standard_processes import *
from EvaluationSoftware.parameter_parsing_modules import *

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_211024/')
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Results_221024/')

crit = ['2D_Mini_VoltageLinearity']
dark_crit = ['2D_Mini_Dark_VoltageLinearity']

# Small matrix
mapping = Path('../../Files/mapping.xlsx')
direction1 = pd.read_excel(mapping, header=1)
direction1 = np.array([int(k[-3:]) for k in direction1['direction_1']])
direction2 = pd.read_excel(mapping, header=1)
direction2 = np.array([int(k[-3:]) for k in direction2['direction_2']])
mapping = Path('../../Files/Mapping_SmallMatrix1.xlsx')
data2 = pd.read_excel(mapping, header=None)
mapping_map = data2.to_numpy().flatten()
translated_mapping = np.array([direction2[np.argwhere(direction1 == i)[0][0]]-1 for i in mapping_map])
readout, position_parser, voltage_parser, current_parser = lambda x, y: ams_2D_assignment_readout(x, y, channel_assignment=translated_mapping), standard_position, standard_voltage, standard_current
A = Analyzer((11, 11), (0.4, 0.4), (0.1, 0.1), readout, position_parser, voltage_parser, current_parser)

linearity(folder_path, results_path, crit, dark_crit, A)


# Big matrix #2
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Results_221024/')
crit = ['2DLarge_LinearityMatrix_']
dark_crit = ['2DLarge_DarkVoltage_']
mapping = Path('../../Files/mapping.xlsx')
direction1 = pd.read_excel(mapping, header=1)
direction1 = np.array([int(k[-3:]) for k in direction1['direction_1']])
direction2 = pd.read_excel(mapping, header=1)
direction2 = np.array([int(k[-3:]) for k in direction2['direction_2']])
mapping = Path('../../Files/Mapping_BigMatrix_2.xlsx')
data2 = pd.read_excel(mapping, header=None)
mapping_map = data2.to_numpy().flatten()
translated_mapping = np.array([direction2[np.argwhere(direction1 == i)[0][0]]-1 for i in mapping_map])
readout, position_parser, voltage_parser, current_parser = lambda x, y: ams_2D_assignment_readout(x, y, channel_assignment=translated_mapping), standard_position, standard_voltage, standard_current
A = Analyzer((11, 11), (0.8, 0.8), (0.2, 0.2), readout, position_parser, voltage_parser, current_parser)

linearity(folder_path, results_path, crit, dark_crit, A)