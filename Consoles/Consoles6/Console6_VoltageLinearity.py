from EvaluationSoftware.standard_processes import *
from EvaluationSoftware.parameter_parsing_modules import *

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_211124/')
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Results_221124/')

crit = ['_2DSmall_vscan_']
dark_crit = ['9_2DSmall_miscshape_xyscan_2.0_nA_nA_1.9_x_0.0_y_48.0.csv']

# Small matrix
mapping = Path('../../Files/mapping.xlsx')
direction1 = pd.read_excel(mapping, header=1)
direction1 = np.array([int(k[-3:]) for k in direction1['direction_1']])
direction2 = pd.read_excel(mapping, header=1)
direction2 = np.array([int(k[-3:]) for k in direction2['direction_2']])
mapping = Path('../../Files/Mapping_SmallMatrix2.xlsx')
data2 = pd.read_excel(mapping, header=None)
mapping_map = data2.to_numpy().flatten()
translated_mapping = np.array([direction2[np.argwhere(direction1 == i)[0][0]]-1 for i in mapping_map])
readout, position_parser, voltage_parser, current_parser = lambda x, y: ams_2D_assignment_readout(x, y, channel_assignment=translated_mapping), standard_position, standard_voltage, current2
A = Analyzer((11, 11), (0.4, 0.4), (0.1, 0.1), readout, position_parser, voltage_parser, current_parser)

linearity(folder_path, results_path, crit, dark_crit, A)
