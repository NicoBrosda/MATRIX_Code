from Consoles.StyleConsoles.Utils_ImageLoad import *
from PIL import Image
from Consoles.Consoles8Gafchromic.Concept8GafCompTests import align_and_compare_images, resample_image, transform_image
from EvaluationSoftware.standard_processes import linearity_return, signal_comparison_channel

results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Style/Paper/')
fig, ax = plt.subplots()

folder_path1 = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_irradiation_111224/')
crit = ['exp1_', 'exp2_', 'exp3_', 'exp4_', 'exp5_', 'exp6_']
dark_crit = ['exp8_']

mapping = Path('../../Files/mapping.xlsx')
data = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:])-1 for k in data['direction_2']]
readout, position_parser, voltage_parser, current_parser = lambda x, y: ams_channel_assignment_readout(x, y, channel_assignment=channel_assignment), standard_position, standard_voltage, lambda c: current3(c, conversion_factor=0.688)
A = Analyzer((1, 128), (0.4, 0.4), (0.1, 0.1), readout, position_parser, voltage_parser, current_parser)
A.scale = 'nano'

standard_layout = lambda axes: standard_layout(axes, second_axis=True)
currents, fit_currents, signal, fit, std, fit_std, fit_r2, std_r2 = linearity_return(folder_path1, crit, dark_crit, A)

folder_path2 = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_130225/')
crit = ['exp3_', 'exp4_', 'exp5_', 'exp6_', 'exp7_', 'exp8_']
dark_crit = ['exp2_']
readout, position_parser, voltage_parser, current_parser = lambda x, y: ams_channel_assignment_readout(x, y, channel_assignment=channel_assignment), standard_position, standard_voltage, current4
B = Analyzer((1, 128), (0.4, 0.4), (0.1, 0.1), readout, position_parser, voltage_parser, current_parser)
B.scale = 'nano'

signal_comparison_channel(folder_path1, results_path,
                          ['exp6_25nA_nA_1.9_x_0_y_0', 'exp12_25nA_12_12_2024_15_16_36', 'exp12_25nA_12_12_2024_15_20_36', 'exp12_25nA_12_12_2024_15_24_37', 'exp12_25nA_12_12_2024_15_28_37', 'exp12_25nA_12_12_2024_15_32_38', 'exp12_25nA_12_12_2024_15_36_38', 'exp12_25nA_12_12_2024_15_40_39', 'exp12_25nA_12_12_2024_15_44_39', 'exp12_25nA_12_12_2024_15_48_40', 'exp12_25nA_12_12_2024_15_52_40', 'exp12_25nA_12_12_2024_15_56_41', 'exp12_25nA_12_12_2024_16_00_41', 'exp12_25nA_12_12_2024_16_04_42'],
                  'exp8_dark_without_cup_nA_1.9_x_0_y_0', A,
                          names=['Day before', '15:16', '+4 min', '+8 min', '+12 min', '+16 min', '+20 min', '+24 min', '+28 min', '+32 min', '+36 min', '+40 min', '+44 min', '+48 min', '+52 min', '+56 min', '+60 min'],
                          normed=False, mark_n=[0], save_plot=False, add_plot=ax)
signal_comparison_channel(folder_path2, results_path,
                          ['exp3_nA_1.9_x_28.25_y_0.0'], 'exp2_nA_1.9_x_28.25_y_0.0',
                          B, names=['Months later'], add_plot=ax, normed=False, save_plot=False)

export_plot_data(ax, results_path / "Data_NotNormed_IrradiationHardness_BeamShapeTimeVar.csv")
ax.set_yscale("symlog", linthresh=1e-3)
format_save(save_path=results_path, save_name=f"LogScaleBeamShape", dpi=300, save_format=save_format, fig=fig, legend=True)