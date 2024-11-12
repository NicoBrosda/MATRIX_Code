from EvaluationSoftware.main import *
from EvaluationSoftware.readout_modules import ams_channel_assignment_readout
from EvaluationSoftware.normalization_modules import normalization_from_translated_array

mapping = Path('../Files/mapping.xlsx')
data = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:])-1 for k in data['direction_2']]
readout, position_parser = lambda x, y: ams_2line_readout(x, y, channel_assignment=channel_assignment), standard_position

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Results_221024/2LineMaps/')

dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')

dark_paths_array1 = ['2Line_DarkVoltageScan_200_ um_0_nA_nA_1.9_x_22.0_y_66.625.csv']

norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
norm_array1 = ['2Line_YScan_']

A = Analyzer((2, 64), (0.4, 0.4), (0.1, 0.1), readout=readout, diode_offset=[[0, - 0.25], np.zeros(64)])
dark = dark_paths_array1
A.set_dark_measurement(dark_path, dark)

c_cyc = sns.color_palette("tab10")

# Plots of factors
fig, ax = plt.subplots()
for line in range(np.shape(A.dark)[0]):
    ax.plot(A.dark[line], c=c_cyc[line], label='Line '+str(line))
ax.set_xlabel(r'$\#$ Diode of each line')
ax.set_ylabel('Diode signal (a.u.)')
ax.set_xlim(ax.get_xlim())
ax.set_ylim(ax.get_ylim())
format_save(results_path, 'DarkData', legend=True, fig=fig,
            axes=[ax])
cache = deepcopy(A.dark[0, 27])
A.dark[0, 27] = 0
ax.set_ylim(0, np.max(A.dark))
A.dark[0, 27] = cache
format_save(results_path, 'DarkData_Zoom', legend=True, fig=fig,
            axes=[ax])