from Consoles.StyleConsoles.Utils_ImageLoad import *
from PIL import Image
from Consoles.Consoles8Gafchromic.Concept8GafCompTests import align_and_compare_images, resample_image, transform_image
from EvaluationSoftware.standard_processes import linearity_return, signal_comparison_channel

plt.rcParams["font.family"] = ["Arial"]

# Save path and options for the map
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Style/Paper/')

plot_size = (9.0*cm, 3*8.9*cm / 1.2419)
dpi = 300
save_format = '.png'

fig, [ax1, ax2, ax3] = plt.subplots(3, 1, figsize=plot_size)

# ------------------------------------------------------------------------------------------------------------------
# Ax1: Irradiation Graph
# ------------------------------------------------------------------------------------------------------------------
ax = ax1
ax1_twin = ax1.twinx()

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_irradiation_111224/')
measurements = ['25nA_11_12_2024_10_01_41.txt', '25nA_12_12_2024_15_10_11.txt']

for i, measure in enumerate(measurements):
    if i == 0:
        data_frame = pd.read_csv(folder_path / measure, sep=' ', header=None, skiprows=21, names=['Mean64', 'Mean1', 'Sigma64', 'Sigma1', 'Date'])
    else:
        data_frame = pd.concat([data_frame, pd.read_csv(folder_path / measure, sep=' ', header=None, skiprows=0, names=['Mean64', 'Mean1', 'Sigma64', 'Sigma1', 'Date'])], ignore_index=True)

time = data_frame['Date']

class TimeFormat:
    def __init__(self, param_dict):
        self.year = self.month = self.day = self.hour = self.minute = self.second = 0
        for param in param_dict:
            self.__dict__[param] = param_dict[param]

    def value(self):
        return (self.year*365*24*60*60 + self.month*30*24*60*60 + self.day*24*60*60 + self.hour*60*60 + \
            self.minute*60 + self.second)

    def __sub__(self, other):
        return np.abs(self.value() - other.value())

def time_parser(input_string, order=['day', 'month', 'year', 'hour', 'minute', 'second']):
    input_string = str(input_string)
    time_dict = {}
    for element in order[::-1]:
        try:
            pos = input_string.rindex('_')
        except ValueError:
            try:
                time_dict[element] = float(input_string[0:])
            except ValueError:
                time_dict[element] = 0
            break

        try:
            time_dict[element] = float(input_string[pos+1:])
        except ValueError:
            time_dict[element] = 0
        input_string = input_string[:pos]
    return TimeFormat(time_dict)

times = np.array([time_parser(data_frame['Date'][0]) - time_parser(i) for i in data_frame['Date']]) / 60 / 60

filter = ((data_frame['Mean64'] > 4e4) & (data_frame['Mean64'] < 1.3e5))
times, data = times[filter], data_frame['Mean64'][filter]
# ax.errorbar(times, data_frame['Mean64'], yerr=data_frame['Sigma64'])
A = Analyzer((1, 128), 0.4, 0.1)
A.scale = 'nano'
ax.plot(times[times < 7.7], A.signal_conversion(data[times < 7.7]), ls='', marker='.', color='k')
ax.plot(times[times > 22] - 22 + 7.7, A.signal_conversion(data[times > 22]), ls='', marker='.', color='k')

print(f'Maximal time is {np.max(times[times > 22] - 22 + 7.7)}')

ax1_twin.plot([0, np.max(times[times > 22] - 22 + 7.7)], [0, 2.44], c='r')
ax.set_xlim(ax.get_xlim()), ax.set_ylim(0, ax.get_ylim()[1])
ax1_twin.set_ylim(0, ax1_twin.get_ylim()[1]*1.1)

ax.set_xlabel('Irradiation time (h)', labelpad=1)
ax.set_ylabel(f'Signal current ({scale_dict[A.scale][1]}A)')
ax1_twin.set_ylabel('Cumulated dose (MGy)', color='red', va='center', labelpad=10)

ax1.text(*transform_axis_to_data_coordinates(ax1, [0.5, 0.96]), r'\textbf{2.4$\,$MGy total dose | 43.6$\,$Gy/s} ', fontsize=9, ha='center',
        va='top', color='r')

export_plot_data(ax, results_path / "Data_IrradiationHardness_LongTermDegradation.csv")
# ------------------------------------------------------------------------------------------------------------------
# Ax2: Linearity
# ------------------------------------------------------------------------------------------------------------------
ax = ax2

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

currents2, fit_currents2, signal2, fit2, std2, fit_std2, fit_r22, std_r22 = linearity_return(folder_path2, crit, dark_crit, B)

ax.plot(currents, signal, marker='x', color='b', label='Before irradiation', ls='')
ax.plot(fit_currents, fit, marker='', color='b', ls='--')
ax.plot(currents2, signal2, marker='x', color='r', label='After 3$\\,$mon', ls='')
ax.plot(fit_currents2, fit2, marker='', color='r', ls='--')
 
ax.set_xlabel('Proton current at Faraday cup (nA)', labelpad=2)
ax.set_ylabel(f'Signal current ({scale_dict[A.scale][1]}A)')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim())
ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*2)
ax.text(*transform_axis_to_data_coordinates(ax, [0.98, 0.11]), r'Before $\bar{\mathrm{R}}^2$' + ' = {x:.5f}'.format(x=fit_r2), fontsize=10,
        ha='right', va='bottom', c='b')
ax.text(*transform_axis_to_data_coordinates(ax, [0.98, 0.1]), r'After $\bar{\mathrm{R}}^2$' + ' = {x:.5f}'.format(x=fit_r22), fontsize=10,
        ha='right', va='top', c='r')

ax.legend(loc='upper left', fontsize=9)

export_plot_data(ax, results_path / "Data_IrradiationHardness_Linearity.csv")

# ------------------------------------------------------------------------------------------------------------------
# Ax3: Beam Shape
# ------------------------------------------------------------------------------------------------------------------
ax = ax3

# signal_comparison_channel(folder_path1, results_path, ['exp6_25nA_nA_1.9_x_0_y_0', 'exp12_25nA_12_12_2024_16_04_42'],'exp8_dark_without_cup_nA_1.9_x_0_y_0', A, names=['Before Irradiation', 'After irradiation'], normed=True, mark_n=[], add_plot=ax, save_plot=False, lw=1.5, color_list=['r', 'orange'])
signal_before = signal_comparison_channel(folder_path1, results_path, ['exp6_25nA_nA_1.9_x_0_y_0'],'exp8_dark_without_cup_nA_1.9_x_0_y_0', A, names=['Before'], normed=True, mark_n=[], add_plot=ax, save_plot=False, lw=1.5, color_list=['b'], pos_scale=[0.5, 0])

signal_after = signal_comparison_channel(folder_path2, results_path, ['exp3_nA_1.9_x_28.25_y_0.0'], 'exp2_nA_1.9_x_28.25_y_0.0', B, names=[r'After'], add_plot=ax, normed=True, lw=1.5, color_list=['r'], save_plot=False, pos_scale=[0.5, 0])

ax.plot(signal_after[0][-1], signal_after[0][1]/np.max(signal_before[0][1]), marker='', ls='--', c='r', alpha=1, label='Signal \n loss')

gradient_arrow(ax, (31, 1), ((31, 1.08*(np.max(signal_after[0][1])/np.max(signal_before[0][1])))), lw=2.5,
               cmap=matplotlib.colors.LinearSegmentedColormap.from_list("", ["b", "r"]), alpha=1, zorder=-1)
ax.legend(loc='lower right', fontsize=9)
ax.set_xlim((5, 69)), ax.set_ylim(ax.get_ylim())

x0, y0, x1, y1 = add_png_icon(ax, A, 'top left', zoom=0.2 , translation=None, background=True)
# ax.text(*transform_axis_to_data_coordinates(ax, [0.03, y0-0.05]), 'Diffused Beam Shape \n Only Relative Height \n Reproducable', fontsize=6, ha='left', va='top', c='k')

export_plot_data(ax, results_path / "Data_IrradiationHardness_BeamShape.csv")

# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

ax1.text(*transform_axis_to_data_coordinates(ax1, [0.97, 0.97]), r'\textbf{(a)}', fontsize=9, ha='right',
        va='top', color='k')
ax2.text(*transform_axis_to_data_coordinates(ax2, [0.97, 0.97]), r'\textbf{(b)}', fontsize=9, ha='right',
        va='top', color='k')
ax3.text(*transform_axis_to_data_coordinates(ax3, [0.97, 0.97]), r'\textbf{(c)}', fontsize=9, ha='right',
        va='top', color='k')

'''
ax1.text(*transform_axis_to_data_coordinates(ax1, [0.5, 0.97]), 'Phantom', fontsize=11, ha='center',
        va='top', color='k')
ax2.text(*transform_axis_to_data_coordinates(ax2, [0.5, 0.97]), '1x128 array', fontsize=11, ha='center',
        va='top', color='k')
ax3.text(*transform_axis_to_data_coordinates(ax3, [0.5, 0.97]), '1x128 array', fontsize=11, ha='center',
        va='top', color='k')
ax4.text(*transform_axis_to_data_coordinates(ax4, [0.5, 0.97]), '2x64 array \n 0.25$\\,$mm offset', fontsize=11, ha='center',
        va='top', color='k')
'''

for ax in fig.axes:
    if is_colorbar(ax):
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    else:
        if ax.get_xscale() == 'log':
            pass
        else:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

        if ax.get_yscale() == 'log':
            pass
        else:
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

print(plt.gcf().get_axes())

format_save(save_path=results_path, save_name=f"Graph2_IrradiationHardness", dpi=dpi, plot_size=plot_size, save_format=save_format, fig=fig, legend=False)