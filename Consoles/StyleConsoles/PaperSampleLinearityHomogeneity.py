import matplotlib.pyplot as plt

from Consoles.StyleConsoles.Utils_ImageLoad import *
from PIL import Image
from EvaluationSoftware.standard_processes import linearity_return
from Consoles.Consoles8Gafchromic.Concept8GafCompTests import align_and_compare_images, resample_image, transform_image

# Save path and options for the map
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Style/Paper/')
background_subtraction = True
normalization = True

plot_size = (18  * cm, 7 / 1.2419 * cm)
save_format = '.svg'
dpi = 300
# Setup of the final figure
# Structure: Ax1 Logo Line Super Res - Ax2 Logo Gaf - Ax3 BeamShape 2-Line - Ax4 BeamShape Gaf -
# Ax5 Logo Overlay - Ax6 Beam Overlay
fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=plot_size)
# Adjust spacing: left, right, bottom, top, wspace, hspace
fig.subplots_adjust(wspace=0.3, hspace=0.25)
# Axis limits
y_limits = []
x_limits = []
zero_scale = True
intensity_limits = None
intensity_limitsg = [0, 1]

# ------------------------------------------------------------------------------------------------------------------
# Ax1: Sample structure / Bonded array
# ------------------------------------------------------------------------------------------------------------------
with Image.open('/Users/nico_brosda/Cyrce_Messungen/Style/Structure.png') as img:
    # img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # img = img.transpose(Image.FLIP_TOP_BOTTOM)
    # img = img.rotate(180)
    ax1.imshow(img)
ax1.axis('off')

# ------------------------------------------------------------------------------------------------------------------
# Ax2: Homogeneity
# ------------------------------------------------------------------------------------------------------------------
'''
mapping = Path('../../Files/mapping.xlsx')
data = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:])-1 for k in data['direction_2']]
readout, position_parser, voltage_parser, current_parser = (
    lambda x, y: ams_2line_readout(x, y, channel_assignment=channel_assignment),
    standard_position,
    standard_voltage,
    current5
)

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_260325/')
dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_260325/')
new_measurements = [f'exp10_']
dark_paths = ['exp1_dark_0nA_400um_nA_1.9_x_20.0_y_68.0', 'exp64_darkEnd_0.5nA_400um_nA_1.9_x_20.0_y_68.0',
                     '2exp66_Dark_0.0nA_0um_nA_1.9_x_20.0_y_68.0', '2exp138_DarkEnd_0nA_200um_nA_1.9_x_20.0_y_68.0']
def dark_voltage(voltage):
    return [f'exp1_dark_0nA_400um_nA_{voltage:.1f}_x_20.0_y_68.0',
            f'exp64_darkEnd_0.5nA_400um_nA_{voltage:.1f}_x_20.0_y_68.0',
            f'2exp66_Dark_0.0nA_0um_nA_{voltage:.1f}_x_20.0_y_68.0',
            f'2exp138_DarkEnd_0nA_200um_nA_{voltage:.1f}_x_20.0_y_68.0']

for i, measurement in enumerate(new_measurements):
    instance = Analyzer((2, 64), (0.4, 0.4), (0.1, 0.1), readout=readout,
                        diode_offset=[[0, - 0.25], np.zeros(64)], position_parser=position_parser,
                        voltage_parser=voltage_parser, current_parser=current_parser)

    instance.set_measurement(folder_path, [measurement])
    voltage = voltage_parser(instance.measurement_files[0])
    print(voltage)
    instance.set_dark_measurement(dark_path, dark_voltage(voltage))

    factor, diff = normalization_from_translated_array_v5(instance.measurement_files, instance, align_lines=True, remove_background=True)
'''
mapping = Path('../../Files/mapping.xlsx')
data = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:])-1 for k in data['direction_2']]
readout, position_parser, voltage_parser = (
    lambda x, y: ams_channel_assignment_readout(x, y, channel_assignment=channel_assignment),
    standard_position,
    standard_voltage
)
A = Analyzer((1, 128), 0.4, 0.1, readout=readout)
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_230924/')
dark_path = folder_path
dark_paths = ['voltage_scan_no_beam_nA_1.8000000000000005_x_20.0_y_70.0.csv',
                     'd2_1n_5s_flat_calib_nA_1.8000000000000007_x_20.0_y_70.0.csv']
dark_paths_array3_1V = ['Array3_VoltageScan_dark_nA_1.0_x_0.0_y_40.0.csv']
dark_paths_array3 = ['Array3_VoltageScan_dark_nA_1.8_x_0.0_y_40.0.csv']
measurements = ['uniformity_scan_', 'Normalization2']

factors = []
for i, measurement in enumerate(measurements):
    instance = Analyzer((1, 128), (0.4, 0.4), (0.1, 0.1), readout=readout,
                        position_parser=position_parser, voltage_parser=voltage_parser)

    instance.set_measurement(folder_path, [measurement])
    voltage = voltage_parser(instance.measurement_files[0])
    print(voltage)
    instance.set_dark_measurement(dark_path, dark_paths)
    factor, diff = normalization_from_translated_array_v5(instance.measurement_files, instance, align_lines=True, remove_background=True)
    factors.append(factor)

print(factor.flatten())
bin_size = 0.0025
data_min, data_max = 0.98, 1.07
bins = np.arange(start=data_min, stop=data_max + bin_size, step=bin_size)
color = sns.color_palette("hls", len(factors))
for i, factor in enumerate(factors):
    ax2.hist(factor.flatten(), bins=bins, edgecolor='k', color=color[i])
ax2.set_xlabel('Signal Homogeneity')
ax2.set_ylabel('Array diodes per homogeneity bin')

axins = ax2.inset_axes([0.5, 0.5, 0.47, 0.47])
for i, factor in enumerate(factors):
    axins.plot(factor.flatten(), color=color[i])
axins.set_xlabel('\\# Diode channel', fontsize=7, ha='center', va='center')
# axins.set_ylabel('Signal Homogeneity', fontsize=5, ha='center', va='center')
x1, x2, y1, y2 = 0, 128, 0.96, 1.04
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
fig.axes.append(axins)
axins.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
axins.yaxis.set_minor_locator(ticker.AutoMinorLocator())
axins.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
axins.xaxis.set_minor_locator(ticker.AutoMinorLocator())
# ------------------------------------------------------------------------------------------------------------------
# Ax3: Linearity
# ------------------------------------------------------------------------------------------------------------------
ax = ax3
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_irradiation_111224/')
crit = ['exp1_', 'exp2_', 'exp3_', 'exp4_', 'exp5_', 'exp6_']
dark_crit = ['exp8_']

mapping = Path('../../Files/mapping.xlsx')
data = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:])-1 for k in data['direction_2']]
readout, position_parser, voltage_parser, current_parser = lambda x, y: ams_channel_assignment_readout(x, y, channel_assignment=channel_assignment), standard_position, standard_voltage, lambda c: current3(c, conversion_factor=0.688)
A = Analyzer((1, 128), (0.4, 0.4), (0.1, 0.1), readout, position_parser, voltage_parser, current_parser)

currents, fit_currents, signal, fit, std, fit_std, fit_r2, std_r2 = linearity_return(folder_path, crit, dark_crit, A)

ax.plot(currents, signal, marker='x', color='k', label='Signal', ls='-')
ax.plot(fit_currents, fit, marker='', color='r', label='Linear fit', ls='--')

ax.plot(currents, std, marker='o', color='k', label='Signal Std', ls='-')
ax.plot(fit_currents, fit_std, marker='', color='b', label='Std sqrt fit', ls='--')

ax.set_xlabel('Proton current at Faraday cup (nA)')
ax.set_ylabel(f'Signal current ({scale_dict[instance.scale][1]}A)')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim())
ax.set_ylim(ax.get_ylim())
ax.text(currents[2], signal[1], r'Signal linear fit $\bar{\mathrm{R}}^2$' + ' = {x:.5f}'.format(x=fit_r2), fontsize=7,
        ha='left', va='top', c='r')
ax.text(currents[-1], std[-1], r'Std sqrt fit $\bar{\mathrm{R}}^2$' + ' = {x:.3f}'.format(x=std_r2), fontsize=7,
        ha='right', va='bottom', c='b')

ax.legend(loc='lower right')
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

for ax in fig.axes:
    if is_colorbar(ax):
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    elif ax==ax3:
        pass
    else:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

'''
add_diode_geometry_indicator(ax2, A2, position='upper right', fig=fig)
add_diode_geometry_indicator(ax3, A3, position='upper right', fig=fig)
add_diode_geometry_indicator(ax4, A4, position='upper right', fig=fig)
'''

ax1.text(*transform_axis_to_data_coordinates(ax1, [0.03, 0.97]), r'\textbf{(a)}', fontsize=7, ha='left',
        va='top', color='k')
ax2.text(*transform_axis_to_data_coordinates(ax2, [0.03, 0.97]), r'\textbf{(b)}', fontsize=7, ha='left',
        va='top', color='k')
ax3.text(*transform_axis_to_data_coordinates(ax3, [0.03, 0.97]), r'\textbf{(c)}', fontsize=7, ha='left',
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
just_save(save_path=results_path, save_name=f"HomogemeityLinearity", dpi=dpi, plot_size=plot_size, save_format=save_format, fig=fig, axes=[ax1, ax2, axins, ax3], legend=False)