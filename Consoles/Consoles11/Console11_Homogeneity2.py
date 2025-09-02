from copy import deepcopy

import numpy as np

from EvaluationSoftware.main import *
from EvaluationSoftware.readout_modules import ams_channel_assignment_readout
from EvaluationSoftware.normalization_modules import normalization_from_translated_array, normalization_from_translated_array_v4

mapping = Path('../../Files/mapping.xlsx')
data = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:])-1 for k in data['direction_2']]
readout, position_parser, voltage_parser, current_parser = (
    lambda x, y: ams_2line_readout(x, y, channel_assignment=channel_assignment),
    standard_position,
    standard_voltage,
    current5
)

# -----------------------------------------------------------------------------------------------------------------
# Latest y-scans with 2line array
# -----------------------------------------------------------------------------------------------------------------
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_260325/')
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Results_260325/Maps/')
new_measurements = []
new_measurements += [f'exp7_', f'exp8_', f'exp9_', f'exp10_', f'exp11_', f'exp12_', f'exp13_', f'exp33_', f'exp34',
                     f'exp35', f'exp76']
dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_260325/')
dark_paths = ['exp1_dark_0nA_400um_nA_1.9_x_20.0_y_68.0', 'exp64_darkEnd_0.5nA_400um_nA_1.9_x_20.0_y_68.0',
                     '2exp66_Dark_0.0nA_0um_nA_1.9_x_20.0_y_68.0', '2exp138_DarkEnd_0nA_200um_nA_1.9_x_20.0_y_68.0']

def dark_voltage(voltage):
    return [f'exp1_dark_0nA_400um_nA_{voltage:.1f}_x_20.0_y_68.0',
            f'exp64_darkEnd_0.5nA_400um_nA_{voltage:.1f}_x_20.0_y_68.0',
            f'2exp66_Dark_0.0nA_0um_nA_{voltage:.1f}_x_20.0_y_68.0',
            f'2exp138_DarkEnd_0nA_200um_nA_{voltage:.1f}_x_20.0_y_68.0']

line_2025 = len(new_measurements)

# -----------------------------------------------------------------------------------------------------------------
# Older norm of 2line array
# -----------------------------------------------------------------------------------------------------------------
new_measurements += ['2Line_YScan_']
folder_path2 = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
dark_path2 = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')

dark_paths2 = ['2Line_DarkVoltageScan_200_ um_0_nA_nA_1.9_x_22.0_y_66.625.csv']

norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
# -----------------------------------------------------------------------------------------------------------------

shortlabels = ['1,9V_22,7MeV_400um', '1,5V_22,7MeV_400um', '1,1V_22,7MeV_400um', '1,9V_17,5MeV_400um',
          '1,9V_12,7MeV_400um', '1,9V_7,4MeV_400um', '1,9V_2,9MeV_400um', '1,9V_2,9MeV_400um',
          '1,5V_2,9MeV_400um', '1,1V_2,9MeV_400um', '1,9V_23.7MeV_200um', '1,9V_23.7MeV_200um_5monthBefore']

voltages = [1.9, 1.5, 1.1, 1.9, 1.9, 1.9, 1.9, 1.9, 1.5, 1.1, 1.9, 1.9]
energies = [22.74516509213119, 22.74516509213119, 22.74516509213119, 17.518847982018325, 12.698754909667837,
            7.4441422659579235, 2.8730062373391827, 2.8730062373391827, 2.8730062373391827, 2.8730062373391827,
            23.69003946750678, 23.69003946750678]

labels = [f"{shortlabels[i]}_2Line_{new_measurements[i]}" for i in range(len(new_measurements))]


dark_crit = 'exp1_dark_'
list_of_crit = new_measurements
voltage_range = [0.8, 2.0]
factor_cache = []
factorna_cache = []

color = sns.color_palette("hls", len(shortlabels))

for i, measurement in enumerate(new_measurements):
    if 'exp13' not in measurement:
        pass
    c = color[i]

    results_path = Path("/Users/nico_brosda/Cyrce_Messungen/Results_260325/Homogeneity/Process/")

    instance = Analyzer((2, 64), (0.4, 0.4), (0.1, 0.1), readout=readout,
                        diode_offset=[[0, - 0.25], np.zeros(64)], position_parser=position_parser,
                        voltage_parser=voltage_parser, current_parser=current_parser)
    if i > line_2025-1:
        instance.set_measurement(folder_path2, [measurement])
        instance.set_dark_measurement(dark_path2, dark_paths2)
    else:
        instance.set_measurement(folder_path, [measurement])
        voltage = voltage_parser(instance.measurement_files[0])
        print(voltage)
        instance.set_dark_measurement(dark_path, dark_voltage(voltage))

    factor, diff = normalization_from_translated_array_v4(instance.measurement_files, instance, align_lines=True, label=labels[i], color=c, remove_background=True)
    factor_, diff_ = normalization_from_translated_array_v4(instance.measurement_files, instance, align_lines=True, label=labels[i], color=c, remove_background=False)

    factor2, diff2 = normalization_from_translated_array_v4(instance.measurement_files, instance, align_lines=False, label=labels[i], color=c, remove_background=True)
    factor2_, diff2_ = normalization_from_translated_array_v4(instance.measurement_files, instance, align_lines=False, label=labels[i], color=c, remove_background=False)

    factor_cache.append([factor, diff])
    factorna_cache.append([factor2, diff2])
    # --------------------------------------------------------------------------------------------------------------
    # Plotting
    # --------------------------------------------------------------------------------------------------------------

    def fast_yscale(ylim):
        y1, y2 = ax.get_ylim()
        if y1 < ylim[0]:
            y1 = ylim[0]
        if y2 > ylim[1]:
            y2 = ylim[1]
        ax.set_ylim(y1, y2)
        return y1, y2

    # --------------------------------------------------------------------------------------------------------------

    line_masks = []
    factor_limits = [0.9, 1.1]
    for line in range(np.shape(factor)[0]):
        mask = (factor_limits[0] < factor[line]) & (factor_limits[1] > factor[line])
        line_masks.append(mask)

    # --------------------------------------------------------------------------------------------------------------

    line_color = sns.color_palette("tab10")
    fig, ax = plt.subplots()
    for line in range(np.shape(factor)[0]):
        obj = ax.plot(factor[line], c=line_color[line])
        ax.fill_between(x=np.arange(0, len(factor[line]), 1), y1=factor[line]-diff[line], y2=factor[line]+diff[line],
                        color=obj[0].get_color(), alpha=0.3)
    ax.axvline(22, c='k', alpha=0.3)
    ax.set_xlim(ax.get_xlim()), fast_yscale([0.9, 1.1])
    text = labels[i]
    ax.text(*transform_axis_to_data_coordinates(ax, [0.03, 0.97]), text, fontsize=12, ha='left',
            va='top', color=c, bbox={'facecolor': 'white', 'edgecolor': c, 'alpha': 1.0, 'pad': 2, 'zorder': 10})
    ax.set_xlabel('Diode channel')
    ax.set_ylabel('Signal Homogeneity')
    format_save(results_path, f"5AfterProcess_BackgroundRemoved_Align_{labels[i]}", legend=False, fig=fig)

    # --------------------------------------------------------------------------------------------------------------

    fig, ax = plt.subplots()
    for line in range(np.shape(factor2)[0]):
        obj = ax.plot(factor2[line], c=line_color[line])
        ax.fill_between(x=np.arange(0, len(factor2[line]), 1), y1=factor2[line] - diff2[line],
                        y2=factor2[line] + diff2[line], color=obj[0].get_color(), alpha=0.3)
    ax.axvline(22, c='k', alpha=0.3)
    ax.set_xlim(ax.get_xlim()), fast_yscale([0.9, 1.1])
    text = labels[i]
    ax.text(*transform_axis_to_data_coordinates(ax, [0.03, 0.97]), text, fontsize=12, ha='left',
            va='top', color=c, bbox={'facecolor': 'white', 'edgecolor': c, 'alpha': 1.0, 'pad': 2, 'zorder': 10})
    ax.set_xlabel('Diode channel')
    ax.set_ylabel('Signal Homogeneity')
    format_save(results_path, f"6AfterProcess_BackgroundRemoved_NoAlign_{labels[i]}", legend=False, fig=fig)

    # --------------------------------------------------------------------------------------------------------------

    fig, ax = plt.subplots()
    for line in range(np.shape(factor)[0]):
        ax.plot(factor[line], c=line_color[line], label='Background removed')
        ax.plot(factor_[line], c=line_color[line], ls='--', label='With beam instabilities')
    ax.set_xlim(ax.get_xlim()), fast_yscale([0.9, 1.1])
    ax.set_xlabel('Diode channel')
    ax.set_ylabel('Signal Homogeneity')
    text = labels[i]
    ax.text(*transform_axis_to_data_coordinates(ax, [0.03, 0.97]), text, fontsize=12, ha='left',
            va='top', color=c, bbox={'facecolor': 'white', 'edgecolor': c, 'alpha': 1.0, 'pad': 2, 'zorder': 10})
    ax.legend(loc=4)
    format_save(results_path, f"7AfterProcess_CompBackground_Align_{labels[i]}", legend=False, fig=fig)

    # --------------------------------------------------------------------------------------------------------------

    fig, ax = plt.subplots()
    for line in range(np.shape(factor)[0]):
        ax.plot(factor2[line], c=line_color[line], label='Background removed')
        ax.plot(factor2_[line], c=line_color[line], ls='--', label='With beam instabilities')
    ax.set_xlim(ax.get_xlim()), fast_yscale([0.9, 1.1])
    ax.set_xlabel('Diode channel')
    ax.set_ylabel('Signal Homogeneity')
    text = labels[i]
    ax.text(*transform_axis_to_data_coordinates(ax, [0.03, 0.97]), text, fontsize=12, ha='left',
            va='top', color=c, bbox={'facecolor': 'white', 'edgecolor': c, 'alpha': 1.0, 'pad': 2, 'zorder': 10})
    ax.legend(loc=4)
    format_save(results_path, f"8AfterProcess_CompBackground_NoAlign_{labels[i]}", legend=False, fig=fig)

    # --------------------------------------------------------------------------------------------------------------

    fig, ax = plt.subplots()
    for line in range(np.shape(factor)[0]):
        ax.plot(factor[line], c=line_color[line], label='Aligned')
        ax.axhline(np.mean(factor[line][line_masks[line]]), ls='-', c=line_color[line], alpha=0.6)
        ax.plot(factor2[line], c=line_color[line], ls='--', label='Not aligned')
        ax.axhline(np.mean(factor2[line][line_masks[line]]), ls='--', c=line_color[line], alpha=0.6)
    ax.set_xlim(ax.get_xlim()), fast_yscale([0.9, 1.1])
    ax.set_xlabel('Diode channel')
    ax.set_ylabel('Signal Homogeneity')
    text = labels[i]
    ax.text(*transform_axis_to_data_coordinates(ax, [0.03, 0.97]), text, fontsize=12, ha='left',
            va='top', color=c, bbox={'facecolor': 'white', 'edgecolor': c, 'alpha': 1.0, 'pad': 2, 'zorder': 10})
    ax.legend(loc=4)
    format_save(results_path, f"9AfterProcess_CompAlign_BackgroundRemoved_{labels[i]}", legend=False, fig=fig)

    # --------------------------------------------------------------------------------------------------------------

    fig, ax = plt.subplots()
    for line in range(np.shape(factor)[0]):
        ax.plot(factor_[line], c=line_color[line], label='Aligned')
        ax.axhline(np.mean(factor_[line][line_masks[line]]), ls='-', c=line_color[line], alpha=0.6)

        ax.plot(factor2_[line], c=line_color[line], ls='--', label='Not aligned')
        ax.axhline(np.mean(factor2_[line][line_masks[line]]), ls='--', c=line_color[line], alpha=0.6)
    ax.set_xlim(ax.get_xlim()), fast_yscale([0.9, 1.1])
    ax.set_xlabel('Diode channel')
    ax.set_ylabel('Signal Homogeneity')
    text = labels[i]
    ax.text(*transform_axis_to_data_coordinates(ax, [0.03, 0.97]), text, fontsize=12, ha='left',
            va='top', color=c, bbox={'facecolor': 'white', 'edgecolor': c, 'alpha': 1.0, 'pad': 2, 'zorder': 10})
    ax.legend(loc=4)
    format_save(results_path, f"10AfterProcess_CompAlign_NotRemoved_{labels[i]}", legend=False, fig=fig)

    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------
# Factor Comp
# --------------------------------------------------------------------------------------------------------------
color = sns.color_palette("hls", len(shortlabels))

fig, ax = plt.subplots()
for i, obj in enumerate(factor_cache):
    ax.plot(np.concatenate((obj[0][0], obj[0][1]), axis=0), label=labels[i], color=color[i])
ax.set_xlim(ax.get_xlim()), fast_yscale([0.95, 1.05])
ax.set_xlabel('Diode channel')
ax.set_ylabel('Signal Homogeneity')
ax.set_xlim(ax.get_xlim()), ax.set_ylim(ax.get_ylim())
text = "Comparison of aligned \n normalization factors"
ax.text(*transform_axis_to_data_coordinates(ax, [0.03, 0.97]), text, fontsize=12, ha='left',
        va='top', color='k', bbox={'facecolor': 'white', 'edgecolor': 'k', 'alpha': 1.0, 'pad': 2, 'zorder': 10})
format_save(results_path, f"11AlignFactorComp", legend=False, fig=fig)

fig, ax = plt.subplots()
for i, obj in enumerate(factorna_cache):
    ax.plot(np.concatenate((obj[0][0], obj[0][1]), axis=0), label=labels[i], color=color[i])
ax.set_xlim(ax.get_xlim()), fast_yscale([0.95, 1.05])
ax.set_xlabel('Diode channel')
ax.set_ylabel('Signal Homogeneity')
ax.set_xlim(ax.get_xlim()), ax.set_ylim(ax.get_ylim())
text = "Comparison of not aligned \n normalization factors"
ax.text(*transform_axis_to_data_coordinates(ax, [0.03, 0.97]), text, fontsize=12, ha='left',
        va='top', color='k', bbox={'facecolor': 'white', 'edgecolor': 'k', 'alpha': 1.0, 'pad': 2, 'zorder': 10})
format_save(results_path, f"11NoAlignFactorComp", legend=False, fig=fig)

fig, ax = plt.subplots()
ax.plot(np.concatenate((factorna_cache[-1][0][0], factorna_cache[-1][0][1]), axis=0), label=labels[-1], color='r')
ax.plot(np.concatenate((factorna_cache[-2][0][0], factorna_cache[-2][0][1]), axis=0), label=labels[-2], color='b')
ax.plot(np.concatenate((factor_cache[-1][0][0], factor_cache[-1][0][1]), axis=0), color='r', ls='--', alpha=0.5, zorder=-1)
ax.plot(np.concatenate((factor_cache[-2][0][0], factor_cache[-2][0][1]), axis=0), color='b', ls='--', alpha=0.5, zorder=-1)
ax.set_xlim(ax.get_xlim()), fast_yscale([0.975, 1.025])
ax.set_xlabel('Diode channel')
ax.set_ylabel('Signal Homogeneity')
ax.set_xlim(ax.get_xlim()), ax.set_ylim(ax.get_ylim())
text = "Long term normalization deviation"
ax.text(*transform_axis_to_data_coordinates(ax, [0.03, 0.97]), text, fontsize=12, ha='left',
        va='top', color='k', bbox={'facecolor': 'white', 'edgecolor': 'k', 'alpha': 1.0, 'pad': 2, 'zorder': 10})
ax.legend(loc=4)
format_save(results_path, f"11LongTimeComp", legend=False, fig=fig)

# --------------------------------------------------------------------------------------------------------------
# Hist
# --------------------------------------------------------------------------------------------------------------
align_std = []
na_std = []
diffs = []
for i, obj in enumerate(factor_cache):
    c = color[i]
    fig, ax = plt.subplots()
    data = np.append(obj[0][0][line_masks[0]], obj[0][1][line_masks[1]])
    bin_size = 0.0025
    data_min, data_max = 0.95, 1.05
    bins = np.arange(start=data_min, stop=data_max + bin_size, step=bin_size)
    ax.hist(data, bins=bins, edgecolor='k', color=c)
    ax.set_xlim(ax.get_xlim()), ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*1.2)
    text = (f"Line Align - {shortlabels[i]} \n"
            f"Std of response homogeneity {np.std(data)*100:.2f}$\\,$\\% \n"
            f"Average difference from mean signal {np.mean(obj[1])*100:.2f}±{np.std(obj[1])*100:.2f}$\\,$\\%")
    ax.text(*transform_axis_to_data_coordinates(ax, [0.03, 0.97]), text, fontsize=12, ha='left',
            va='top', color=c, bbox={'facecolor': 'white', 'edgecolor': c, 'alpha': 0.7, 'pad': 2, 'zorder': 10})
    ax.set_xlabel('Signal Homogeneity')
    ax.set_ylabel('Counts per homogeneity bin')
    format_save(results_path, f"12HistAlign_{labels[i]}", legend=False, fig=fig)

    align_std.append(np.std(data))
    diffs.append(np.mean(obj[1]))

    fig, ax = plt.subplots()
    data = np.append(factorna_cache[i][0][0][line_masks[0]], factorna_cache[i][0][1][line_masks[1]])
    bin_size = 0.0025
    data_min, data_max = 0.95, 1.05
    bins = np.arange(start=data_min, stop=data_max + bin_size, step=bin_size)
    ax.hist(data, bins=bins, edgecolor='k', color=c)
    ax.set_xlim(ax.get_xlim()), ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*1.2)
    text = (f"No Line Align - {shortlabels[i]} \n"
            f"Std of response homogeneity {np.std(data) * 100:.2f}$\\,$\\% \n"
            f"Average difference from mean signal {np.mean(obj[1]) * 100:.2f}±{np.std(obj[1]) * 100:.2f}$\\,$\\%")
    ax.text(*transform_axis_to_data_coordinates(ax, [0.03, 0.97]), text, fontsize=12, ha='left',
            va='top', color=c, bbox={'facecolor': 'white', 'edgecolor': c, 'alpha': 0.7, 'pad': 2, 'zorder': 10})
    ax.set_xlabel('Signal Homogeneity')
    ax.set_ylabel('Counts per homogeneity bin')
    format_save(results_path, f"13HistNoAlign_{labels[i]}", legend=False, fig=fig)

    na_std.append(np.std(data))

fig, ax = plt.subplots()
for i, obj in enumerate(factor_cache):
    c = color[i]
    data = np.append(obj[0][0][line_masks[0]], obj[0][1][line_masks[1]])
    bin_size = 0.0025
    data_min, data_max = 0.95, 1.05
    bins = np.arange(start=data_min, stop=data_max + bin_size, step=bin_size)
    ax.hist(data, bins=bins, color=c, alpha=0.3)
ax.set_xlabel('Signal Homogeneity')
ax.set_ylabel('Counts per homogeneity bin')
ax.set_xlim(ax.get_xlim()), ax.set_ylim(ax.get_ylim())
text = "Aligned homogeneity \n histograms"
ax.text(*transform_axis_to_data_coordinates(ax, [0.03, 0.97]), text, fontsize=12, ha='left',
        va='top', color='k', bbox={'facecolor': 'white', 'edgecolor': 'k', 'alpha': 1.0, 'pad': 2, 'zorder': 10})
format_save(results_path, f"12HistAlignComp", legend=False, fig=fig)

fig, ax = plt.subplots()
for i, obj in enumerate(factorna_cache):
    c = color[i]
    data = np.append(obj[0][0][line_masks[0]], obj[0][1][line_masks[1]])
    bin_size = 0.0025
    data_min, data_max = 0.95, 1.05
    bins = np.arange(start=data_min, stop=data_max + bin_size, step=bin_size)
    ax.hist(data, bins=bins, color=c, alpha=0.3)
ax.set_xlabel('Signal Homogeneity')
ax.set_ylabel('Counts per homogeneity bin')
ax.set_xlim(ax.get_xlim()), ax.set_ylim(ax.get_ylim())
text = "Not aligned homogeneity \n histograms"
ax.text(*transform_axis_to_data_coordinates(ax, [0.03, 0.97]), text, fontsize=12, ha='left',
        va='top', color='k', bbox={'facecolor': 'white', 'edgecolor': 'k', 'alpha': 1.0, 'pad': 2, 'zorder': 10})
format_save(results_path, f"13HistNoAlignComp", legend=False, fig=fig)

# --------------------------------------------------------------------------------------------------------------
# Factor Comp by voltage - energy - time
# --------------------------------------------------------------------------------------------------------------
param_cmapv = sns.color_palette("flare_r", as_cmap=True)
param_colormapperv = lambda param: color_mapper(param, np.min(voltages), np.max(voltages))
param_colorv = lambda param: param_cmapv(param_colormapperv(param))

param_cmap = sns.color_palette("crest_r", as_cmap=True)
param_colormapper = lambda param: color_mapper(param, np.min(energies), np.max(energies))
param_color = lambda param: param_cmap(param_colormapper(param))

fig, ax = plt.subplots()
for i, obj in enumerate(factor_cache):
    if i == 0 or i >= len(factor_cache)-1:
        continue
    ax.plot(np.concatenate((obj[0][0], obj[0][1]), axis=0), label=labels[i], color=param_colorv(voltages[i]))
ax.set_xlim(ax.get_xlim()), fast_yscale([0.95, 1.05])
ax.set_xlabel('Diode channel')
ax.set_ylabel('Signal Homogeneity')
improved_gradient_scale(voltages, param_cmapv, ax_in=ax, param_unit='$\\,$V', point=[0.85, 0.94],
                        param_mapper=param_colormapperv, param_format='.1f')
format_save(results_path, f"14VoltageComp", legend=False, fig=fig)

fig, ax = plt.subplots()
for i, obj in enumerate(factor_cache):
    if i == 0 or i >= len(factor_cache)-1:
        continue
    y = (np.mean(obj[0][0])-np.mean(obj[0][1])) * 100
    ax.plot(voltages[i], y, label=labels[i], color=param_color(energies[i]), marker='x')
ax.set_xlim(ax.get_xlim()), ax.set_ylim(ax.get_ylim())
ax.set_xlabel('ams voltage (V)')
ax.set_ylabel('Line signal offset (\\%)')
improved_gradient_scale(energies, param_cmap, ax_in=ax, param_unit='$\\,$MeV', point=[0.85, 0.94],
                        param_mapper=param_colormapper, param_format='.1f')
format_save(results_path, f"14VoltageOffset", legend=False, fig=fig)

fig, ax = plt.subplots()
for i, obj in enumerate(factor_cache):
    if i == 0 or i >= len(factor_cache)-1:
        continue
    ax.plot(np.concatenate((obj[0][0], obj[0][1]), axis=0), label=labels[i], color=param_color(energies[i]))
ax.set_xlim(ax.get_xlim()), fast_yscale([0.95, 1.05])
ax.set_xlabel('Diode channel')
ax.set_ylabel('Signal Homogeneity')
improved_gradient_scale(energies, param_cmap, ax_in=ax, param_unit='$\\,$MeV', point=[0.85, 0.94],
                        param_mapper=param_colormapper, param_format='.1f')
format_save(results_path, f"15EnergyComp", legend=False, fig=fig)

fig, ax = plt.subplots()
for i, obj in enumerate(factor_cache):
    if i == 0 or i >= len(factor_cache)-1:
        continue
    y = (np.mean(obj[0][0])-np.mean(obj[0][1])) * 100
    ax.plot(energies[i], y, label=labels[i], color=param_colorv(voltages[i]), marker='x')
ax.set_xlim(ax.get_xlim()), ax.set_ylim(ax.get_ylim())
ax.set_xlabel('Incident Proton Energy (MeV)')
ax.set_ylabel('Line signal offset (\\%)')
improved_gradient_scale(voltages, param_cmapv, ax_in=ax, param_unit='$\\,$V', point=[0.08, 0.94],
                        param_mapper=param_colormapperv, param_format='.1f')
format_save(results_path, f"15EnergyOffset", legend=False, fig=fig)

# --------------------------------------------------------------------------------------------------------------
# Std and signal deviation comp
# --------------------------------------------------------------------------------------------------------------
align_std, na_std, diffs = np.array(align_std)*100, np.array(na_std)*100, np.array(diffs)*100

fig, ax = plt.subplots()
for i, label in enumerate(labels):
    c = color[i]
    if i == 0:
        ax.plot(i, align_std[i], marker='x', color=c, label='Std of Signal Response Distribution')
        ax.plot([i - 0.5, i + 0.5], [diffs[i], diffs[i]], ls='--', color=c, label='Average Signal Spread')
    else:
        ax.plot(i, align_std[i], marker='x', color=c)
        ax.plot([i-0.5, i+0.5], [diffs[i], diffs[i]], ls='--', color=c)
ax.set_ylabel('Response Homogeneity (\\%)')
ax.set_xticks([i for i in range(len(labels))])
ax.set_xticklabels(shortlabels, rotation=90, ha='center')
just_save(results_path, f"16AlignResponseHomogeneityComp", legend=True,)

fig, ax = plt.subplots()
for i, label in enumerate(labels):
    c = color[i]
    if i == 0:
        ax.plot(i, na_std[i], marker='x', color=c, label='Std of Signal Response Distribution')
        ax.plot([i - 0.5, i + 0.5], [diffs[i], diffs[i]], ls='--', color=c, label='Average Signal Spread')
    else:
        ax.plot(i, na_std[i], marker='x', color=c)
        ax.plot([i-0.5, i+0.5], [diffs[i], diffs[i]], ls='--', color=c)
ax.set_ylabel('Response Homogeneity (\\%)')
ax.set_xticks([i for i in range(len(labels))])
ax.set_xticklabels(shortlabels, rotation=90, ha='center')
just_save(results_path, f"16NoAlignResponseHomogeneityComp", legend=True)

fig, ax = plt.subplots()
for i, label in enumerate(labels):
    c = color[i]
    if i == 0:
        ax.plot(energies[i], align_std[i], marker='x', color=c, label='Std of Signal Response Distribution')
        ax.plot([energies[i] - 0.8, energies[i] + 0.8], [diffs[i], diffs[i]], ls='--', color=c, label='Average Signal Spread')
    else:
        ax.plot(energies[i], align_std[i], marker='x', color=c)
        ax.plot([energies[i] - 0.8, energies[i] + 0.8], [diffs[i], diffs[i]], ls='--', color=c)
ax.set_ylabel('Response Homogeneity (\\%)')
ax.set_xlabel('Incident Proton Energy (MeV)')
format_save(results_path, f"16AlignEnergyComp", legend=True, fig=fig)

fig, ax = plt.subplots()
for i, label in enumerate(labels):
    c = color[i]
    if i == 0:
        ax.plot(energies[i], na_std[i], marker='x', color=c, label='Std of Signal Response Distribution')
        ax.plot([energies[i] - 0.8, energies[i] + 0.8], [diffs[i], diffs[i]], ls='--', color=c, label='Average Signal Spread')
    else:
        ax.plot(energies[i], na_std[i], marker='x', color=c)
        ax.plot([energies[i] - 0.8, energies[i] + 0.8], [diffs[i], diffs[i]], ls='--', color=c)
ax.set_ylabel('Response Homogeneity (\\%)')
ax.set_xlabel('Incident Proton Energy (MeV)')
format_save(results_path, f"16NoAlignEnergyComp", legend=True, fig=fig)

fig, ax = plt.subplots()
for i, label in enumerate(labels):
    c = color[i]
    if i == 0:
        ax.plot(voltages[i], align_std[i], marker='x', color=c, label='Std of Signal Response Distribution')
        ax.plot([voltages[i] - 0.1, voltages[i] + 0.1], [diffs[i], diffs[i]], ls='--', color=c, label='Average Signal Spread')
    else:
        ax.plot(voltages[i], align_std[i], marker='x', color=c)
        ax.plot([voltages[i] - 0.1, voltages[i] + 0.1], [diffs[i], diffs[i]], ls='--', color=c)
ax.set_ylabel(f'Response Homogeneity (\\%)')
ax.set_xlabel('ams voltage (V)')
format_save(results_path, f"16AlignVoltageComp", legend=True, fig=fig)

fig, ax = plt.subplots()
for i, label in enumerate(labels):
    c = color[i]
    if i == 0:
        ax.plot(voltages[i], na_std[i], marker='x', color=c, label='Std of Signal Response Distribution')
        ax.plot([voltages[i] - 0.1, voltages[i] + 0.1], [diffs[i], diffs[i]], ls='--', color=c, label='Average Signal Spread')
    else:
        ax.plot(voltages[i], na_std[i], marker='x', color=c)
        ax.plot([voltages[i] - 0.1, voltages[i] + 0.1], [diffs[i], diffs[i]], ls='--', color=c)
ax.set_ylabel(f'Response Homogeneity (\\%)')
ax.set_xlabel('ams voltage (V)')
format_save(results_path, f"16NoAlignVoltageComp", legend=True, fig=fig)