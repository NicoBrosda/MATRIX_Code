from copy import deepcopy

import numpy as np

from EvaluationSoftware.main import *
from EvaluationSoftware.readout_modules import ams_channel_assignment_readout
from EvaluationSoftware.normalization_modules import normalization_from_translated_array

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
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Results_260325/EnergyResponse/')

# ----------------------- Short summary log of measurements -----------------------
# Exp 1 : Dark voltage scan ["dark_"]
# Exp 2 - 5 : Linearity (100 pA - 1 nA at target with 400 um diffuser) ["linearity1_"]
# -> exp 6 empty (not possible)
# -> Until here no um in filename for diffuser thickness - ================= Fixed =================
# Exp 7 - 9 : Norming at different detector voltages ["norm{}V_"]
# Exp 10 - 13 : Norming at different proton energies ["norm{}V_P{}_"]
# Exp 14 - 32 : Mapping of wheel aperture for all wheel positions, P0 - P18 ["energydiffmap_P{}_"]
# Exp 33 - 35 : Norming at different voltages for low proton energy ["norm{}V_P{}_"]
# -> Exp 33 named with 1.9 V instead of 1,9 V - ================= Fixed =================
# Exp 36 - 43 : Mapping of PEEK wedge (wrong position) P0 - P6, P18 ["PEEKwedge_P{}_"]
# Exp 44 - 62 : Mapping of PEEK wedge (correct position) P0, P18, P1 - P17 ["PEEKwedge_P{}_"]
# -> exp 47 with P3 instead of P2 in filename!!! - ================= Fixed =================
# Exp 63 : Increased distance 10 mm, P0 ["LargerGap10mm_P{}_"]
# Exp 64 : Dark Voltage End of Day ["darkEnd_"]
# Exp 65 : Increased distance 10 mm, P7 ["LargerGap10mm_P{}_"]
# ------------------------ End day1 ----------------------------------------------
# Exp 66 : Dark Voltage Scan Day2 1-2V ["Dark_"]
# Exp 67 : Dark Voltage Scan Day2 0-2V ["Dark2_"]
# Exp 68 : Voltage Scan 0.8-2V with beam (PEEK wedge, P7, 10 mm distance) ["BeamCurrent1_"]
# Exp 69 - 71 : Increased distance 10 mm, P7, P12, P16 ["Distance10mm_P{}_"]
# Exp 72 - 75 : Increased distance 20 mm, P0, P7, P12, P16 ["Distance20mm_P{}_"]
# Gafchromic I - VII
# -> Switch to 200 um diffuser
# Exp 76 : Norming day2 P0 and 1.9 V ["normday2_"]
# Exp 77 - 95 : Mapping of wheel aperture for all wheel positions, P0 - P18 ["energyDep_"]
# -> Exp 80 contains two runs - only the run with _bis_ is good (no beam in other run) - ====== Fixed =======
# Exp 96 - 97 : Mapping of PEEK wedge (wrong position) P0, P18 ["PEEKWedge_P{}_"]
# Exp 98 - 117 : Mapping of PEEK wedge (correct position) P18, P0 - P17, P19 ["PEEKWedge_P{}_"]
# Gafchromic VIII - XI
# Exp 118 - 125 : Wedge border in middle of aperture P19 - P12 ["PEEKWedgeMiddle_P{}_"]
# -> Exp 120 named labeled falsely with 118 - needs to be identified with P19 / P17 for real Exp 120 - ==== Fixed =====
# Gafchromic XII
# Exp 126 - 128 : Straggling test distance 5 mm - P0, P12, Misc ["Round8mm_5mm_P{}_", "Misc_5mm_P0_"]
# Exp 129 - 131 : Straggling test distance 10 mm - Misc, P0, P12 ["Round8mm_10mm_P{}_", "Misc_10mm_P0_"]
# Exp 132 - 134 : Straggling test distance 20 mm - P12, P0, Misc ["Round8mm_20mm_P{}_", "Misc_20mm_P0_"]
# Exp 135 - 137 : Straggling test distance 40 mm - Misc, P0, P12 ["Round8mm_40mm_P{}_", "Misc_40mm_P0_"]
# Exp 138 : Dark voltage scan end 0-2V ["DarkEnd_"]

new_measurements = []

diff_400um = [f'exp{i+14}_energydiffmap_P{i}_' for i in range(19)]
diff_200um= [f'exp{i+77}_energyDep_P{i}_' for i in range(19)] + ['exp118_PEEKWedgeMiddle_P19_']

new_measurements += diff_400um
new_measurements += diff_200um

data_wheel_200 = pd.read_csv('../../Files/energies_after_wheel_diffusor200.txt', sep='\t', header=4,
                         names=['position', 'thickness', 'energies'])
data_wheel_400 = pd.read_csv('../../Files/energies_after_wheel_diffusor400.txt', sep='\t', header=4,
                         names=['position', 'thickness', 'energies'])

dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_260325/')

dark_paths_array1 = ['exp1_dark_0nA_400um_nA_1.9_x_20.0_y_68.0', 'exp64_darkEnd_0.5nA_400um_nA_1.9_x_20.0_y_68.0',
                     '2exp66_Dark_0.0nA_0um_nA_1.9_x_20.0_y_68.0', '2exp138_DarkEnd_0nA_200um_nA_1.9_x_20.0_y_68.0']

norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_260325/')
norm_array1 = ['exp7_norm1,9V_']
# norm_array1 = ['2exp76_normday2_P0_']

cache_400 = []
cache_200 = []

for k, crit in enumerate(new_measurements[0:]):
    print('-'*50)
    print(crit)
    print('-'*50)

    if k < len(diff_400um):
        diffuser = 400
    else:
        diffuser = 200

    wheel_position = int(crit[crit.rindex('_P')+2:crit.rindex('_')])

    A = Analyzer((2, 64), (0.4, 0.4), (0.1, 0.1), readout=readout,
                 diode_offset=[[0, - 0.25], np.zeros(64)], position_parser=position_parser,
                 voltage_parser=voltage_parser, current_parser=current_parser)
    dark = dark_paths_array1
    A.set_measurement(folder_path, crit)
    A.set_dark_measurement(dark_path, dark)
    norm = norm_array1
    norm_func = lambda list_of_files, instance, method='least_squares': normalization_from_translated_array_v3(
        list_of_files, instance, method, align_lines=True)
    A.normalization(norm_path, norm, normalization_module=norm_func)
    A.load_measurement()
    A.create_map(inverse=[True, False])

    # -------------- Save signal level to a cache list --------------
    if 'PEEK' in crit:
        # I will consider the whole image even with the Bragg wedge - because the signal seems to be max below the wedge
        # Otherwise likely range 64-28:24
        signals = A.signal_conversion(np.array([i['signal'][:, 0:] for i in A.measurement_data]).flatten())
        stds = A.signal_conversion(np.array([i['std'][:, 0:] for i in A.measurement_data]).flatten())
    else:
        signals = A.signal_conversion(np.array([i['signal'] for i in A.measurement_data]).flatten())
        stds = A.signal_conversion(np.array([i['std'] for i in A.measurement_data]).flatten())

    signal_indices = np.argsort(signals)[-200:]
    signal_levels = signals[signal_indices]
    std = np.sqrt(np.mean(stds[signal_indices])**2+np.std(signal_levels)**2)
    signal_level = np.mean(signal_levels)
    if diffuser == 400:
        cache_400.append([wheel_position, signal_level, std])
    elif diffuser == 200:
        cache_200.append([wheel_position, signal_level, std])

    # -------------- Plot 1: Hist of Signal map --------------
    fig, ax = plt.subplots()
    bins = 150
    ax.hist(signals, bins=bins, color='k')
    ax.axvline(signal_level, color='b', label='estimated signal level')
    ax.axvline(signal_levels[0], color='r', label='lower border signal level')

    ax.set_xlabel(f'Signal Current ({scale_dict[A.scale][1]}A)')
    ax.set_ylabel(f'counts per bin ({bins} bins)')
    format_save(results_path / f'Histograms{diffuser}um/', A.name, legend=True)

# -------------- Plot 2: Curve signal vs energy for 400 um diffuser--------------
energy_cmap = sns.color_palette("crest_r", as_cmap=True)
energy_colormapper = lambda energy: color_mapper(energy, np.min(data_wheel_400['energies'][0:len(cache_400)]), np.max(data_wheel_400['energies'][0:len(cache_400)]))
energy_color = lambda energy: energy_cmap(energy_colormapper(energy))

fig, ax = plt.subplots()
ax.plot(data_wheel_400['energies'][0:len(cache_400)], [i[1] for i in cache_400], c='k', ls='-', marker='')
for i in cache_400:
    ax.errorbar(data_wheel_400['energies'][i[0]], i[1], i[2], c=energy_color(data_wheel_400['energies'][i[0]]), marker='x', capsize=5, markersize=7)

ax.set_xlabel(f'Proton energy (MeV)')
ax.set_ylabel(f'Signal Current ({scale_dict[A.scale][1]}A)')

ax.set_xlim(ax.get_xlim())
ax.set_ylim(ax.get_ylim())

gradient_arrow(ax, transform_axis_to_data_coordinates(ax, [0.9, 0.925]),
                       transform_axis_to_data_coordinates(ax, [0.9, 0.795]), cmap=energy_cmap, lw=10, zorder=5)
ax.text(*transform_axis_to_data_coordinates(ax, [0.82, 0.94]),
        f"{np.min(data_wheel_400['energies'][0:len(cache_400)]): .2f}$\,${'MeV'}", fontsize=13,
        c=energy_color(np.min(data_wheel_400['energies'])), zorder=3)  # , bbox={'facecolor': freq_colour(1033), 'alpha': 0.2, 'pad': 2})
ax.text(*transform_axis_to_data_coordinates(ax, [0.80, 0.71]),
        f"{np.max(data_wheel_400['energies'][0:len(cache_400)]): .2f}$\,${'MeV'}", fontsize=13,
        c=energy_color(np.max(data_wheel_400['energies'])), zorder=3)  # , bbox={'facecolor': freq_colour(32033), 'alpha': 0.2, 'pad': 2})
format_save(results_path, f'400umResponse', legend=False)


# -------------- Plot 3: Hist of Signal map --------------
energy_cmap = sns.color_palette("crest_r", as_cmap=True)
energy_colormapper = lambda energy: color_mapper(energy, np.min(data_wheel_200['energies'][0:len(cache_200)]), np.max(data_wheel_200['energies'][0:len(cache_200)]))
energy_color = lambda energy: energy_cmap(energy_colormapper(energy))

fig, ax = plt.subplots()
ax.plot(data_wheel_200['energies'][0:len(cache_200)], [i[1] for i in cache_200], c='k', ls='-', marker='')
for i in cache_200:
    ax.errorbar(data_wheel_200['energies'][i[0]], i[1], i[2], c=energy_color(data_wheel_200['energies'][i[0]]), marker='x', capsize=5, markersize=7)

ax.set_xlabel(f'Proton energy (MeV)')
ax.set_ylabel(f'Signal Current ({scale_dict[A.scale][1]}A)')

ax.set_xlim(ax.get_xlim())
ax.set_ylim(ax.get_ylim())

gradient_arrow(ax, transform_axis_to_data_coordinates(ax, [0.9, 0.925]),
                       transform_axis_to_data_coordinates(ax, [0.9, 0.795]), cmap=energy_cmap, lw=10, zorder=5)
ax.text(*transform_axis_to_data_coordinates(ax, [0.82, 0.94]),
        f"{np.min(data_wheel_200['energies'][0:len(cache_200)]): .2f}$\,${'MeV'}", fontsize=13,
        c=energy_color(np.min(data_wheel_200['energies'])), zorder=3)  # , bbox={'facecolor': freq_colour(1033), 'alpha': 0.2, 'pad': 2})
ax.text(*transform_axis_to_data_coordinates(ax, [0.80, 0.71]),
        f"{np.max(data_wheel_200['energies'][0:len(cache_200)]): .2f}$\,${'MeV'}", fontsize=13,
        c=energy_color(np.max(data_wheel_200['energies'])), zorder=3)  # , bbox={'facecolor': freq_colour(32033), 'alpha': 0.2, 'pad': 2})
format_save(results_path , f'200umResponse', legend=False)