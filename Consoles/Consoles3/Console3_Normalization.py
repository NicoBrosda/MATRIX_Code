import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

from EvaluationSoftware.main import *
from EvaluationSoftware.readout_modules import ams_channel_assignment_readout
from EvaluationSoftware.normalization_modules import normalization_from_translated_array
from skimage.filters import threshold_otsu as ski_threshold_otsu

mapping = Path('../../Files/mapping.xlsx')
data = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:])-1 for k in data['direction_2']]

readout, position_parser = lambda x, y: ams_channel_assignment_readout(x, y, channel_assignment=channel_assignment), standard_position

A = Analyzer((1, 128), 0.5, 0.0, readout=readout)


folder_path = Path('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/matrix_230924/')

dark_paths_array1 = ['voltage_scan_no_beam_nA_1.8000000000000005_x_20.0_y_70.0.csv',
                     'd2_1n_5s_flat_calib_nA_1.8000000000000007_x_20.0_y_70.0.csv']

dark_paths_array3_1V = ['Array3_VoltageScan_dark_nA_1.0_x_0.0_y_40.0.csv']

dark_paths_array3 = ['Array3_VoltageScan_dark_nA_1.8_x_0.0_y_40.0.csv']

norm_array1 = ['Normalization2']
norm_array1 = ['uniformity_scan_']
norm_array1 = ['uniformity_scan_', 'Normalization2']

y_scans = ['uniformity_scan_', 'BraggYScan']

norm_array3 = ['Array3_DiffuserYScan']

norm_arrays = norm_array1+norm_array3
# norm_arrays = norm_array3

storage = []
storage_array1 = []

for k, crit in enumerate(norm_arrays):
    print('-'*50)
    print(crit)
    print('-'*50)

    # Correct sizing of the arrays
    if 'Array3' in crit:
        A.diode_size = (0.25, 0.5)
        A.diode_size = (0.17, 0.4)
        A.diode_spacing = (0.08, 0.1)

    else:
        A.diode_size = (0.5, 0.5)
        A.diode_size = (0.4, 0.4)
        A.diode_spacing = (0.1, 0.1)

    # Filtering for correct files - Logo would be found in Array3_Logo...
    if crit == 'Logo':
        A.set_measurement(folder_path, crit, blacklist=['png', 'Array3'])
    elif crit == 'Array3_DiffuserYScan':
        A.set_measurement(folder_path, crit, blacklist=['png', 'x_0.0_'])
    else:
        A.set_measurement(folder_path, crit)

    # Dark Subtraction - correct file assignment
    if crit == 'Array3_Logo':
        dark = dark_paths_array3
    elif 'Array3' in crit:
        dark = dark_paths_array3_1V
    else:
        dark = dark_paths_array1

    A.set_dark_measurement(folder_path, dark)

    # A.plot_diodes('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/Norm/AfterNorm/', direction='y', plotting_range=None)

    if 'Array3' in crit:
        norm = norm_array3
    else:
        norm = norm_array1

    # -----------------------------------------------------------------------------------------------------------------
    # Start of norm module - deconstructed
    # -----------------------------------------------------------------------------------------------------------------
    list_of_files = A.measurement_files
    instance = A
    method = 'least_squares'
    correction = 0

    # Little helper function to group the recalculated positions
    def group(input_list, group_range):
        class Group:
            def __init__(self, value):
                self.mean = value
                self.start_value = value
                self.members = 1

            def add(self, value):
                self.mean = (self.mean * self.members + value) / (self.members + 1)
                self.members += 1

            def check_add(self, value):
                return (self.mean * self.members + value) / (self.members + 1)

        if not isinstance(input_list, (np.ndarray, pd.DataFrame)):
            input_list = np.array(input_list)
        input_list = input_list.flatten()

        groups = []
        for j in input_list:
            if len(groups) == 0:
                groups.append(Group(j))
                continue
            in_group = False
            for group in groups:
                if group.mean - group_range <= j <= group.mean + group_range:
                    # check for drift of the group:
                    if group.start_value - group_range < group.check_add(j) < group.start_value + group_range:
                        group.add(j)
                        in_group = True
                        break

            if not in_group:
                groups.append(Group(j))

        return [g.mean for g in groups if g.members != 0]

    # Direction of the measurement: x or y
    if instance.diode_dimension[0] > instance.diode_dimension[1]:
        sp = 0
    elif instance.diode_dimension[0] < instance.diode_dimension[1]:
        sp = 1
    else:
        sp = 0
    # Load in the data from a list of files in a folder; save position and signal
    position = []
    signals = []
    for file in tqdm(list_of_files):
        # The parsing of the position out of the name and save it
        position.append(instance.pos_parser(file))
        signal = instance.readout(file, instance)['signal']
        signals.append((np.array(signal)-instance.dark).flatten())

    # Sort the arrays by the position
    indices = np.argsort(np.array(position)[:, sp])
    signals = np.array(signals)[indices]
    position = np.array(position)[indices]

    # -----------------------------------------------------------------------------------------------------------------
    # Plot 1: Signals before shifting to real positions
    # -----------------------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots()
    diode_cmap = sns.color_palette("coolwarm", as_cmap=True)
    diode_colormapper = lambda diode: color_mapper(diode, 0, instance.diode_dimension[sp])
    diode_color = lambda diode: diode_cmap(diode_colormapper(diode))
    for i in range(instance.diode_dimension[sp]):
        ax.plot(position[:, 1], signals[:, i], c=diode_color(i), zorder=1)
    ax.set_xlabel('Position of stage during y-scan - no conversion (mm)')
    ax.set_ylabel('Diode signal (a.u.)')
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    gradient_arrow(ax, transform_axis_to_data_coordinates(ax, [0.11, 0.92]),
                   transform_axis_to_data_coordinates(ax, [0.11, 0.79]), cmap=diode_cmap, lw=10, zorder=5)
    ax.text(*transform_axis_to_data_coordinates(ax, [0.035, 0.94]), r'Diode $\#$1', fontsize=13,
            c=diode_color(0), zorder=3)  # , bbox={'facecolor': freq_colour(1033), 'alpha': 0.2, 'pad': 2})
    ax.text(*transform_axis_to_data_coordinates(ax, [0.02, 0.71]), r'Diode $\#$' + str(instance.diode_dimension[sp]),
            fontsize=13, c=diode_color(instance.diode_dimension[sp]), zorder=3)  # , bbox={'facecolor': freq_colour(32033), 'alpha': 0.2, 'pad': 2})
    format_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/NormMethod/', crit+'01_PreShift',
                legend=False)

    # -----------------------------------------------------------------------------------------------------------------
    # Recalculate the positions considering the size of the diodes and thus the expected real positions
    positions = []
    for i in range(np.shape(signals)[1]):
        cache = deepcopy(position)
        cache[:, sp] = cache[:, sp] + i * (instance.diode_size[sp] + instance.diode_spacing[sp] + correction)
        positions.append(cache)
    positions = np.array(positions)

    # -----------------------------------------------------------------------------------------------------------------
    # Plot 2: Signals after shifting to real positions
    # -----------------------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots()
    diode_cmap = sns.color_palette("coolwarm", as_cmap=True)
    diode_colormapper = lambda diode: color_mapper(diode, 0, instance.diode_dimension[sp])
    diode_color = lambda diode: diode_cmap(diode_colormapper(diode))
    for i in range(instance.diode_dimension[sp]):
        ax.plot(positions[i, :, 1], signals[:, i], c=diode_color(i))
    ax.set_xlabel('Real position of diodes during measurement (mm)')
    ax.set_ylabel('Diode signal (a.u.)')
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    gradient_arrow(ax, transform_axis_to_data_coordinates(ax, [0.11, 0.92]),
                   transform_axis_to_data_coordinates(ax, [0.11, 0.79]), cmap=diode_cmap, lw=10, zorder=5)
    ax.text(*transform_axis_to_data_coordinates(ax, [0.035, 0.94]), r'Diode $\#$1', fontsize=13,
            c=diode_color(0), zorder=3)  # , bbox={'facecolor': freq_colour(1033), 'alpha': 0.2, 'pad': 2})
    ax.text(*transform_axis_to_data_coordinates(ax, [0.02, 0.71]), r'Diode $\#$' + str(instance.diode_dimension[sp]),
            fontsize=13, c=diode_color(instance.diode_dimension[sp]),
            zorder=3)  # , bbox={'facecolor': freq_colour(32033), 'alpha': 0.2, 'pad': 2})
    format_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/NormMethod/', crit+'02_AfterShift',
                legend=False)

    # -----------------------------------------------------------------------------------------------------------------
    # Try to detect the signal level
    threshold = ski_threshold_otsu(signals)
    print(threshold)

    # Group the positions after their recalculation to gain a grid, from which the mean calculation is meaningful
    group_distance = instance.diode_size[sp]
    groups = group(positions[:, :, sp].flatten(), group_distance)

    # Calculate the mean for each grouped position, consider only the diode signals that were close to this position
    mean_result = []
    mean_new = []
    mean_x_new = []
    groups = np.sort(groups)
    for mean in groups:
        indices = []
        for k, channel in enumerate(np.array(positions)[:, :, sp]):
            index_min = np.argsort(np.abs(channel - mean))[0]
            if np.abs(channel[index_min] - mean) <= group_distance:
                indices.append(index_min)
            else:
                indices.append(None)
        cache = 0
        cache_new = 0
        j = 0
        j_new = 0
        for i in range(len(indices)):
            if indices[i] is not None and signals[indices[i]][i] != 0:
                cache += signals[indices[i]][i]
                j += 1
            if indices[i] is not None and signals[indices[i]][i] >= threshold:
                cache_new += signals[indices[i]][i]
                j_new += 1
        if j > 0:
            mean_result.append(cache / j)
        else:
            mean_result.append(cache)
        if j_new > 0:
            mean_new.append(cache_new / j_new)
            mean_x_new.append(mean)

    mean_result = np.array(mean_result)
    mean_x = np.array(groups)
    mean_x_new, mean_new = np.array(mean_x_new), np.array(mean_new)

    # -----------------------------------------------------------------------------------------------------------------
    # Plot 3: Signals and their calculated mean
    # -----------------------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots()
    diode_cmap = sns.color_palette("coolwarm", as_cmap=True)
    diode_colormapper = lambda diode: color_mapper(diode, 0, instance.diode_dimension[sp])
    diode_color = lambda diode: diode_cmap(diode_colormapper(diode))
    for i in range(instance.diode_dimension[sp]):
        ax.plot(positions[i, :, 1], signals[:, i], c=diode_color(i), zorder=1)
    ax.plot(mean_x, mean_result, c='k', ls='-', zorder=2, label='Mean total signal')

    ax.set_xlabel('Real position of diodes during measurement (mm)')
    ax.set_ylabel('Diode signal (a.u.)')
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    gradient_arrow(ax, transform_axis_to_data_coordinates(ax, [0.11, 0.92]),
                   transform_axis_to_data_coordinates(ax, [0.11, 0.79]), cmap=diode_cmap, lw=10, zorder=5)
    ax.text(*transform_axis_to_data_coordinates(ax, [0.035, 0.94]), r'Diode $\#$1', fontsize=13,
            c=diode_color(0), zorder=3)  # , bbox={'facecolor': freq_colour(1033), 'alpha': 0.2, 'pad': 2})
    ax.text(*transform_axis_to_data_coordinates(ax, [0.02, 0.71]), r'Diode $\#$' + str(instance.diode_dimension[sp]),
            fontsize=13, c=diode_color(instance.diode_dimension[sp]),
            zorder=3)  # , bbox={'facecolor': freq_colour(32033), 'alpha': 0.2, 'pad': 2})
    format_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/NormMethod/', crit + '03_Mean',
                legend=False)

    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    # Plot 4: Threshold between signal and no signal region
    # -----------------------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots()
    diode_cmap = sns.color_palette("coolwarm", as_cmap=True)
    diode_colormapper = lambda diode: color_mapper(diode, 0, instance.diode_dimension[sp])
    diode_color = lambda diode: diode_cmap(diode_colormapper(diode))
    for i in range(instance.diode_dimension[sp]):
        ax.plot(positions[i, :, 1], signals[:, i], c=diode_color(i), zorder=1)
    ax.plot(mean_x, mean_result, c='k', ls='-', zorder=2, label='Mean total signal')
    ax.plot(mean_x_new, mean_new, c='gold', ls='--', zorder=2, label='Mean above threshold')

    ax.axhline(threshold, color='m', label='Threshold to identify signal')
    ax.set_xlabel('Real position of diodes during measurement (mm)')
    ax.set_ylabel('Diode signal (a.u.)')
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    gradient_arrow(ax, transform_axis_to_data_coordinates(ax, [0.11, 0.92]),
                   transform_axis_to_data_coordinates(ax, [0.11, 0.79]), cmap=diode_cmap, lw=10, zorder=5)
    ax.text(*transform_axis_to_data_coordinates(ax, [0.035, 0.94]), r'Diode $\#$1', fontsize=13,
            c=diode_color(0), zorder=3)  # , bbox={'facecolor': freq_colour(1033), 'alpha': 0.2, 'pad': 2})
    ax.text(*transform_axis_to_data_coordinates(ax, [0.02, 0.71]), r'Diode $\#$' + str(instance.diode_dimension[sp]),
            fontsize=13, c=diode_color(instance.diode_dimension[sp]),
            zorder=3)  # , bbox={'facecolor': freq_colour(32033), 'alpha': 0.2, 'pad': 2})
    ax.legend(loc=4)
    format_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/NormMethod/', crit + '04_Threshold',
                legend=False)

    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    # Plot 5: Zoom into mean at signal region
    # -----------------------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots()
    diode_cmap = sns.color_palette("coolwarm", as_cmap=True)
    diode_colormapper = lambda diode: color_mapper(diode, 0, instance.diode_dimension[sp])
    diode_color = lambda diode: diode_cmap(diode_colormapper(diode))
    for diode in range(instance.diode_dimension[sp]):
        above_threshold = (signals[:, diode] > threshold)
        ax.plot(positions[diode, above_threshold, 1], signals[above_threshold, diode], c=diode_color(diode), zorder=1)
    above_threshold = mean_result > threshold
    ax.plot(mean_x[above_threshold], mean_result[above_threshold], c='k', ls='-', zorder=2, label='Mean total signal')
    ax.plot(mean_x_new, mean_new, c='gold', ls='--', zorder=2, label='Mean above threshold')

    ax.set_xlabel('Real position of diodes during measurement (mm)')
    ax.set_ylabel('Diode signal (a.u.)')
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    gradient_arrow(ax, transform_axis_to_data_coordinates(ax, [0.11, 0.27]),
                   transform_axis_to_data_coordinates(ax, [0.11, 0.14]), cmap=diode_cmap, lw=10, zorder=5)
    ax.text(*transform_axis_to_data_coordinates(ax, [0.035, 0.29]), r'Diode $\#$1', fontsize=13,
            c=diode_color(0), zorder=3)  # , bbox={'facecolor': freq_colour(1033), 'alpha': 0.2, 'pad': 2})
    ax.text(*transform_axis_to_data_coordinates(ax, [0.02, 0.06]), r'Diode $\#$' + str(instance.diode_dimension[sp]),
            fontsize=13, c=diode_color(instance.diode_dimension[sp]),
            zorder=3)  # , bbox={'facecolor': freq_colour(32033), 'alpha': 0.2, 'pad': 2})
    ax.legend(loc=4)
    format_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/NormMethod/', crit + '05_Zoom',
                legend=False)
    # -----------------------------------------------------------------------------------------------------------------
    # Mean Restriction to signal region
    # above_threshold = mean_result > threshold
    # mean_x_new = mean_x[above_threshold]
    # mean_new = mean_result[above_threshold]
    # Interpolation
    factor = np.array([])
    factor_new = np.array([])
    for channel in range(np.shape(signals)[1]):
        mean_interp = np.interp(positions[channel, :, sp], mean_x, mean_result)
        restrained_position = (np.min(mean_x_new) <= positions[channel, :, sp]) & (positions[channel, :, sp] <= np.max(mean_x_new))
        mean_interp_new = np.interp(positions[channel, :, sp][restrained_position], mean_x_new, mean_new)
        if isinstance(method, (float, int, np.float64)):
            # Method 1: Threshold for range consideration, for each diode channel mean of the factor between points
            factor2 = mean_interp[signals[:, channel] > method] / signals[:, channel][signals[:, channel] > method]
            factor2 = np.mean(factor2)
            if np.isnan(factor2):
                factor2 = 0
        elif method == 'least_squares':
            # Method 2: Optimization with least squares method, recommended
            func_opt = lambda a: mean_interp - signals[:, channel] * a
            func_opt_new = lambda a: mean_interp_new - signals[:, channel][restrained_position] * a
            factor2 = least_squares(func_opt, 1)
            factor_new_cache = least_squares(func_opt_new, 1)
            if factor2.nfev == 1 and factor2.optimality == 0.0:
                factor2 = 0
            else:
                factor2 = factor2.x
            if factor_new_cache.nfev == 1 and factor_new_cache.optimality == 0.0:
                factor_new_cache = 0
            else:
                factor_new_cache = factor_new_cache.x
        else:
            # Standard method: For the moment method 1 with automatic threshold
            factor2 = mean_interp[signals[:, channel] > np.mean(signals[:, channel])] / \
                      signals[:, channel][signals[:, channel] > np.mean(signals[:, channel])]
            factor2 = np.mean(factor2)
            if np.isnan(factor2):
                factor2 = 0
        factor = np.append(factor, factor2)
        factor_new = np.append(factor_new, factor_new_cache)
    # Save a raw factor to allow later changes on the factor's parameters
    # -----------------------------------------------------------------------------------------------------------------
    # Plot 6: Factor
    # -----------------------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots()
    diode_cmap = sns.color_palette("coolwarm", as_cmap=True)
    diode_colormapper = lambda diode: color_mapper(diode, 0, instance.diode_dimension[sp])
    diode_color = lambda diode: diode_cmap(diode_colormapper(diode))
    ax.plot(factor, c='b', ls='-',  label='Considering whole factor')
    ax.plot(factor_new, c='r', ls='--', label='Considering only above threshold region')
    ax.set_xlabel(r'$\#$Diode')
    ax.set_ylabel(r'Deviation mean to $\#$Diode signal alias Normalization factor')
    ax.set_ylim(0.93, 1.07)
    format_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/NormMethod/', crit+'06_Factor',
                legend=True)

    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    # Plot 7: Factor applied - Zoom
    # -----------------------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots()
    diode_cmap = sns.color_palette("coolwarm", as_cmap=True)
    diode_colormapper = lambda diode: color_mapper(diode, 0, instance.diode_dimension[sp])
    diode_color = lambda diode: diode_cmap(diode_colormapper(diode))
    for diode in range(instance.diode_dimension[sp]):
        above_threshold = (signals[:, diode] > threshold)
        ax.plot(positions[diode, above_threshold, 1], signals[above_threshold, diode]*factor[diode], c=diode_color(diode), zorder=1, alpha=0.8)
        ax.plot(positions[diode, above_threshold, 1], signals[above_threshold, diode], c='grey', zorder=0, alpha=0.4)

    above_threshold = mean_result > threshold
    ax.plot(mean_x[above_threshold], mean_result[above_threshold], c='k', zorder=2)

    ax.set_xlabel('Real position of diodes during measurement (mm)')
    ax.set_ylabel('Diode signal (a.u.)')
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    gradient_arrow(ax, transform_axis_to_data_coordinates(ax, [0.11, 0.27]),
                   transform_axis_to_data_coordinates(ax, [0.11, 0.14]), cmap=diode_cmap, lw=10, zorder=5)
    ax.text(*transform_axis_to_data_coordinates(ax, [0.035, 0.29]), r'Diode $\#$1', fontsize=13,
            c=diode_color(0), zorder=3)  # , bbox={'facecolor': freq_colour(1033), 'alpha': 0.2, 'pad': 2})
    ax.text(*transform_axis_to_data_coordinates(ax, [0.02, 0.06]), r'Diode $\#$' + str(instance.diode_dimension[sp]),
            fontsize=13, c=diode_color(instance.diode_dimension[sp]),
            zorder=3)  # , bbox={'facecolor': freq_colour(32033), 'alpha': 0.2, 'pad': 2})
    format_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/NormMethod/', crit + '07_FactorApplied',
                legend=False)
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    # Plot 8: Factor_new applied - Zoom
    # -----------------------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots()
    diode_cmap = sns.color_palette("coolwarm", as_cmap=True)
    diode_colormapper = lambda diode: color_mapper(diode, 0, instance.diode_dimension[sp])
    diode_color = lambda diode: diode_cmap(diode_colormapper(diode))
    for diode in range(instance.diode_dimension[sp]):
        above_threshold = (signals[:, diode] > threshold)
        ax.plot(positions[diode, above_threshold, 1], signals[above_threshold, diode] * factor_new[diode],
                c=diode_color(diode), zorder=1, alpha=0.8)
        ax.plot(positions[diode, above_threshold, 1], signals[above_threshold, diode], c='grey', zorder=0, alpha=0.4)

    above_threshold = mean_result > threshold
    ax.plot(mean_x_new, mean_new, c='gold', ls='--', zorder=2)

    ax.set_xlabel('Real position of diodes during measurement (mm)')
    ax.set_ylabel('Diode signal (a.u.)')
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    gradient_arrow(ax, transform_axis_to_data_coordinates(ax, [0.11, 0.27]),
                   transform_axis_to_data_coordinates(ax, [0.11, 0.14]), cmap=diode_cmap, lw=10, zorder=5)
    ax.text(*transform_axis_to_data_coordinates(ax, [0.035, 0.29]), r'Diode $\#$1', fontsize=13,
            c=diode_color(0), zorder=3)  # , bbox={'facecolor': freq_colour(1033), 'alpha': 0.2, 'pad': 2})
    ax.text(*transform_axis_to_data_coordinates(ax, [0.02, 0.06]), r'Diode $\#$' + str(instance.diode_dimension[sp]),
            fontsize=13, c=diode_color(instance.diode_dimension[sp]),
            zorder=3)  # , bbox={'facecolor': freq_colour(32033), 'alpha': 0.2, 'pad': 2})
    format_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/NormMethod/',
                crit + '08_FactorNewApplied',
                legend=False)
    # -----------------------------------------------------------------------------------------------------------------
    if not 'Array3' in crit:
        storage_array1.append([crit, instance, (mean_x_new, mean_new), (positions, signals), threshold, factor_new])
    storage.append([crit, instance, (mean_x_new, mean_new), (positions, signals), threshold, factor_new])

# -----------------------------------------------------------------------------------------------------------------
# Plot 9: Factor Array 1 comparison
# -----------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots()
diode_cmap = sns.color_palette("coolwarm", as_cmap=True)
diode_colormapper = lambda diode: color_mapper(diode, 0, instance.diode_dimension[sp])
diode_color = lambda diode: diode_cmap(diode_colormapper(diode))
c = sns.color_palette("tab10")
for i, stored_data in enumerate(storage_array1):
    ax.plot(stored_data[-1], c=c[i], ls='-', label=stored_data[0])
ax.set_xlabel(r'$\#$Diode')
ax.set_ylabel(r'Deviation mean to $\#$Diode signal alias Normalization factor')
ax.set_ylim(0.93, 1.07)
format_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/NormMethod/', 'Array1_09_Factor_Comparison',
            legend=True)
# -----------------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------------------
# Plot 10: Calibs Array 1 comparison
# -----------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots()
diode_cmap = sns.color_palette("coolwarm", as_cmap=True)
diode_colormapper = lambda diode: color_mapper(diode, 0, instance.diode_dimension[sp])
diode_color = lambda diode: diode_cmap(diode_colormapper(diode))

c = sns.color_palette("tab10")
for i, stored_data in enumerate(storage_array1):
    signals = stored_data[3][1]
    positions = stored_data[3][0]
    threshold = stored_data[4]
    factor = stored_data[-1]
    for diode in range(stored_data[1].diode_dimension[sp]):
        above_threshold = (signals[:, diode] > threshold)
        ax.plot(positions[diode, above_threshold, 1], signals[above_threshold, diode] * factor[diode],
                c=c[i], zorder=1, alpha=0.3)
        ax.plot(positions[diode, above_threshold, 1], signals[above_threshold, diode], c='grey', zorder=0, alpha=0.1)
    ax.plot(*stored_data[2], c=c[i], zorder=-2, alpha=1, label=stored_data[0])
    ax.plot(*stored_data[2], c='k', zorder=2, alpha=1)

ax.set_xlabel('Real position of diodes during measurement (mm)')
ax.set_ylabel('Diode signal (a.u.)')
ax.set_xlim(ax.get_xlim())
ax.set_ylim(ax.get_ylim())

format_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/NormMethod/',
            crit + 'Array1_10_SignalComparison',
            legend=True)
# -----------------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------------------
# Plot 11: Factor Array 1 comparison
# -----------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots()
diode_cmap = sns.color_palette("coolwarm", as_cmap=True)
diode_colormapper = lambda diode: color_mapper(diode, 0, instance.diode_dimension[sp])
diode_color = lambda diode: diode_cmap(diode_colormapper(diode))
c = sns.color_palette("tab10")
for i, stored_data in enumerate(storage):
    ax.plot(stored_data[-1], c=c[i], ls='-', label=stored_data[0])
ax.set_xlabel(r'$\#$Diode')
ax.set_ylabel(r'Deviation mean to $\#$Diode signal alias Normalization factor')
ax.set_ylim(0.93, 1.07)
format_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/NormMethod/', 'All_11_Factor_Comparison',
            legend=True)
# -----------------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------------------
# Plot 12: Comparison of all normalizations
# -----------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots()
diode_cmap = sns.color_palette("coolwarm", as_cmap=True)
diode_colormapper = lambda diode: color_mapper(diode, 0, instance.diode_dimension[sp])
diode_color = lambda diode: diode_cmap(diode_colormapper(diode))

c = sns.color_palette("tab10")
for i, stored_data in enumerate(storage):
    signals = stored_data[3][1]
    positions = stored_data[3][0]
    threshold = stored_data[4]
    factor = stored_data[-1]
    for diode in range(stored_data[1].diode_dimension[sp]):
        above_threshold = (signals[:, diode] > threshold)
        ax.plot(positions[diode, above_threshold, 1], signals[above_threshold, diode] * factor[diode],
                c=c[i], zorder=1, alpha=0.2)
        ax.plot(positions[diode, above_threshold, 1], signals[above_threshold, diode], c='grey', zorder=0, alpha=0.05)
    ax.plot(*stored_data[2], c=c[i], zorder=-2, alpha=1, label=stored_data[0])
    ax.plot(*stored_data[2], c='k', zorder=2, alpha=1)

ax.set_xlabel('Real position of diodes during measurement (mm)')
ax.set_ylabel('Diode signal (a.u.)')
ax.set_xlim(ax.get_xlim())
ax.set_ylim(ax.get_ylim())

format_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/NormMethod/',
            crit + 'All_12_SignalComparison',
            legend=True)
# -----------------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------------------
# Plot 13: Factor of Array1, Array3, older Array
# -----------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots()
diode_cmap = sns.color_palette("coolwarm", as_cmap=True)
diode_colormapper = lambda diode: color_mapper(diode, 0, instance.diode_dimension[sp])
diode_color = lambda diode: diode_cmap(diode_colormapper(diode))
c = sns.color_palette("tab10")
for i, stored_data in enumerate(storage):
    ax.plot(stored_data[-1], c=c[i], ls='-', label=stored_data[0])
ax.plot(np.load(Path("/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/matrix_19062024/['5s_flat_calib_']_normalization_factor.npy"))[0], c=c[i+1], label='Factor old 64 diode array')
ax.set_xlabel(r'$\#$Diode')
ax.set_ylabel(r'Deviation mean to $\#$Diode signal alias Normalization factor')
ax.set_ylim(0.5, 2)
format_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/NormMethod/', 'ALL_13_Factor_Comparison',
            legend=True)
# -----------------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------------------
# Plot 14: Signals of Array1, Array3, older Array
# -----------------------------------------------------------------------------------------------------------------
folder_path = Path('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/matrix_19062024/')
dark_path = 'd2_1n_3s_beam_all_without_diffuser_dark.csv'

for crit_num, crit in enumerate(['5s_flat_calib_']):
    excluded = []

    files = os.listdir(folder_path)
    files = array_txt_file_search(files, blacklist=['.png'], searchlist=[crit],
                                  file_suffix='.csv', txt_file=False)

    A_old = Analyzer((1, 64), 0.42, 0.08, ams_constant_signal_readout, standard_position)
    A_old.set_dark_measurement(folder_path, [dark_path])
    position = []
    signals = []
    for file in tqdm(files):
        # The parsing of the position out of the name and save it
        position.append(standard_position(file))
        signal = ams_constant_signal_readout(folder_path / file, A_old)['signal']
        signals.append((np.array(signal) - A_old.dark).flatten())

    # Sort the arrays by the position
    indices = np.argsort(np.array(position)[:, sp])
    signals = np.array(signals)[indices]
    position = np.array(position)[indices]

    for signal in signals:
        signal[60] = 0

    # signals = signals[:, ::-1]

    # Recalculate the positions considering the size of the diodes and thus the expected real positions
    positions = []
    for i in range(np.shape(signals)[1]):
        cache = deepcopy(position)
        cache[:, sp] = cache[:, sp] + (64-i) * (A_old.diode_size[sp] + A_old.diode_spacing[sp])
        positions.append(cache)
    positions = np.array(positions)

    # Try to detect the signal level
    threshold = ski_threshold_otsu(signals)

    # Group the positions after their recalculation to gain a grid, from which the mean calculation is meaningful
    group_distance = A_old.diode_size[sp]
    groups = group(positions[:, :, sp].flatten(), group_distance)

    # Calculate the mean for each grouped position, consider only the diode signals that were close to this position
    mean_result = []
    mean_new = []
    mean_x_new = []
    groups = np.sort(groups)
    for mean in groups:
        indices = []
        for k, channel in enumerate(np.array(positions)[:, :, sp]):
            index_min = np.argsort(np.abs(channel - mean))[0]
            if np.abs(channel[index_min] - mean) <= group_distance:
                indices.append(index_min)
            else:
                indices.append(None)
        cache = 0
        cache_new = 0
        j = 0
        j_new = 0
        for i in range(len(indices)):
            if indices[i] is not None and signals[indices[i]][i] != 0:
                cache += signals[indices[i]][i]
                j += 1
            if indices[i] is not None and signals[indices[i]][i] >= threshold:
                cache_new += signals[indices[i]][i]
                j_new += 1
        if j > 0:
            mean_result.append(cache / j)
        else:
            mean_result.append(cache)
        if j_new > 0:
            mean_new.append(cache_new / j_new)
            mean_x_new.append(mean)

    mean_result = np.array(mean_result)
    mean_x = np.array(groups)
    mean_x_new, mean_new = np.array(mean_x_new), np.array(mean_new)

storage.append(['5s_flat_calib_', A_old, (mean_x_new, mean_new), (positions, signals), threshold, np.load(Path("/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/matrix_19062024/['5s_flat_calib_']_normalization_factor.npy"))[0]])
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
fig, ax = plt.subplots()
diode_cmap = sns.color_palette("coolwarm", as_cmap=True)
diode_colormapper = lambda diode: color_mapper(diode, 0, instance.diode_dimension[sp])
diode_color = lambda diode: diode_cmap(diode_colormapper(diode))

c = sns.color_palette("tab10")
for i, stored_data in enumerate(storage):
    signals = stored_data[3][1]
    positions = stored_data[3][0]
    threshold = stored_data[4]
    factor = stored_data[-1]
    for diode in range(stored_data[1].diode_dimension[sp]):
        above_threshold = (signals[:, diode] > threshold)
        try:
            ax.plot(positions[diode, above_threshold, 1], signals[above_threshold, diode] * factor[diode],
                    c=c[i], zorder=1, alpha=0.2)
        except np.core._exceptions._UFuncNoLoopError:
            pass
        ax.plot(positions[diode, above_threshold, 1], signals[above_threshold, diode], c='grey', zorder=0, alpha=0.05)
    ax.plot(*stored_data[2], c=c[i], zorder=-2, alpha=1, label=stored_data[0])
    ax.plot(*stored_data[2], c='k', zorder=2, alpha=1)

ax.set_xlabel('Real position of diodes during measurement (mm)')
ax.set_ylabel('Diode signal (a.u.)')
ax.set_xlim(ax.get_xlim())
ax.set_ylim(ax.get_ylim())

format_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/NormMethod/', 'ALL_14_SignalComparison',
            legend=True)
# -----------------------------------------------------------------------------------------------------------------