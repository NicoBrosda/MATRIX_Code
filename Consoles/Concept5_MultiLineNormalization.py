import matplotlib.pyplot as plt
import numpy as np

from EvaluationSoftware.main import *

mapping = Path('../Files/mapping.xlsx')
data = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:])-1 for k in data['direction_2']]
readout, position_parser = lambda x, y: ams_2line_readout(x, y, channel_assignment=channel_assignment), standard_position

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Results_221024/')

new_measurements = ['_GafComp200_', '_GafComp400_', '_GafComp40_', '_GafCompLogo_', '_GafCompMisc_', '_GafCompPEEK_',
                    '_MouseFoot_', '_MouseFoot2_', '2Line_Beam_']
live_scan_array1 = [str(round(i+1, 0))+'_live1_' for i in range(9)]
# new_measurements_array_matrix = ['']

dark_path = Path('/Users/nico_brosda//Cyrce_Messungen/matrix_221024/')

dark = ['2Line_DarkVoltageScan_200_ um_0_nA_nA_1.9_x_22.0_y_66.625.csv']

norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
norm_array1 = ['2Line_YScan_']
name = norm_array1[0]


A = Analyzer((2, 64), (0.4, 0.4), (0.1, 0.1), readout=readout, diode_offset=[[0, - 0.25], np.zeros(64)], position_parser=standard_position)
A.set_dark_measurement(dark_path, dark)
A.set_measurement(norm_path, norm_array1)


def normalization_from_translated_array_v3(list_of_files, instance, method='least_squares', align_lines=True):
    # Load in the data from a list of files in a folder; save position and signal
    position = []
    signals = []
    for file in tqdm(list_of_files):
        # The parsing of the position out of the name and save it
        position.append(instance.pos_parser(file))
        signal = instance.readout(file, instance)['signal']
        signals.append((np.array(signal) - instance.dark))

    # Direction of the measurement: x or y
    x_pos = np.sort(np.array(list(set([i[0] for i in position]))))
    y_pos = np.sort(np.array(list(set([i[1] for i in position]))))
    pos = [x_pos, y_pos]

    # Choose the direction that is considered for normalisation
    if len(x_pos) > len(y_pos):
        sp = 0
    elif len(y_pos) > len(x_pos):
        sp = 1
    else:
        sp = 1

    # If multiple positions in the other directions are given, this version filters for the step with the largest number
    # of translation steps
    if len(pos[1-sp]) > 1:
        pos_choice = []
        for i in pos[1-sp]:
            pos_choice.append(len(np.array(list(set([j[sp] for j in position if j[1-sp] == i]))).sort()))
        choice = x_pos[np.argmax([pos_choice])]
        signals = [sig for i, sig in enumerate(signals) if position[i][1-sp] == choice]
        position = [i for i in position if i[1-sp] == choice]
        pos[sp] = np.array(list(set([i[sp] for i in position]))).sort()
        pos[1-sp] = np.array([choice])

    # Stop normalization if some easy testable conditions are not met (and normalization is not possible)
    if len(pos[0]) == 1 and len(pos[1]) == 1:
        print('This measurement is not usable for this normalization since there is effectively no translation'
              ' - thus, no factor can be calculated!')
        return None
    if instance.diode_dimension[sp] == 1:
        print('This measurement is not usable for this normalization since no diode overlay will exist for the given '
              'translation - thus, no factor can be calculated!')
        return None

    # Some info about steps and step width
    steps = len(pos[sp])
    step_width = np.mean([pos[sp][i+1] - pos[sp][i] for i in range(steps-1)])
    diode_periodicity = instance.diode_size[sp] + instance.diode_spacing[sp]
    if step_width > instance.diode_dimension[sp]*diode_periodicity:
        print('This measurement is not usable for a normalization since no diode overlay will exist - '
              'thus, no factor can be calculated!')
        return None

    print('Normalization is calculated from translation direction', ['x', 'y'][sp], 'with', steps,
          'at a mean step width of', step_width, 'mm for a diode periodicity of', diode_periodicity, 'mm.')

    # Sort the arrays by the position
    indices = np.argsort(np.array(position)[:, sp])
    signals = np.array(signals)[indices]
    position = np.array(position)[indices]

    # ------------------------------------------------------------------------------------------------------------------
    fig2, ax2 = plt.subplots()
    diode_cmap = sns.color_palette("coolwarm", as_cmap=True)
    diode_colormapper = lambda diode: color_mapper(diode, 0, instance.diode_dimension[sp])
    diode_color = lambda diode: diode_cmap(diode_colormapper(diode))
    c_cyc = sns.color_palette("tab10")
    c2_cyc = sns.color_palette("dark")
    # ------------------------------------------------------------------------------------------------------------------

    # Main loop: For each diode line orthogonal to translation region calculate a factor
    factor_new = np.zeros(instance.diode_dimension)
    mean_cache = []
    for line in range(instance.diode_dimension[1-sp]):
        # Recalculate the positions considering the geometry of the diode array
        positions = []
        for i in range(instance.diode_dimension[sp]):
            cache = deepcopy(position)
            cache[:, sp] = cache[:, sp] + i * (instance.diode_size[sp] + instance.diode_spacing[sp]) + instance.diode_offset[1-sp][line]
            positions.append(cache)
        positions = np.array(positions)

        # Try to detect the signal level
        threshold = ski_threshold_otsu(signals[:, line])
        mean_over = np.mean(signals[(signals > threshold)])

        # Group the positions after their recalculation to gain a grid, from which the mean calculation is meaningful
        group_distance = instance.diode_size[sp]
        groups = group(positions[:, :, sp].flatten(), group_distance)

        # Calculate the mean for each grouped position, consider only the diode signals that were close to this position
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
            cache_new = 0
            j_new = 0
            for i in range(len(indices)):
                if indices[i] is not None and threshold <= signals[indices[i], line, i] <= 1.5 * mean_over:
                    cache_new += signals[indices[i], line, i]
                    j_new += 1
            if j_new > 0:
                mean_new.append(cache_new / j_new)
                mean_x_new.append(mean)

        mean_x_new, mean_new = np.array(mean_x_new), np.array(mean_new)

        if align_lines:
            mean_cache.append([mean_x_new, mean_new])
        # Interpolation
        factor_cache = []
        for channel in range(instance.diode_dimension[sp]):
            restrained_position = (np.min(mean_x_new) <= positions[channel, :, sp]) & (
                        positions[channel, :, sp] <= np.max(mean_x_new))
            mean_interp_new = np.interp(positions[channel, :, sp][restrained_position], mean_x_new, mean_new)
            if isinstance(method, (float, int, np.float64)):
                # Method 1: Threshold for range consideration, for each diode channel mean of the factor between points
                factor_new_cache = mean_interp_new[signals[:, line, channel] > method] / signals[:, line, channel][signals[:, line, channel] > method]
                factor_new_cache = np.mean(factor_new_cache)
                if np.isnan(factor_new_cache):
                    factor_new_cache = 0
            elif method == 'least_squares':
                # Method 2: Optimization with least squares method, recommended
                func_opt_new = lambda a: mean_interp_new - signals[:, line, channel][restrained_position] * a
                factor_new_cache = least_squares(func_opt_new, 1)
                if factor_new_cache.nfev == 1 and factor_new_cache.optimality == 0.0:
                    factor_new_cache = 0
                else:
                    factor_new_cache = factor_new_cache.x[0]
            else:
                # Standard method: For the moment method 1 with automatic threshold
                factor_new_cache = mean_interp_new / signals[signals[:, line, channel] > threshold][:, line, channel]
                factor_new_cache = np.mean(factor_new_cache)
                if np.isnan(factor_new_cache):
                    factor_new_cache = 0

            factor_cache.append(factor_new_cache)
        factor_new[line] = np.array(factor_cache).flatten()

        # ------------------------------------------------------------------------------------------------------------------
        fig, ax = plt.subplots()
        for i in range(instance.diode_dimension[sp]):
            ax.plot(positions[i, :, sp], signals[:, line, i], c=diode_color(i), zorder=1)
        ax.plot(mean_x_new, mean_new, ls='--', color='y', zorder=3, label=r'Mean of diode line $\#$'+str(line))
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
        format_save(results_path / 'NormMethod/', name+'DiodesAndMean_'+'line'+str(line), legend=False, fig=fig, axes=[ax])
        # ------------------------------------------------------------------------------------------------------------------
        for i in range(instance.diode_dimension[sp]):
            ax2.plot(positions[i, :, sp], signals[:, line, i], ls='-', c=c2_cyc[line], zorder=1, alpha=0.2)
        ax2.plot(mean_x_new, mean_new, ls='-', color=c_cyc[line], zorder=3, label=r'Mean of diode line $\#$' + str(line))
    ax2.set_xlabel('Real position of diodes during measurement (mm)')
    ax2.set_ylabel('Diode signal (a.u.)')
    ax2.set_xlim(ax2.get_xlim())
    ax2.set_ylim(ax2.get_ylim())
    format_save(results_path / 'NormMethod/', name + 'DiodesAndMean', legend=True, fig=fig2,
                axes=[ax2])
    ax2.set_ylim(mean_over*0.9, mean_over*1.1)
    ax2.set_xlim(min(mean_x_new)*0.97, max(mean_x_new)*1.03)
    format_save(results_path / 'NormMethod/', name + 'DiodesAndMeanZoom', legend=True, fig=fig2,
                axes=[ax2])

    # ------------------------------------------------------------------------------------------------------------------
    # Plots of factors
    fig, ax = plt.subplots()
    for line in range(np.shape(factor_new)[0]):
        ax.plot(factor_new[line], c=c_cyc[line], label='Line '+str(line))
    ax.set_xlabel(r'$\#$ Diode of each line')
    ax.set_ylabel('Normalization factor (aligned to mean of line)')
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    format_save(results_path / 'NormMethod/', name + 'FactorLine', legend=True, fig=fig,
                axes=[ax])
    ax.set_ylim(0.92, 1.08)
    format_save(results_path / 'NormMethod/', name + 'FactorLine_Zoom', legend=True, fig=fig,
                axes=[ax])
    factor_line = deepcopy(factor_new)
    # ------------------------------------------------------------------------------------------------------------------

    if align_lines:
        # Calculate the mean of the mean from the different lines
        x_mean = mean_cache[0][0]
        mean_cache = [np.interp(x_mean, mean_cache[i][0], mean_cache[i][1]) for i in range(len(mean_cache))]
        overall_mean = np.mean(mean_cache)
        line_factor = []
        for line, m in enumerate(mean_cache):
            func_opt_new = lambda a: overall_mean - m * a
            factor_new_cache = least_squares(func_opt_new, 1)
            if factor_new_cache.nfev == 1 and factor_new_cache.optimality == 0.0:
                factor_new_cache = 1
            else:
                factor_new_cache = factor_new_cache.x[0]
            line_factor.append(factor_new_cache)
            factor_new[line]= factor_new[line] * factor_new_cache

        # --------------------------------------------------------------------------------------------------------------
        fig, ax = plt.subplots()
        for line in range(np.shape(factor_new)[0]):
            ax.plot(factor_new[line], c=c_cyc[line], label='Line ' + str(line))
        ax.set_xlabel(r'$\#$ Diode of each line')
        ax.set_ylabel('Normalization factor (aligned to global mean)')
        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())
        format_save(results_path / 'NormMethod/', name + 'FactorGlobal', legend=True, fig=fig,
                    axes=[ax])
        ax.set_ylim(0.92, 1.08)
        format_save(results_path / 'NormMethod/', name + 'FactorGlobal_Zoom', legend=True, fig=fig,
                    axes=[ax])

        fig, ax = plt.subplots()
        for line in range(np.shape(factor_new)[0]):
            ax.plot(factor_new[line], c=c_cyc[line], label='Global Line ' + str(line))
        for line in range(np.shape(factor_line)[0]):
            ax.plot(factor_line[line], ls='--', c=c_cyc[line], label='Individual Line ' + str(line))
        ax.set_xlabel(r'$\#$ Diode of each line')
        ax.set_ylabel('Normalization factor')
        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())
        format_save(results_path / 'NormMethod/', name + 'FactorComp', legend=True, fig=fig,
                    axes=[ax])
        ax.set_ylim(0.92, 1.08)
        format_save(results_path / 'NormMethod/', name + 'FactorComp_Zoom', legend=True, fig=fig,
                    axes=[ax])
        # --------------------------------------------------------------------------------------------------------------

    return factor_new


factor = normalization_from_translated_array_v3(A.measurement_files, A, align_lines=True)