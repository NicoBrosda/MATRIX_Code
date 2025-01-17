import matplotlib.pyplot as plt
import numpy as np

from EvaluationSoftware.main import *


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
          ' steps at a mean step width of', step_width, 'mm for a diode periodicity of', diode_periodicity, 'mm.')

    # Sort the arrays by the position
    indices = np.argsort(np.array(position)[:, sp])
    signals = np.array(signals)[indices]
    position = np.array(position)[indices]

    # ------------------------------------------------------------------------------------------------------------------
    fig2, ax2 = plt.subplots()
    diode_cmap = sns.color_palette("coolwarm", as_cmap=True)
    diode_colormapper = lambda diode: color_mapper(diode, 0, instance.diode_dimension[sp])
    diode_color = lambda diode: diode_cmap(diode_colormapper(diode))
    if instance.diode_dimension[1-sp] > 5:
        line_cmap = sns.color_palette("crest", as_cmap=True)
        line_colormapper = lambda line: color_mapper(line, 0, instance.diode_dimension[1-sp])
        c_cyc = lambda line: line_cmap(line_colormapper(line))
        c_cyc = [c_cyc(line) for line in range(instance.diode_dimension[1-sp])]
        csc = 1.2
        c2_cyc = [(color[0]/csc, color[1]/csc, color[2]/csc) for color in c_cyc]
        legend = False
    else:
        c_cyc = sns.color_palette("tab10")
        c2_cyc = sns.color_palette("dark")
        legend = True
    # ------------------------------------------------------------------------------------------------------------------

    # Main loop: For each diode line orthogonal to translation region calculate a factor
    factor_new = np.zeros(instance.diode_dimension)
    mean_cache = []
    for line in range(instance.diode_dimension[1-sp]):
        # Recalculate the positions considering the geometry of the diode array
        positions = []
        if sp == 0:
            # iteration = range(instance.diode_dimension[sp])[::-1]
            iteration = range(instance.diode_dimension[sp])
        else:
            iteration = range(instance.diode_dimension[sp])
        for i in iteration:
            cache = deepcopy(position)
            cache[:, sp] = cache[:, sp] + (i - (instance.diode_dimension[sp]-1)/2) * (instance.diode_size[sp] + instance.diode_spacing[sp]) + instance.diode_offset[1-sp][line]
            print((i - (instance.diode_dimension[sp]-1)/2) * (instance.diode_size[sp] + instance.diode_spacing[sp]) + instance.diode_offset[1-sp][line])
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
        format_save(results_path, name+'DiodesAndMean_'+'line'+str(line), legend=False, fig=fig, axes=[ax])
        # ------------------------------------------------------------------------------------------------------------------
        for i in range(instance.diode_dimension[sp]):
            ax2.plot(positions[i, :, sp], signals[:, line, i], ls='-', c=c2_cyc[line], zorder=1, alpha=0.2)
        ax2.plot(mean_x_new, mean_new, ls='-', color=c_cyc[line], zorder=3, label=r'Mean of diode line $\#$' + str(line))
    ax2.set_xlabel('Real position of diodes during measurement (mm)')
    ax2.set_ylabel('Diode signal (a.u.)')
    ax2.set_xlim(ax2.get_xlim())
    ax2.set_ylim(ax2.get_ylim())
    format_save(results_path, name + 'DiodesAndMean', legend=legend, fig=fig2,
                axes=[ax2])
    ax2.set_ylim(mean_over*0.9, mean_over*1.1)
    ax2.set_xlim(min(mean_x_new)*0.97, max(mean_x_new)*1.03)
    format_save(results_path, name + 'DiodesAndMeanZoom', legend=legend, fig=fig2,
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
    format_save(results_path, name + 'FactorLine', legend=legend, fig=fig,
                axes=[ax])
    ax.set_ylim(0.92, 1.08)
    format_save(results_path, name + 'FactorLine_Zoom', legend=legend, fig=fig,
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
        format_save(results_path, name + 'FactorGlobal', legend=legend, fig=fig,
                    axes=[ax])
        ax.set_ylim(0.92, 1.08)
        format_save(results_path, name + 'FactorGlobal_Zoom', legend=legend, fig=fig,
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
        format_save(results_path, name + 'FactorComp', legend=legend, fig=fig,
                    axes=[ax])
        ax.set_ylim(0.92, 1.08)
        format_save(results_path, name + 'FactorComp_Zoom', legend=legend, fig=fig,
                    axes=[ax])
        # --------------------------------------------------------------------------------------------------------------

    return factor_new


mapping = Path('../../Files/mapping.xlsx')
data = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:])-1 for k in data['direction_2']]
direction1 = pd.read_excel(mapping, header=1)
direction1 = np.array([int(k[-3:]) for k in direction1['direction_1']])
direction2 = pd.read_excel(mapping, header=1)
direction2 = np.array([int(k[-3:]) for k in direction2['direction_2']])

mapping = Path('../../Files/Mapping_BigMatrix_2.xlsx')
data2 = pd.read_excel(mapping, header=None)
mapping_map = data2.to_numpy().flatten()
mapping_large2 = np.array([direction2[np.argwhere(direction1 == i)[0][0]]-1 for i in mapping_map])

mapping = Path('../../Files/Mapping_MatrixArray.xlsx')
data2 = pd.read_excel(mapping, header=None)
mapping_map = data2.to_numpy().flatten()
mapping_large1 = np.array([direction2[np.argwhere(direction1 == i)[0][0]]-1 for i in mapping_map])

mapping = Path('../../Files/Mapping_SmallMatrix1.xlsx')
data2 = pd.read_excel(mapping, header=None)
mapping_map = data2.to_numpy().flatten()
mapping_small1 = np.array([direction2[np.argwhere(direction1 == i)[0][0]]-1 for i in mapping_map])

path_1110 = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_111024/')
matrix_1110 = ['2DLarge_YScan_', '2DLarge_XScan_']

path_2110 = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_211024/')
matrix_2110 = ['2D_Mini_YScan_', '2D_Mini_YScanAfter_']

path_2210 = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
matrix_2210 = ['2Line_YScan', '2DLarge_YTranslation_']

all_matrix = matrix_1110 + matrix_2110 + matrix_2210
for i, crit in enumerate(all_matrix[0:]):
    results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Results_221024/NormMethod/')
    if i < len(matrix_1110):
        folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_111024/')
    elif i < len(matrix_1110) + len(matrix_2110):
        folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_211024/')
    else:
        folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')

    if '2Line' in crit:
        readout, position_parser = lambda x, y: ams_2line_readout(x, y, channel_assignment=channel_assignment), standard_position
        results_path = results_path / (crit + '/')
        dark_path = folder_path
        dark = ['2Line_DarkVoltageScan_200_ um_0_nA_nA_1.9_x_22.0_y_66.625.csv']
        A = Analyzer((2, 64), (0.4, 0.4), (0.1, 0.1), readout=readout, diode_offset=[[0, - 0.25], np.zeros(64)],
                     position_parser=standard_position)
    elif '2D_Mini_' in crit:
        readout, position_parser = lambda x, y: ams_2D_assignment_readout(x, y, channel_assignment=mapping_small1), standard_position
        results_path = results_path / (crit + '/')
        dark_path = folder_path
        dark = ['2D_Mini_Dark_VoltageLinearity_200_um_0_nA_nA_1.9_x_22.0_y_71.25.csv']
        A = Analyzer((11, 11), 0.4, 0.1, readout=readout)
    elif '_111024' in str(folder_path):
        readout, position_parser = lambda x, y: ams_2D_assignment_readout(x, y, channel_assignment=mapping_large1), standard_position
        results_path = results_path / ('Large1_' + crit + '/')
        dark_path = folder_path
        dark = ['2DLarge_dark_200_um_0_nA__nA_1.9_x_21.0_y_70.35.csv']
        A = Analyzer((11, 11), 0.8, 0.2, readout=readout)
    else:
        readout, position_parser = lambda x, y: ams_2D_assignment_readout(x, y, channel_assignment=mapping_large2), standard_position
        results_path = results_path / ('Large2_' + crit + '/')
        dark_path = folder_path
        dark = ['2DLarge_DarkVoltage_200_ um_0_nA_nA_1.9_x_44.0_y_66.625.csv']
        A = Analyzer((11, 11), 0.8, 0.2, readout=readout)

    A.set_dark_measurement(dark_path, dark)
    A.set_measurement(folder_path, crit)
    name = crit
    factor = normalization_from_translated_array_v3(A.measurement_files, A, align_lines=True)