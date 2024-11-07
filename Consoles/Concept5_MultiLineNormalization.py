import matplotlib.pyplot as plt
import numpy as np

from EvaluationSoftware.main import *

mapping = Path('../Files/mapping.xlsx')
data = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:])-1 for k in data['direction_2']]
readout, position_parser = lambda x, y: ams_2line_readout(x, y, channel_assignment=channel_assignment), standard_position

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Results_221024/2LineMaps/')

new_measurements = ['_GafComp200_', '_GafComp400_', '_GafComp40_', '_GafCompLogo_', '_GafCompMisc_', '_GafCompPEEK_',
                    '_MouseFoot_', '_MouseFoot2_', '2Line_Beam_']
live_scan_array1 = [str(round(i+1, 0))+'_live1_' for i in range(9)]
# new_measurements_array_matrix = ['']

dark_path = Path('/Users/nico_brosda//Cyrce_Messungen/matrix_221024/')

dark = ['2Line_DarkVoltageScan_200_ um_0_nA_nA_1.9_x_22.0_y_66.625.csv']

norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
norm_array1 = ['2Line_YScan_']

A = Analyzer((2, 64), (0.4, 0.4), (0.1, 0.1), readout=readout, diode_offset=[[0, - 0.25], np.zeros(64)], position_parser=standard_position)
A.set_dark_measurement(dark_path, dark)
A.set_measurement(norm_path, norm_array1)


def normalization_from_translated_array_v3(list_of_files, instance, method='least_squares'):
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

    print('Normalization is calculated from translation direction ', ['x', 'y'][sp], ' with ', steps,
          'at a mean step width of ', step_width, ' mm for a diode periodicity of ', diode_periodicity, ' mm.')

    # Sort the arrays by the position
    indices = np.argsort(np.array(position)[:, sp])
    signals = np.array(signals)[indices]
    position = np.array(position)[indices]

    # ------------------------------------------------------------------------------------------------------------------
    # Main loop: For each diode line orthogonal to translation region calculate a factor
    factor_new = np.zeros(instance.diode_dimension)
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
                if indices[i] is not None and signals[indices[i], line, i] >= threshold:
                    cache_new += signals[indices[i], line, i]
                    j_new += 1
            if j_new > 0:
                mean_new.append(cache_new / j_new)
                mean_x_new.append(mean)

        mean_x = np.array(groups)
        mean_x_new, mean_new = np.array(mean_x_new), np.array(mean_new)

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
    return factor_new


factor = normalization_from_translated_array_v3(A.measurement_files, A)

fig, ax = plt.subplots()
for line in range(np.shape(factor)[0]):
    ax.plot(factor[line])
plt.show()