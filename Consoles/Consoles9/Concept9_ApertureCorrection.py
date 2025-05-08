import SimpleITK as sitk
import h5py
from EvaluationSoftware.main import *
from EvaluationSoftware.readout_modules import ams_channel_assignment_readout
from EvaluationSoftware.normalization_modules import normalization_from_translated_array
from scipy.constants import e
from scipy.optimize import curve_fit

fast_mode = True

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
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Results_260325/Concept_ApertureCorrection/')
results_stem = Path('/Users/nico_brosda/Cyrce_Messungen/Results_260325/')

cmap=sns.color_palette("rocket_r", as_cmap=True)

new_measurements = []
aperture_measurements = []

# Used Bragg wedge measurements
diff_400um =  ([f'exp{44}_PEEKwedge_P{0}_'] + [f'exp{i+46}_PEEKwedge_P{i+1}_' for i in range(17)] +
               [f'exp{45}_PEEKwedge_P{18}_'])
diff_200um= ([f'exp{i+99}_PEEKWedge_P{i}_' for i in range(18)] + [f'exp{98}_PEEKWedge_P{18}_'] +
             [f'exp{117}_PEEKWedge_P{19}_'])
diff_200um_middle = [f'exp{i+118}_PEEKWedgeMiddle_P{19-i}_' for i in range(8)][::-1]
new_measurements += diff_400um
new_measurements += diff_200um
new_measurements += diff_200um_middle

# Also load round aperture scans for correction purposes
diff_400um_aperture = [f'exp{i+14}_energydiffmap_P{i}_' for i in range(19)]
diff_200um_aperture= [f'exp{i+77}_energyDep_P{i}_' for i in range(19)] + ['exp118_PEEKWedgeMiddle_P19_']
new_measurements += diff_400um_aperture
new_measurements += diff_200um_aperture

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

try:
    aperture_400 = np.load(results_stem / 'Fast_Mode/aperture_400.npz', allow_pickle=True)['arr_0']
    aperture_200 = np.load(results_stem / 'Fast_Mode/aperture_200.npz', allow_pickle=True)['arr_0']
    wedge_400 = np.load(results_stem / 'Fast_Mode/wedge_400.npz', allow_pickle=True)['arr_0']
    wedge_200 = np.load(results_stem / 'Fast_Mode/wedge_200.npz', allow_pickle=True)['arr_0']
    wedge_200_middle = np.load(results_stem / 'Fast_Mode/wedge_200_middle.npz', allow_pickle=True)['arr_0']

    A = Analyzer((2, 64), (0.4, 0.4), (0.1, 0.1), readout=readout,
                 diode_offset=[[0, - 0.25], np.zeros(64)], position_parser=position_parser,
                 voltage_parser=voltage_parser, current_parser=current_parser)
except FileNotFoundError as _error:
    print(_error)
    aperture_400 = []
    aperture_200 = []
    wedge_400 = []
    wedge_200 = []
    wedge_200_middle = []

    for k, crit in enumerate(new_measurements[0:]):
        print('-' * 50)
        print(crit)
        print('-' * 50)

        if k < len(diff_400um):
            diffuser = 400
        elif k < len(diff_400um) + len(diff_200um):
            diffuser = 200
        elif k < len(diff_400um) + len(diff_200um) + len(diff_200um_middle):
            diffuser = 200
        elif k < len(diff_400um) + len(diff_200um) + len(diff_200um_middle) + len(diff_400um_aperture):
            diffuser = 400
        else:
            diffuser = 200

        wheel_position = int(crit[crit.rindex('_P') + 2:crit.rindex('_')])

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

        # -------------- Save signal map to a cache list --------------
        save_obj = A.maps[0]
        save_obj['wheel_position'] = wheel_position
        if k < len(diff_400um):
            wedge_400.append(save_obj)
        elif k < len(diff_400um) + len(diff_200um):
            wedge_200.append(save_obj)
        elif k < len(diff_400um) + len(diff_200um) + len(diff_200um_middle):
            wedge_200_middle.append(save_obj)
        elif k < len(diff_400um) + len(diff_200um) + len(diff_200um_middle) + len(diff_400um_aperture):
            aperture_400.append(save_obj)
        else:
            aperture_200.append(save_obj)

    if fast_mode:
        np.savez(results_stem / 'Fast_Mode/aperture_400.npz', aperture_400)
        np.savez(results_stem / 'Fast_Mode/aperture_200.npz', aperture_200)
        np.savez(results_stem / 'Fast_Mode/wedge_400.npz', wedge_400)
        np.savez(results_stem / 'Fast_Mode/wedge_200.npz', wedge_200)
        np.savez(results_stem / 'Fast_Mode/wedge_200_middle.npz', wedge_200_middle)

# ----------------------------------------------------------------------------------------------------------------
# Calculate normed responses (normed to incoming protons)
#----------------------------------------------------------------------------------------------------------------
rescale_current = 1e6
scale_current = 'a'
currents_400_aperture = np.array([887, 888, 885, 880, 876, 872, 884, 880, 876, 871, 888, 887, 884, 881, 881, 877, 882, 880, 879]) * 1e-12 / e / rescale_current
currents_200_aperture = np.array([1.73, 1.72, 1.72, 1.70, 1.71, 1.70, 1.72, 1.71, 1.70, 1.72, 1.72, 1.71, 1.70, 1.72, 1.72, 1.71, 1.70, 1.69, 1.69, 1.76]) *1e-9 / e /rescale_current

factor_200 = []
for i in range(len(currents_200_aperture)):
    aperture_200[i]['z'] = aperture_200[i]['z'] / currents_200_aperture[i]
    factor_200.append(aperture_200[i])
    factor_200[i]['z'] = factor_200[i]['z'] / np.max(factor_200[i]['z'])

factor_400 = []
for i in range(len(currents_400_aperture)):
    aperture_400[i]['z'] = aperture_400[i]['z'] / currents_400_aperture[i]
    factor_400.append(aperture_400[i])
    factor_400[i]['z'] = factor_400[i]['z'] / np.max(factor_400[i]['z'])

rescale_current = 1e6
scale_current = 'a'
# Correctly ordered from P0 (or P12) increasing
currents_400 = np.array([882.5, 879, 877.5, 877, 874.5, 874, 872, 876.5, 876, 875.5, 873.5, 873.5, 872.5, 872, 870.5, 888, 888.5, 888, 881]) * 1e-12 / e / rescale_current
currents_200 = np.array([1.74, 1.736, 1.7295, 1.7245, 1.7195, 1.715, 1.7115, 1.7085, 1.710, 1.7345, 1.726, 1.735, 1.7335, 1.731, 1.7265, 1.726, 1.725, 1.7225, 1.7455, 1.7225]) * 1e-9 / e / rescale_current
currents_200_middle = np.array([1.7815, 1.781, 1.778, 1.778, 1.777, 1.7765, 1.7745, 1.774]) * 1e-9 / e / rescale_current

for i in range(len(currents_200)):
    wedge_200[i]['z'] = wedge_200[i]['z'] / currents_200[i]

for i in range(len(currents_400)):
    wedge_400[i]['z'] = wedge_400[i]['z'] / currents_400[i]

for i in range(len(currents_200_middle)):
    wedge_200_middle[i]['z'] = wedge_200_middle[i]['z'] / currents_200_middle[i]

#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------

def correct_wedge_with_aperture(wheel_position, map_set='400', threshold=0.1):
    """
    Corrects a wedge map by dividing by aperture map values at matching coordinates.

    Args:
        wheel_position: Index to access the specific position in the map lists
        map_set: Which map set to use ('400', '200', or '200_middle')
        threshold: Signal threshold for aperture map (default: 0.1)

    Returns:
        Dictionary with 'z', 'x', 'y', 'position' keys for corrected wedge map
    """
    # Select the appropriate maps based on the set parameter
    if map_set == '400':
        wedge_map = wedge_400[wheel_position]
        aperture_map = factor_400[wheel_position]
    elif map_set == '200':
        wedge_map = wedge_200[wheel_position]
        aperture_map = factor_200[wheel_position]
    elif map_set == '200_middle':
        wedge_map = wedge_200_middle[wheel_position - 12]
        aperture_map = factor_200[wheel_position]  # Note: using aperture_200 for 200_middle
    else:
        raise ValueError(f"Unknown map set: {map_set}. Use '400', '200', or '200_middle'")

    if (map_set == '200_middle' or map_set == '200') and wheel_position == 19:
        print(f"No valid aperture map for {map_set} at position {wheel_position} existent - Returning a zero map")
        return {'z': np.zeros(wedge_map['z'].shape), 'x': wedge_map['x'], 'y': wedge_map['y'],
                'position': wedge_map['position']}

    # Extract data and coordinates from input maps
    wedge_data = wedge_map['z']
    x_wedge = wedge_map['x']
    y_wedge = wedge_map['y']

    aperture_data = aperture_map['z']
    x_aperture = aperture_map['x']
    y_aperture = aperture_map['y']

    # Create result dictionary with same structure as input
    corrected_wedge_map = deepcopy(wedge_map)
    corrected_wedge_data = corrected_wedge_map['z']

    # Check if both datasets have identical coordinate systems
    if np.array_equal(x_wedge, x_aperture) and np.array_equal(y_wedge, y_aperture):
        # Fast path: direct vectorized operation when coordinates match exactly
        print(f"Using fast path for {map_set} at position {wheel_position}")

        # Create mask for valid aperture signals
        valid_aperture = aperture_data > threshold

        # Apply correction only where aperture signal is valid
        # and avoid division by zero
        valid_division = np.logical_and(valid_aperture, aperture_data != 0)
        corrected_wedge_data[valid_division] = wedge_data[valid_division] / aperture_data[valid_division]

        # Set values to zero where aperture signal is below threshold
        corrected_wedge_data[~valid_division] = 0  # or keep original with: wedge_data[~valid_division]
    else:
        # Slower path: process only common coordinates
        print(f"Using slow path for {map_set} at position {wheel_position}")

        # Find common coordinates
        common_x = np.intersect1d(x_wedge, x_aperture)
        common_y = np.intersect1d(y_wedge, y_aperture)

        # Process only points that exist in both datasets
        for x in common_x:
            for y in common_y:
                # Find indices for this point in both arrays
                i_wedge = np.where(x_wedge == x)[0]
                j_wedge = np.where(y_wedge == y)[0]

                i_aperture = np.where(x_aperture == x)[0]
                j_aperture = np.where(y_aperture == y)[0]

                # Check if the point exists in both datasets and indices are valid
                if (i_wedge.size > 0 and j_wedge.size > 0 and
                        i_aperture.size > 0 and j_aperture.size > 0):
                    i_w, j_w = i_wedge[0], j_wedge[0]
                    i_a, j_a = i_aperture[0], j_aperture[0]

                    # Ensure indices are within bounds
                    if (i_w < wedge_data.shape[1] and j_w < wedge_data.shape[0] and
                            i_a < aperture_data.shape[1] and j_a < aperture_data.shape[0]):
                        # Apply correction logic
                        if aperture_data[j_a, i_a] > threshold and aperture_data[j_a, i_a] != 0:
                            corrected_wedge_data[j_w, i_w] = wedge_data[j_w, i_w] / aperture_data[j_a, i_a]
                        else:
                            # No sufficient signal in aperture map or division by zero
                            corrected_wedge_data[j_w, i_w] = 0  # or keep original with: wedge_data[i_w, j_w]
                    else:
                        pass
                        # print(f"Warning: Index out of bounds: wedge[{i_w},{j_w}], aperture[{i_a},{j_a}]")

    return corrected_wedge_map

# Example usage with wheel position and map set:
# corrected_map = correct_wedge_with_aperture(wheel_position=5, map_set='400')
A.scale = 'atto'
# threshold = 0.4

def plot_process(save_folder='Process', map_set='400', wheel_position=0, threshold=0.1, plot_corrected_map=False):
    save_path = results_path / f'{save_folder}_{map_set}/'
    if map_set == '400':
        wedge = wedge_400[wheel_position]
        factor = deepcopy(factor_400[wheel_position])
        crit = diff_400um[wheel_position]
    elif map_set == '200':
        wedge = wedge_200[wheel_position]
        factor = deepcopy(factor_200[wheel_position])
        crit = diff_200um[wheel_position]
    elif map_set == '200_middle':
        wedge = wedge_200_middle[wheel_position-12]
        factor = deepcopy(factor_200[wheel_position])
        crit = diff_200um_middle[wheel_position-12]
    else:
        raise ValueError(f"Unknown map set: {map_set}. Use '400', '200', or '200_middle'")

    corrected = correct_wedge_with_aperture(wheel_position, map_set=map_set, threshold=threshold)

    # ----------------------------------------------------------------------------------------------------------------
    # Plot factor map and histogram
    # ----------------------------------------------------------------------------------------------------------------
    fig, (ax, ax2) = plt.subplots(1, 2)
    A.maps[0]['x'], A.maps[0]['y'], A.maps[0]['z'] = factor['y'], factor['x'], deepcopy(factor['z'].T)
    A.maps[0]['z'][A.maps[0]['z'] < threshold] = 0
    A.plot_map(None, pixel='fill', ax_in=ax, fig_in=fig, cmap=sns.color_palette("Greys", as_cmap=True),
               alpha=1, colorbar=True)
    ax2.hist(factor['z'].flatten(), bins=100)
    ax2.axvline(x=threshold, color='r', linestyle='--')
    plot_size = (2 * latex_textwidth, latex_textwidth / 1.2419)
    format_save(save_path, save_name=f'FactorMap_{crit}{threshold: .2f}', save=True, legend=False, fig=fig, plot_size=plot_size)

    # ----------------------------------------------------------------------------------------------------------------
    # Plot factor map overlay / wedge overlay
    # ----------------------------------------------------------------------------------------------------------------
    fig, (ax, ax2) = plt.subplots(1, 2)
    A.maps[0]['x'], A.maps[0]['y'], A.maps[0]['z'] = factor['y'], factor['x'], factor['z'].T
    A.plot_map(None, pixel='fill', ax_in=ax, fig_in=fig, cmap=cmap)
    A.maps[0]['x'], A.maps[0]['y'], A.maps[0]['z'] = factor['y'], factor['x'], deepcopy(factor['z'].T)
    A.maps[0]['z'][A.maps[0]['z'] < threshold] = 0
    A.maps[0]['z'][A.maps[0]['z'] > threshold] = 1
    A.plot_map(None, pixel='fill', ax_in=ax, fig_in=fig, cmap=sns.color_palette("Greys", as_cmap=True), alpha=0.5,
               colorbar=False)
    A.maps[0]['x'], A.maps[0]['y'], A.maps[0]['z'] = wedge['y'], wedge['x'], wedge['z'].T
    A.plot_map(None, pixel='fill', ax_in=ax2, fig_in=fig, cmap=cmap)
    A.maps[0]['x'], A.maps[0]['y'], A.maps[0]['z'] = factor['y'], factor['x'], deepcopy(factor['z'].T)
    A.maps[0]['z'][A.maps[0]['z'] < threshold] = 0
    A.plot_map(None, pixel='fill', ax_in=ax2, fig_in=fig, cmap=sns.color_palette("Greys", as_cmap=True), alpha=0.5,
               colorbar=False)
    plot_size = (2 * latex_textwidth, latex_textwidth / 1.2419)
    format_save(save_path, save_name=f'OverlayComp_{crit}{threshold: .2f}', save=True, legend=False, fig=fig,
                plot_size=plot_size)

    # ----------------------------------------------------------------------------------------------------------------
    # Plot Corrected map and histogram
    # ----------------------------------------------------------------------------------------------------------------
    fig, (ax, ax2) = plt.subplots(1, 2)
    A.maps[0]['x'], A.maps[0]['y'], A.maps[0]['z'] = corrected['y'], corrected['x'], corrected['z'].T
    A.plot_map(None, pixel='fill', ax_in=ax, fig_in=fig, cmap=cmap, alpha=1, colorbar=True)

    ax2.hist(factor['z'].flatten(), bins=100)
    ax2.axvline(x=threshold, color='r', linestyle='--')
    plot_size = (2 * latex_textwidth, latex_textwidth / 1.2419)
    format_save(save_path, save_name=f'CorrectedHist_{crit}{threshold: .2f}', save=True, legend=False, fig=fig, plot_size=plot_size)

    # ----------------------------------------------------------------------------------------------------------------
    # Plot factor map/wedge overlay and corrected map
    # ----------------------------------------------------------------------------------------------------------------
    fig, (ax, ax2) = plt.subplots(1, 2)
    A.maps[0]['x'], A.maps[0]['y'], A.maps[0]['z'] = wedge['y'], wedge['x'], wedge['z'].T
    A.plot_map(None, pixel='fill', ax_in=ax, fig_in=fig, cmap=cmap)
    A.maps[0]['x'], A.maps[0]['y'], A.maps[0]['z'] = factor['y'], factor['x'], deepcopy(factor['z'].T)
    A.maps[0]['z'][A.maps[0]['z'] < threshold] = 0
    A.plot_map(None, pixel='fill', ax_in=ax, fig_in=fig, cmap=sns.color_palette("Greys", as_cmap=True), alpha=0.5,
               colorbar=False)
    A.maps[0]['x'], A.maps[0]['y'], A.maps[0]['z'] = corrected['y'], corrected['x'], corrected['z'].T
    A.plot_map(None, pixel='fill', ax_in=ax2, fig_in=fig, cmap=cmap, alpha=1, colorbar=True)
    plot_size = (2 * latex_textwidth, latex_textwidth / 1.2419)
    format_save(save_path, save_name=f'OverlayFactor_{crit}{threshold: .2f}', save=True, legend=False, fig=fig, plot_size=plot_size)

    # ----------------------------------------------------------------------------------------------------------------
    # Plot factor map region/wedge overlay and corrected map
    # ----------------------------------------------------------------------------------------------------------------
    fig, (ax, ax2) = plt.subplots(1, 2)
    A.maps[0]['x'], A.maps[0]['y'], A.maps[0]['z'] = wedge['y'], wedge['x'], wedge['z'].T
    A.plot_map(None, pixel='fill', ax_in=ax, fig_in=fig, cmap=cmap)
    A.maps[0]['x'], A.maps[0]['y'], A.maps[0]['z'] = factor['y'], factor['x'], deepcopy(factor['z'].T)
    A.maps[0]['z'][A.maps[0]['z'] < threshold] = 0
    A.maps[0]['z'][A.maps[0]['z'] > threshold] = 1
    A.plot_map(None, pixel='fill', ax_in=ax, fig_in=fig, cmap=sns.color_palette("Greys", as_cmap=True), alpha=0.5, colorbar=False)

    A.maps[0]['x'], A.maps[0]['y'], A.maps[0]['z'] = corrected['y'], corrected['x'], corrected['z'].T
    A.plot_map(None, pixel='fill', ax_in=ax2, fig_in=fig, cmap=cmap, alpha=1,
               colorbar=True)
    plot_size = (2 * latex_textwidth, latex_textwidth / 1.2419)
    format_save(save_path, save_name=f'OverlayRegion_{crit}{threshold: .2f}', save=True, legend=False, fig=fig, plot_size=plot_size)

    # ----------------------------------------------------------------------------------------------------------------
    # Plot Comparison Before/After Correction
    # ----------------------------------------------------------------------------------------------------------------
    fig, (ax, ax2) = plt.subplots(1, 2)
    A.maps[0]['x'], A.maps[0]['y'], A.maps[0]['z'] = wedge['y'], wedge['x'], wedge['z'].T
    A.plot_map(None, pixel='fill', ax_in=ax, fig_in=fig, cmap=cmap)
    A.maps[0]['x'], A.maps[0]['y'], A.maps[0]['z'] = corrected['y'], corrected['x'], corrected['z'].T
    A.plot_map(None, pixel='fill', ax_in=ax2, fig_in=fig, cmap=cmap, alpha=1, colorbar=True)
    plot_size = (2 * latex_textwidth, latex_textwidth / 1.2419)
    format_save(save_path, save_name=f'BeforeAfter_{crit}{threshold: .2f}', save=True, legend=False, fig=fig, plot_size=plot_size)

    if plot_corrected_map:
        fig, ax = plt.subplots()
        A.maps[0]['x'], A.maps[0]['y'], A.maps[0]['z'] = corrected['y'], corrected['x'], corrected['z'].T
        A.plot_map(None, pixel='fill', ax_in=ax, fig_in=fig, cmap=cmap, alpha=1, colorbar=True)
        plot_size = (1 * latex_textwidth, latex_textwidth / 1.2419)
        format_save(results_path / f'{map_set}_corrected/', save_name=f'Corrected_{crit}{threshold: .2f}', save=True, legend=False,
                    fig=fig, plot_size=plot_size)

# threshold = 0.5
for threshold in np.linspace(0.05, 0.8, 16):
# for threshold in [0.1]:
    plot_process(save_folder='ThresholdComp_P1', map_set='200', wheel_position=1, threshold=threshold, plot_corrected_map=False)
    plot_process(save_folder='ThresholdComp_P16', map_set='200', wheel_position=16, threshold=threshold, plot_corrected_map=False)
    plot_process(save_folder='ThresholdComp_P1', map_set='400', wheel_position=1, threshold=threshold, plot_corrected_map=False)
    plot_process(save_folder='ThresholdComp_P16', map_set='400', wheel_position=16, threshold=threshold, plot_corrected_map=False)
    plot_process(save_folder='ThresholdComp_P12', map_set='200_middle', wheel_position=12, threshold=threshold, plot_corrected_map=False)
    plot_process(save_folder='ThresholdComp_P18', map_set='200_middle', wheel_position=18, threshold=threshold, plot_corrected_map=False)

threshold = 0.3
for wheel_position in range(19):
    plot_process(save_folder=f'Process{threshold: .2f}', map_set='200', wheel_position=wheel_position, threshold=threshold, plot_corrected_map=True)
    plot_process(save_folder=f'Process{threshold: .2f}', map_set='400', wheel_position=wheel_position, threshold=threshold, plot_corrected_map=True)
    if wheel_position >= 12:
        plot_process(save_folder=f'Process{threshold: .2f}', map_set='200_middle', wheel_position=wheel_position, threshold=threshold, plot_corrected_map=True)

