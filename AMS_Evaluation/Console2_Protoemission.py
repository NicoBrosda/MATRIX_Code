import numpy as np
import scipy.signal

from read_MATRIX import *
from EvaluationSoftware.main import *
from skimage.filters import threshold_otsu as ski_threshold_otsu
from skimage import morphology

special = False
extended = False
# Extended:
if special:
    first_measurements = ['noscrew']
    new_measurements = []
elif extended:
    first_measurements = ['noscrew', 'screwsmaller_horizontal', '_screw_', 'screw8_vertical', 'screw8_horizontal_',
                          'screw8_horizontal2_']
    new_measurements = ['10s_iphcmatrixcrhea_', '5s_biseau_blanc_topolino_decal7_', '5s_biseau_blanc_topolino_nA_',
                        '5s_biseau_blanc_vide_', '5s_biseau2D_vide_nA_', '5s_misc_shapes_', '5s_topolino_thin_vide_',
                        '5s_topolino_thin_nA_', '10s_neonoff_iphcmatrixcrhea_nA_', '10s_neonoff_iphcmatrixcrhea_suite_']
else:
    first_measurements = ['noscrew', 'screwsmaller_horizontal']
    new_measurements = ['10s_iphcmatrixcrhea_', '5s_misc_shapes_']

A = Analyzer((1, 64), 0.42, 0.08)
for k, crit in enumerate(first_measurements+new_measurements):
    if k == 0:
        A.excluded[0, 36] = True
        A.readout = ams_otsus_readout
        A.pos_parser = first_measurements_position
        folder_path = Path('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/iphc_python_misc/matrix_27052024/')
        paths = ['e2_500p_bottom_nA_2.csv',
                 'e2_500p_nA_2.csv',
                 'e2_500p_top_nA_2.csv']
        A.normalization(folder_path, paths, normalization_module=simple_normalization, cache_save=False)
    if k == len(first_measurements):
        A = Analyzer((1, 64), 0.42, 0.08)
        folder_path = Path('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/matrix_19062024/')
        A.set_dark_measurement(folder_path, 'd2_1n_3s_beam_all_without_diffuser_dark.csv')
        A.normalization(folder_path, '5s_flat_calib_', normalization_module=normalization_from_translated_array)
    A.set_measurement(folder_path, crit)
    A.load_measurement()
    if k < len(first_measurements):
        A.create_map(inverse=[True, False])
        intensity_limits = (0, 1200)
    else:
        A.create_map(inverse=[False, False])
        intensity_limits = None

    A.plot_map('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_19062024/ProtoEmission/new_maps/',
               contour=True, intensity_limits=intensity_limits)
    A.plot_map('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_19062024/ProtoEmission/new_maps/',
               contour=False, intensity_limits=intensity_limits)

    # Detect the signal and no signal values in signal array
    threshold1 = threshold_otsu(A.map['z'].flatten())
    threshold2 = ski_threshold_otsu(A.map['z'])
    # print(threshold1)
    # print(threshold2)
    dark_mean = np.mean(A.map['z'][A.map['z'] < threshold2])
    signal_mean = np.mean(A.map['z'][A.map['z'] > threshold2])

    fig, ax = plt.subplots()
    if intensity_limits is not None:
        ax.set_xlim(-100, intensity_limits[1])
    ax.hist(A.map['z'].flatten(), bins=1000, color='k')
    ax.axvline(threshold1, c='b', label='Previous implementation')
    ax.axvline(threshold2, c='r', label='Scikit Image Otsus')
    ax.set_xlabel('Signal')
    format_save(Path('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_19062024/ProtoEmission/methods/'),
                'New_thresholding_'+crit, legend=True)

    # -----------------------------------------------------------------------------------------------------------------
    # Plot where edges are recognized
    A.plot_map(None, contour=False, intensity_limits=intensity_limits)
    fig, ax = plt.gcf(), plt.gca()
    # For each column: detect points where this threshold is exceeded
    for i, column in enumerate(A.map['z'].T):
        for j, el in enumerate(column):
            if j == 0:
                continue
            if column[j - 1] < threshold2 <= column[j]:
                ax.plot(A.map['x'][i]+0.125, A.map['y'][j-1]+0.25, ls='', marker='^', c='lime', markersize=1)

            if column[j - 1] >= threshold2 > column[j]:
                ax.plot(A.map['x'][i]+0.125, A.map['y'][j-1]+0.25, ls='', marker='^', c='gold', markersize=1)
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    ax.plot(0, 0, ls='', marker='^', c='lime', label='Signal in next cell is above threshold', zorder=-5)
    ax.plot(0, 0, ls='', marker='^', c='gold', label='Signal in next cell is below threshold', zorder=-5)
    format_save(Path('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_19062024/ProtoEmission/methods/'),
                'edge_detection_'+crit, legend=True)

    # -----------------------------------------------------------------------------------------------------------------
    # Plot signature recognizing:
    A.plot_map(None, contour=False, intensity_limits=intensity_limits)
    fig, ax = plt.gcf(), plt.gca()
    # Exclude columns with complete zero:
    zero_column = []
    for j, column in enumerate(A.map['z']):
        if A.map['z'][j].mean() == 0 and A.map['z'][j].std() == 0:
            zero_column.append(j)
    # For each column: detect points where this threshold is exceeded
    for i, column in enumerate(A.map['z'].T):
        for j, el in enumerate(column):
            if j == 0 or j == 1 or j == 2:
                continue
            if j in zero_column or j-1 in zero_column or j-2 in zero_column or j-3 in zero_column:
                continue
            if column[j-3] < threshold2 <= column[j-2] and column[j-2] >= threshold2 > column[j-1] \
                    and column[j-1] < threshold2 <= column[j]:
                ax.plot(A.map['x'][i] + 0.125, A.map['y'][j - 3] + 0.25, ls='', marker='x', c='lime', markersize=3)
            if column[j-3] >= threshold2 > column[j-2] and column[j-2] < threshold2 <= column[j-1] \
                    and column[j-1] >= threshold2 > column[j]:
                ax.plot(A.map['x'][i] + 0.125, A.map['y'][j - 3] + 0.25, ls='', marker='x', c='gold', markersize=3)
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    ax.plot(0, 0, ls='', marker='x', c='lime', label='Signature edge no signal to signal', zorder=-5)
    ax.plot(0, 0, ls='', marker='x', c='gold', label='Signature edge signal to no signal', zorder=-5)
    format_save(Path('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_19062024/ProtoEmission/methods'),
                'edge_signature_'+crit, legend=True)

    # -----------------------------------------------------------------------------------------------------------------
    # Plot signature recognizing after simple artefact filter:
    A.map['z'] = simple_artefact_filter(A.map['z'], threshold2)
    A.plot_map(None, contour=False, intensity_limits=intensity_limits)
    fig, ax = plt.gcf(), plt.gca()
    # Exclude columns with complete zero:
    zero_column = []
    for j, column in enumerate(A.map['z']):
        if A.map['z'][j].mean() == 0 and A.map['z'][j].std() == 0:
            zero_column.append(j)
    # For each column: detect points where this threshold is exceeded
    for i, column in enumerate(A.map['z'].T):
        for j, el in enumerate(column):
            if j == 0 or j == 1 or j == 2:
                continue
            if j in zero_column or j - 1 in zero_column or j - 2 in zero_column or j - 3 in zero_column:
                continue
            if column[j - 3] < threshold2 <= column[j - 2] and column[j - 2] >= threshold2 > column[j - 1] and column[
                j - 1] < threshold2 <= column[j]:
                ax.plot(A.map['x'][i] + 0.125, A.map['y'][j - 3] + 0.25, ls='', marker='x', c='lime', markersize=3)
            if column[j - 3] >= threshold2 > column[j - 2] and column[j - 2] < threshold2 <= column[j - 1] and column[
                j - 1] >= threshold2 > column[j]:
                ax.plot(A.map['x'][i] + 0.125, A.map['y'][j - 3] + 0.25, ls='', marker='x', c='gold', markersize=3)
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    ax.plot(0, 0, ls='', marker='x', c='lime', label='Signature edge no signal to signal', zorder=-5)
    ax.plot(0, 0, ls='', marker='x', c='gold', label='Signature edge signal to no signal', zorder=-5)
    format_save(Path('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_19062024/ProtoEmission/methods/'),
                'edge_signature_filtered_'+crit, legend=True)

    # -----------------------------------------------------------------------------------------------------------------
    # Plot advanced signature recognizing:
    if k < len(first_measurements):
        A.create_map(inverse=[True, False])
    else:
        A.create_map(inverse=[False, False])
    if k < len(first_measurements):
        # Simple case correction:
        A.map['z'] = simple_artefact_filter(A.map['z'], threshold2)
    A.plot_map(None, contour=False, intensity_limits=intensity_limits)
    fig, ax = plt.gcf(), plt.gca()
    # Exclude columns with complete zero:
    zero_column = []
    for j, column in enumerate(A.map['z']):
        if A.map['z'][j].mean() == 0 and A.map['z'][j].std() == 0:
            zero_column.append(j)
    # For each column: detect points where this threshold is exceeded
    for i, column in enumerate(A.map['z'].T):
        for j, el in enumerate(column):
            if j == 0 or j == 1 or j == 2 or j == 3:
                continue
            if j in zero_column or j - 1 in zero_column or j - 2 in zero_column or j - 3 in zero_column or j - 4 in zero_column:
                continue
            # Signal on edge recognition
            if column[j - 4] < threshold2 <= column[j - 3] and column[j - 3] >= threshold2 > column[j - 2] and column[
                j - 2] < threshold2 <= column[j-1] and column[j] >= threshold2:
                ax.plot(A.map['x'][i] + 0.125, A.map['y'][j - 4] + 0.25, ls='', marker='x', c='lime', markersize=3)
            # Signal off edge recognition
            if column[j - 3] >= threshold2 > column[j - 2] and column[j - 2] < threshold2 <= column[j - 1] and column[
                j - 1] >= threshold2 > column[j] and column[j-4] >= threshold2:
                ax.plot(A.map['x'][i] + 0.125, A.map['y'][j - 3] + 0.25, ls='', marker='x', c='gold', markersize=3)
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    ax.plot(0, 0, ls='', marker='x', c='lime', label='Signature edge no signal to signal', zorder=-5)
    ax.plot(0, 0, ls='', marker='x', c='gold', label='Signature edge signal to no signal', zorder=-5)
    format_save(Path('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_19062024/ProtoEmission/methods/'),
                'edge_signature_enhanced_' + crit, legend=True)

    # -----------------------------------------------------------------------------------------------------------------
    # Simple correction plot
    # Exclude columns with complete zero:
    zero_column = []
    for j, column in enumerate(A.map['z']):
        if A.map['z'][j].mean() == 0 and A.map['z'][j].std() == 0:
            zero_column.append(j)
    # For each column: detect points where this threshold is exceeded
    # I save the location of the pixel before the signal starts (for the signal-on edge) and the location with the
    # last full signal pixel for case 2 (for the signal-off edge)
    signature_on = []
    signature_off = []

    for i, column in enumerate(A.map['z'].T):
        for j, el in enumerate(column):
            if j == 0 or j == 1 or j == 2:
                continue
            if j in zero_column or j - 1 in zero_column or j - 2 in zero_column or j - 3 in zero_column:
                continue
            if column[j - 3] < threshold2 <= column[j - 2] and column[j - 2] >= threshold2 > column[j - 1] and column[
                j - 1] < threshold2 <= column[j]:
                signature_on.append([j-2, i])
            if column[j - 3] >= threshold2 > column[j - 2] and column[j - 2] < threshold2 <= column[j - 1] and column[
                j - 1] >= threshold2 > column[j]:
                signature_off.append([j-3, i])

    delta_list = []
    corrected_image = deepcopy(A.map['z'])
    image_cache = deepcopy(A.map['z'])
    for edge in signature_on:
        delta_list.append(2*corrected_image[edge[0], edge[1]] / corrected_image[edge[0]+2, edge[1]])
        corrected_image[edge[0]+1, edge[1]] = corrected_image[edge[0], edge[1]] + corrected_image[edge[0]+1, edge[1]] - dark_mean
        corrected_image[edge[0], edge[1]] = dark_mean
    for edge in signature_off:
        delta_list.append(2*corrected_image[edge[0]+2, edge[1]] / corrected_image[edge[0], edge[1]])
        corrected_image[edge[0]+1, edge[1]] = corrected_image[edge[0]+2, edge[1]] + corrected_image[edge[0]+1, edge[1]] - dark_mean
        corrected_image[edge[0]+2, edge[1]] = dark_mean

    print(delta_list)
    print(np.mean(delta_list), np.std(delta_list))
    A.map['z'] = corrected_image
    A.plot_map(None, contour=False, intensity_limits=intensity_limits)
    format_save(Path('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_19062024/ProtoEmission/new_maps/'), crit+'_simple_')
    A.plot_map(None, contour=True, intensity_limits=intensity_limits)
    format_save(Path('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_19062024/ProtoEmission/new_maps/'),
                crit + '_simple_'+'contour')
    A.plot_map(None, contour=False, intensity_limits=intensity_limits)
    format_save(Path('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_19062024/ProtoEmission/corrected/'),
                crit + '_simple_')

    # -----------------------------------------------------------------------------------------------------------------
    def delta_func(sp1, s, signal):
        value = np.sqrt(4 * sp1 * (sp1 - signal) + s ** 2 + 4 * s * sp1)
        return np.array([(+value + s + 2 * sp1) / signal, (-value + s + 2 * sp1) / signal])


    def alpha_func(sp1, s, signal, delta=None):
        if delta is None:
            return 2 * sp1 / (delta_func(sp1, s, signal)[0] * signal)
        else:
            return 2 * sp1 / (delta * signal)

    # Advanced correction plot, considering the edge
    A.map['z'] = image_cache

    # Exclude columns with complete zero:
    zero_column = []
    for j, column in enumerate(A.map['z']):
        if A.map['z'][j].mean() == 0 and A.map['z'][j].std() == 0:
            zero_column.append(j)
    # For each column: detect points where this threshold is exceeded
    # I save the location of the pixel before the signal starts (for the signal-on edge) and the location with the
    # last full signal pixel for case 2 (for the signal-off edge)
    signature_on = []
    signature_off = []

    # For each column: detect points where this threshold is exceeded
    for i, column in enumerate(A.map['z'].T):
        for j, el in enumerate(column):
            if j == 0 or j == 1 or j == 2 or j == 3:
                continue
            if j in zero_column or j - 1 in zero_column or j - 2 in zero_column or j - 3 in zero_column or j - 4 in zero_column:
                continue
            # Signal on edge recognition
            if column[j - 4] < threshold2 <= column[j - 3] and column[j - 3] >= threshold2 > column[j - 2] and column[
                j - 2] < threshold2 <= column[j - 1] and column[j] >= threshold2:
                signature_on.append([j-3, i])
            # Signal off edge recognition
            if column[j - 3] >= threshold2 > column[j - 2] and column[j - 2] < threshold2 <= column[j - 1] and column[
                j - 1] >= threshold2 > column[j] and column[j - 4] >= threshold2:
                signature_off.append([j-3, i])

    corrected_image = deepcopy(A.map['z'])

    delta_track = []
    for edge in signature_on:
        sn = corrected_image[edge[0]+1, edge[1]]
        snp1 = corrected_image[edge[0], edge[1]] - dark_mean
        snm1 = corrected_image[edge[0]+2, edge[1]]
        snm2 = corrected_image[edge[0]+3, edge[1]]
        signal = (snm1+snm2)/2
        if signal > 0:
            delta_track.append(delta_func(snp1, sn, signal))
    for edge in signature_off:
        sn = corrected_image[edge[0]+1, edge[1]]
        snp1 = corrected_image[edge[0]+2, edge[1]] - dark_mean
        snm1 = corrected_image[edge[0], edge[1]]
        snm2 = corrected_image[edge[0]-1, edge[1]]
        signal = (snm1 + snm2) / 2
        if signal > 0:
            delta_track.append(delta_func(snp1, sn, signal))

    delta_track = np.array(delta_track)
    delta = delta_track[:, 0][delta_track[:, 0] == delta_track[:, 0]].mean()
    delta = 1.8
    for edge in signature_on:
        sn = corrected_image[edge[0]+1, edge[1]]
        snp1 = corrected_image[edge[0], edge[1]] - dark_mean
        snm1 = corrected_image[edge[0]+2, edge[1]]
        snm2 = corrected_image[edge[0]+3, edge[1]]
        signal = (snm1 + snm2) / 2
        alpha = alpha_func(snp1, sn, signal, delta=delta)
        corrected_image[edge[0], edge[1]] = dark_mean
        corrected_image[edge[0]+1, edge[1]] = sn - snp1/alpha + 2*snp1
        corrected_image[edge[0]+2, edge[1]] = signal
    for edge in signature_off:
        sn = corrected_image[edge[0]+1, edge[1]]
        snp1 = corrected_image[edge[0]+2, edge[1]] - dark_mean
        snm1 = corrected_image[edge[0], edge[1]]
        snm2 = corrected_image[edge[0]-1, edge[1]]
        signal = (snm1 + snm2) / 2
        alpha = alpha_func(snp1, sn, signal, delta=delta)
        corrected_image[edge[0]+2, edge[1]] = dark_mean
        corrected_image[edge[0]+1, edge[1]] = sn - snp1/alpha + 2*snp1
        corrected_image[edge[0], edge[1]] = signal

    A.map['z'] = corrected_image
    A.plot_map(None, contour=False, intensity_limits=intensity_limits)
    format_save(Path('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_19062024/ProtoEmission/new_maps/'),
                crit + '_edge_')
    A.plot_map(None, contour=True, intensity_limits=intensity_limits)
    format_save(Path('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_19062024/ProtoEmission/new_maps/'),
                crit + '_edge_' + 'contour')
    A.plot_map(None, contour=False, intensity_limits=intensity_limits)
    format_save(Path('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_19062024/ProtoEmission/corrected/'),
                crit + '_edge_')