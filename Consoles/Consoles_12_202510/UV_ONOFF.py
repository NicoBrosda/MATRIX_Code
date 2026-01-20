import numpy as np

from EvaluationSoftware.movie_modules import *
from skimage.filters import threshold_otsu

import numpy as np
import matplotlib.pyplot as plt


def frequency_spectrum(x, fs):
    N = len(x)

    # FFT
    X = np.fft.fft(x)

    # Only positive frequencies
    X = X[:N//2]
    freqs = np.fft.fftfreq(N, d=1/fs)[:N//2]

    # Magnitude spectrum (power)
    magnitude = np.abs(X)

    return freqs, magnitude


def fft_spectrum(x, fs, window="hann"):
    """
    Returns frequencies and magnitude spectrum using a window.
    x : input signal
    fs : sampling rate in Hz
    window : "hann", "hamming", "blackman", or None
    """

    N = len(x)

    # Choose window
    if window == "hann":
        w = np.hanning(N)
    elif window == "hamming":
        w = np.hamming(N)
    elif window == "blackman":
        w = np.blackman(N)
    elif window is None:
        w = np.ones(N)
    else:
        raise ValueError("Unknown window")

    # Apply window
    xw = x * w

    # FFT
    X = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(N, d=1/fs)

    # Amplitude correction for window + FFT length
    magnitude = (2.0 / np.sum(w)) * np.abs(X)

    return freqs, magnitude


def spectrum_from_autocorr(x, fs):
    # Remove DC
    x = x - np.mean(x)

    # Compute autocorrelation
    ac = np.correlate(x, x, mode='full')
    ac = ac[len(ac)//2:]  # keep positive lags

    # FFT of autocorrelation (Wiener–Khinchin)
    S = np.fft.rfft(ac)
    freqs = np.fft.rfftfreq(len(ac), d=1/fs)

    # Power spectral density (real & non-negative)
    psd = np.abs(S)

    return freqs, psd


def detect_edges(signal, threshold=None, min_distance=1):
    """
    Erkennt steigende und fallende Flanken in einem Signal basierend auf der Ableitung.

    Parameter:
    - signal: 1D-Array des Signals
    - threshold: Schwellenwert für die Flankenerkennung (optional, wird automatisch bestimmt)
    - min_distance: Minimaler Abstand zwischen erkannten Flanken in Samples

    Rückgabe:
    - rising_edges: Indizes der steigenden Flanken
    - falling_edges: Indizes der fallenden Flanken
    """
    # Ableitung des Signals berechnen
    derivative = np.diff(signal)

    # Wenn kein Schwellenwert angegeben wird, automatisch einen berechnen
    if threshold is None:
        # Otsu-Methode oder einfacher: Standardabweichung × Faktor
        threshold = threshold_otsu(signal)

    # Boolean mask above threshold
    above = signal > threshold

    # Rising edges: False → True
    rising_edges = np.where(np.diff(above.astype(int)) == 1)[0] + 1

    # Falling edges: True → False
    falling_edges = np.where(np.diff(above.astype(int)) == -1)[0] + 1

    # Flanken filtern, die zu nahe beieinander liegen
    rising_edges = filter_close_indices(rising_edges, min_distance)
    falling_edges = filter_close_indices(falling_edges, min_distance)

    return rising_edges, falling_edges


def schmitt_edges(x, thr_low, thr_high):
    state = False
    rising = []
    falling = []

    for i in range(1, len(x)):
        if not state and x[i-1] <= thr_high and x[i] > thr_high:
            state = True
            rising.append(i)
        elif state and x[i-1] >= thr_low and x[i] < thr_low:
            state = False
            falling.append(i)

    return np.array(rising), np.array(falling)


def filter_close_indices(indices, min_distance):
    """
    Filtert Indizes, die zu nahe beieinander liegen, und behält nur den ersten.
    """
    if len(indices) == 0:
        return indices

    filtered_indices = [indices[0]]

    for idx in indices[1:]:
        if idx - filtered_indices[-1] >= min_distance:
            filtered_indices.append(idx)

    return np.array(filtered_indices)


def quick_norm_return(array_name):
    # Get the correct mapping for the matrix array
    mapping = Path('../../Files/mapping.xlsx')
    direction1 = pd.read_excel(mapping, header=1)
    direction1 = np.array([int(k[-3:]) for k in direction1['direction_1']])
    direction2 = pd.read_excel(mapping, header=1)
    direction2 = np.array([int(k[-3:]) for k in direction2['direction_2']])
    if 'BigMatrix' == array_name:
        mapping = Path('../../Files/Mapping_MatrixArray.xlsx')
        dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_071025_UVTest/BigMatrix/')
        matrix_dark = ['Dark1,9.csv']
        norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_071025_UVTest/BigMatrix/scans/')
        norm = ['yscan_NoLense_']
        data2 = pd.read_excel(mapping, header=None)
        mapping_map = data2.to_numpy().flatten()
        translated_mapping = np.array([direction2[np.argwhere(direction1 == i)[0][0]] - 1 for i in mapping_map])
        readout, position_parser = lambda x, y: ams_2D_assignment_readout(x, y, channel_assignment=translated_mapping), standard_position
        A = Analyzer((11, 11), 0.8, 0.2, readout=readout)
    elif 'SmallMatrix' == array_name:
        mapping = Path('../../Files/Mapping_BigMatrix_2.xlsx')
        dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_071025_UVTest/SmallMatrix/')
        matrix_dark = ['DarkNoise2_1,9']
        norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_071025_UVTest/SmallMatrix/Maps/')
        norm = 'UVDiode_YScan_'
        data2 = pd.read_excel(mapping, header=None)
        mapping_map = data2.to_numpy().flatten()
        translated_mapping = np.array([direction2[np.argwhere(direction1 == i)[0][0]] - 1 for i in mapping_map])
        readout, position_parser = lambda x, y: ams_2D_assignment_readout(x, y, channel_assignment=translated_mapping), standard_position
        A = Analyzer((11, 11), 0.4, 0.1, readout=readout)
    elif '2Line' == array_name:
        dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_071025_UVTest/2Line/Lense/')
        matrix_dark = ['2Line_Dark_1,9.csv']
        norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_071025_UVTest/')
        norm = ['2Line_YScan_']
        readout = lambda x, y: ams_2line_readout(x, y, channel_assignment=direction2-1)
        A = Analyzer((2, 64), (0.4, 0.4), (0.1, 0.1), readout=readout,
                     diode_offset=[[0, - 0.25], np.zeros(64)], position_parser=standard_position)
    else:
        return None

    A.set_dark_measurement(dark_path, matrix_dark)
    norm_func = lambda list_of_files, instance, method='least_squares': normalization_from_translated_array_v5(
        list_of_files, instance, method, align_lines=False)
    # A.normalization(norm_path, norm, normalization_module=norm_func)
    return A
# '''
# ---------------------------------------------------------------------------------------------------------------------
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_071025_UVTest/BigMatrix/ON_OFF_Behavior/')
results_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/ResultsUV/ONOFF/')

A = quick_norm_return('BigMatrix')
for crit in os.listdir(folder_path):
    break
    A.set_measurement(folder_path, crit)

    data = ams_fast_time_profiles(A.measurement_files[0], A, excluded=0, keep_first_row=True)['signal']
    data = data.reshape(-1, np.shape(data)[2])
    # Example Plot recorded data:
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    for ch in data:
        print('---')
        edges = detect_edges(ch)
        if len(edges[0]) < 10 and len(edges[0]) >= 1 and len(edges[1]) >= 1:
            rise, fall = edges
        print(edges)

        print(np.shape(ch))
        threshold = threshold_otsu(ch)
        # print(schmitt_edges(ch, threshold*1.1, threshold*0.9))

        print(threshold)
        print('---')
        if threshold < 100:
            continue

        freqs, magnitude = frequency_spectrum(ch, 1000)
        ax2.plot(freqs, magnitude)

        signal_level = ch[ch > threshold].mean()
        # ch = ch / signal_level

        line1 = ax.plot(ch, alpha=0.8, zorder=-1)
        # ax.plot(smooth_signal_lowpass(ch), c=line1[0].get_color(), zorder=1, alpha=0.8)
        # ax.plot(moving_average(ch), c=line1[0].get_color(), zorder=2, alpha=0.8)
        ax.axhline(threshold, ls='--', c=line1[0].get_color(), zorder=3)

    # break
    ax.set_xlabel(r'Time (Samples)')
    ax.set_ylabel(r'Measured Amplitude')
    # ax.set_ylim([0.8, 1.2])
    # format_save(save_path=results_path, save_name=f'Rise_{crit}', save_format='.png', fig=fig, axes=[ax])

    # ax.set_ylim([0, 1.2])
    ax.set_xlim([rise[0]-200, rise[0]+200])
    format_save(save_path=results_path, save_name=f'Rise_{crit}', save_format='.png', fig=fig, axes=[ax])

    ax.set_xlim([fall[0]-100, fall[0]+100])
    format_save(save_path=results_path, save_name=f'Fall_{crit}', save_format='.png', fig=fig, axes=[ax])

    ax2.set_yscale('log')
    ax2.axvline(50, c='r', zorder=-1)
    ax2.axvline(60, c='b', zorder=-1)
    format_save(save_path=results_path, save_name=f'FFT_{crit}', save_format='.png', fig=fig2, axes=[ax2])
    # ax.set_ylim([-0.05, 0.2])
    # format_save(save_path=results_path, save_name=f'Afterglow_{crit}', save_format='.png', fig=fig, axes=[ax])


folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_071025_UVTest/SmallMatrix/')
results_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/ResultsUV/Noise/')

A = quick_norm_return('SmallMatrix')
for crit in os.listdir(folder_path):
    break
    if not '.csv' in crit:
        continue
    A.set_measurement(folder_path, crit)

    data = ams_fast_time_profiles(A.measurement_files[0], A, excluded=0, keep_first_row=True)['signal']
    data = data.reshape(-1, np.shape(data)[2])
    # Example Plot recorded data:
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    rise, fall = [300], [300]
    for ch in data:
        print('---')
        edges = detect_edges(ch)
        if len(edges[0]) < 10 and len(edges[0]) >= 1 and len(edges[1]) >= 1:
            rise, fall = edges
        print(edges)

        print(np.shape(ch))
        threshold = threshold_otsu(ch)
        # print(schmitt_edges(ch, threshold*1.1, threshold*0.9))

        print(threshold)
        print('---')
        if threshold < 100:
            continue

        freqs, magnitude = frequency_spectrum(ch, 1000)
        ax2.plot(freqs, magnitude)

        signal_level = ch[ch > threshold].mean()
        # ch = ch / signal_level

        line1 = ax.plot(ch, alpha=0.8, zorder=-1)
        # ax.plot(smooth_signal_lowpass(ch), c=line1[0].get_color(), zorder=1, alpha=0.8)
        # ax.plot(moving_average(ch), c=line1[0].get_color(), zorder=2, alpha=0.8)
        ax.axhline(threshold, ls='--', c=line1[0].get_color(), zorder=3)

    # break
    ax.set_xlabel(r'Time (Samples)')
    ax.set_ylabel(r'Measured Amplitude')
    # ax.set_ylim([0.8, 1.2])
    format_save(save_path=results_path, save_name=f'Over_{crit}', save_format='.png', fig=fig, axes=[ax])

    # ax.set_ylim([0, 1.2])
    ax.set_xlim([rise[0]-200, rise[0]+200])
    format_save(save_path=results_path, save_name=f'Rise_{crit}', save_format='.png', fig=fig, axes=[ax])

    ax.set_xlim([fall[0]-100, fall[0]+100])
    format_save(save_path=results_path, save_name=f'Fall_{crit}', save_format='.png', fig=fig, axes=[ax])

    ax2.set_yscale('log')
    ax2.axvline(50, c='r', zorder=-1)
    ax2.axvline(60, c='b', zorder=-1)
    format_save(save_path=results_path, save_name=f'FFT_{crit}', save_format='.png', fig=fig2, axes=[ax2])
    # ax.set_ylim([-0.05, 0.2])
    # format_save(save_path=results_path, save_name=f'Afterglow_{crit}', save_format='.png', fig=fig, axes=[ax])


folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_071025_UVTest/2Line/Lense/')
results_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/ResultsUV/Noise/')

A = quick_norm_return('2Line')
for crit in ['2Line_Dark', '2Line_Signal']:
    A.set_measurement(folder_path, crit)

    data = ams_fast_time_profiles(A.measurement_files[0], A, excluded=0, keep_first_row=True)['signal']
    data = data.reshape(-1, np.shape(data)[2])
    # Example Plot recorded data:
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()

    rise, fall = [300], [300]
    for ch in data:
        print('---')
        edges = detect_edges(ch)
        if len(edges[0]) < 10 and len(edges[0]) >= 1 and len(edges[1]) >= 1:
            rise, fall = edges
        print(edges)

        print(np.shape(ch))
        threshold = threshold_otsu(ch)
        # print(schmitt_edges(ch, threshold*1.1, threshold*0.9))

        print(threshold)
        print('---')
        if threshold < 100:
            continue

        freqs, magnitude = frequency_spectrum(ch, 2000)
        ax2.plot(freqs, magnitude, alpha=0.5)

        freqs, magnitude = fft_spectrum(ch, 2000, 'hann')
        ax3.plot(freqs, magnitude, alpha=0.5)

        freqs, magnitude = spectrum_from_autocorr(ch, 2000)
        ax4.plot(freqs, magnitude, alpha=0.5)

        signal_level = ch[ch > threshold].mean()
        # ch = ch / signal_level

        line1 = ax.plot(ch, alpha=0.8, zorder=-1)
        # ax.plot(smooth_signal_lowpass(ch), c=line1[0].get_color(), zorder=1, alpha=0.8)
        # ax.plot(moving_average(ch), c=line1[0].get_color(), zorder=2, alpha=0.8)
        ax.axhline(threshold, ls='--', c=line1[0].get_color(), zorder=3)

    # break
    ax.set_xlabel(r'Time (Samples)')
    ax.set_ylabel(r'Measured Amplitude')
    # ax.set_ylim([0.8, 1.2])
    # format_save(save_path=results_path, save_name=f'Over_{crit}', save_format='.png', fig=fig, axes=[ax])

    # ax.set_ylim([0, 1.2])
    ax.set_xlim([rise[0]-200, rise[0]+200])
    # format_save(save_path=results_path, save_name=f'Rise_{crit}', save_format='.png', fig=fig, axes=[ax])

    ax.set_xlim([fall[0]-100, fall[0]+100])
    # format_save(save_path=results_path, save_name=f'Fall_{crit}', save_format='.png', fig=fig, axes=[ax])

    ax2.set_yscale('log')
    ax2.axvline(50, c='r', zorder=-1)
    ax2.axvline(60, c='b', zorder=-1)
    format_save(save_path=results_path, save_name=f'FFT_{crit}', save_format='.png', fig=fig2, axes=[ax2])

    ax3.set_yscale('log')
    ax3.axvline(50, c='r', zorder=-1)
    ax3.axvline(60, c='b', zorder=-1)
    format_save(save_path=results_path, save_name=f'WindowedFFT_{crit}', save_format='.png', fig=fig3, axes=[ax3])

    ax4.set_yscale('log')
    ax4.axvline(50, c='r', zorder=-1)
    ax4.axvline(60, c='b', zorder=-1)
    format_save(save_path=results_path, save_name=f'AutoCorr_{crit}', save_format='.png', fig=fig4, axes=[ax4])
    # ax.set_ylim([-0.05, 0.2])
    # format_save(save_path=results_path, save_name=f'Afterglow_{crit}', save_format='.png', fig=fig, axes=[ax])