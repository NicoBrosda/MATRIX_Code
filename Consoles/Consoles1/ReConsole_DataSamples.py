from EvaluationSoftware.main import *
from EvaluationSoftware.readout_modules import ams_channel_assignment_readout
from Consoles.Consoles8Gafchromic.Concept8GafMeasurementComparison import GafImage
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from scipy import signal
from skimage.filters import threshold_otsu


def smooth_signal_lowpass(data, cutoff_freq=45, fs=1000, order=5):
    """
    Tiefpass-Butterworth-Filter zur Entfernung von Hochfrequenzrauschen

    Parameter:
    - data: Eingangssignal
    - cutoff_freq: Grenzfrequenz in Hz (unter 50 Hz für 50 Hz-Rauschen)
    - fs: Abtastrate in Hz
    - order: Filterordnung

    Rückgabe:
    - Geglättetes Signal
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low')
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data


def wavelet_denoising(data, wavelet='db4', level=2):
    """
    Wavelet-basierte Rauschunterdrückung

    Parameter:
    - data: Eingangssignal
    - wavelet: Typ des Wavelets
    - level: Zerlegungsebene

    Rückgabe:
    - Entrauschtes Signal
    """
    import pywt

    # Wavelet-Zerlegung
    coeffs = pywt.wavedec(data, wavelet, level=level)

    # Schwellenwert für jede Zerlegungsebene berechnen und anwenden
    for i in range(1, len(coeffs)):
        # Adaptiver Schwellenwert basierend auf der Varianz
        threshold = np.sqrt(2 * np.log(len(coeffs[i]))) * np.std(coeffs[i]) / 0.6745
        coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')

    # Rekonstruktion des Signals
    filtered_data = pywt.waverec(coeffs, wavelet)

    # Länge anpassen, falls nötig
    return filtered_data[:len(data)]


def moving_average(data, window_size=20):
    """
    Gleitender Mittelwert zur allgemeinen Glättung

    Parameter:
    - data: Eingangssignal
    - window_size: Fenstergröße für die Mittelung

    Rückgabe:
    - Geglättetes Signal
    """
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

save_path = Path('/Users/nico_brosda/Desktop/iphc_python_misc/Results/concept/')
example = '/Users/nico_brosda/Desktop/iphc_python_misc/matrix_27052024/e2_5000p_nA_2.csv'
example2 = '/Users/nico_brosda/Desktop/iphc_python_misc/matrix_27052024/e2_50p_nA_2.csv'

paths = ['/Users/nico_brosda/Desktop/iphc_python_misc/matrix_27052024/e2_500p_bottom_nA_2.csv',
         '/Users/nico_brosda/Desktop/iphc_python_misc/matrix_27052024/e2_500p_nA_2.csv',
         '/Users/nico_brosda/Desktop/iphc_python_misc/matrix_27052024/e2_500p_top_nA_2.csv']

# ---------------------------------------------------------------------------------------------------------------------
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/iphc_python_misc/matrix_27052024/')
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/iphc_python_misc/matrix_27052024/e1/')
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/iphc_python_misc/matrix_27052024/d2/')

save_path = Path('/Users/nico_brosda/Cyrce_Messungen/Investigations/RiseFallBehaviour/27052024/')

readout, position_parser = ams_constant_signal_readout, standard_position
A = Analyzer((1, 64), 0.4, 0.1, readout=readout,
             position_parser=position_parser, voltage_parser=standard_voltage)
# Dark Subtraction - correct file assignment
dark_path = folder_path
dark = ['d2_1n_3s_beam_all_without_diffuser_dark.csv']
# Norm Assignment
norm_path = folder_path
norm = ['5s_flat_calib_']
norm_module = simple_normalization

for i, crit in enumerate(['e2_20000p_nA_2', 'e2_10000p_nA_2', 'e2_5000p_nA_2', 'e2_2000p_nA_2', 'e2_1000p_nA_2', 'e2_500p_nA_2', 'e2_200p_nA_2', 'e2_100p_nA_2']):
    pass
for i, crit in enumerate(['e1_16000p_lin_nA_2', 'e1_16000p_lin_nA_1', 'e1_8000p_lin_nA_2', 'e1_8000p_lin_nA_1']):
    pass
for i, crit in enumerate(['500p_top_nA_2', '1000p_long_nA_2', '2000p_long_nA_2', '4000p_long_nA_2']):

    A.set_measurement(folder_path, crit)
    data = ams_fast_time_profiles(A.measurement_files[0], A, excluded=6, keep_first_row=True)['signal']
    data = data.reshape(-1, np.shape(data)[2])
    # Example Plot recorded data:
    fig, ax = plt.subplots()
    for ch in data:
        print(np.shape(ch))
        threshold = threshold_otsu(ch)
        print(threshold)
        if threshold < 200:
            continue

        signal_level = ch[ch > threshold].mean()
        ch = ch / signal_level

        line1 = ax.plot(ch, alpha=0.05, zorder=-1)
        # ax.plot(smooth_signal_lowpass(ch), c=line1[0].get_color(), zorder=1, alpha=0.8)
        ax.plot(moving_average(ch), c=line1[0].get_color(), zorder=2, alpha=0.8)
        # ax.axhline(threshold / signal_level, ls='--', c=line1[0].get_color(), zorder=3)


    ax.set_xlabel(r'Sample (Total range = 5$\,$s)')
    ax.set_ylabel(r'Measured Amplitude')
    ax.set_ylim([0.8, 1.2])
    format_save(save_path=save_path, save_name=f'Rise_{crit}', save_format='.png', fig=fig, axes=[ax])

    ax.set_ylim([0, 1.2])
    format_save(save_path=save_path, save_name=f'Overall_{crit}', save_format='.png', fig=fig, axes=[ax])

    ax.set_ylim([-0.05, 0.2])
    format_save(save_path=save_path, save_name=f'Afterglow_{crit}', save_format='.png', fig=fig, axes=[ax])
