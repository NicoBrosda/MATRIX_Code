import numpy as np
import scipy.signal
from scipy.signal import find_peaks

from EvaluationSoftware.main import *
from EvaluationSoftware.helper_modules import LineShape, mean_diodes

mapping = Path('../Files/mapping.xlsx')
data = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:])-1 for k in data['direction_2']]

readout, position_parser = lambda x, y: ams_channel_assignment_readout(x, y, channel_assignment=channel_assignment), standard_position

folder_path = Path('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/matrix_230924/')

dark_paths_array1 = ['voltage_scan_no_beam_nA_1.8000000000000005_x_20.0_y_70.0.csv',
                     'd2_1n_5s_flat_calib_nA_1.8000000000000007_x_20.0_y_70.0.csv']

norm = ['uniformity_scan_']

for crit in ['BraggYScan'][0:1]:
    print('-' * 50)
    print(crit)
    print('-' * 50)

    A = Analyzer((1, 128), 0.4, 0.1, readout=readout)

    A.set_measurement(folder_path, crit)

    A.set_dark_measurement(folder_path, dark_paths_array1)

    # Normalization - correct assignment
    A.normalization(folder_path, norm, normalization_module=normalization_from_translated_array_v2)

    A.load_measurement(readout_module=readout)

    positions, signals = A.get_diodes_signal(direction='y', inverse=[False, True])

    # Direction of the measurement: x or y
    if A.diode_dimension[0] > A.diode_dimension[1]:
        sp = 0
    elif A.diode_dimension[0] < A.diode_dimension[1]:
        sp = 1
    else:
        sp = 0

    print(np.shape(positions), np.shape(signals))
    pos_mean, sig_mean = mean_diodes(positions.T, signals, A, sp, threshold=0)
    A.plot_diodes(None, direction='y', plotting_range=(80, 125), inverse=[False, True])

    ax = plt.gca()

    # Parameter to define the middle of the geometry (in data coordinates)
    middle = 92.711
    print('Full material from ', middle-0.966/2-20.503-6.702, ' mm until ', middle-0.966/2-20.503)
    print('Long ascend from ', middle-0.966/2-20.503, ' mm until ', middle-0.966/2)
    print('Free space from ', middle-0.966/2, ' mm until ', middle+0.966/2)
    print('Steep ascend from ', middle+0.966/2, ' mm until ', middle+0.966/2+4.877)
    print('Full material from ', middle+0.966/2+4.877, ' mm until ', middle+0.966/2+4.877+6.951)

    # ax.set_xlim(middle-0.966/2-20.503-6.702-2, middle+0.966/2+4.877+6.951+2)
    shape = LineShape([[0, 10], [6.702, 10], [20.503, 0], [0.966, 0], [4.877, 10], [6.951, 10]], distance_mode=True)
    shape.print_shape()
    # shape.mirror()
    shape.position(93, 6.951 + 4.877)
    shape.add_to_plot(0.0, 0.5, color='grey', alpha=0.7, zorder=-1, edgecolor='k')
    format_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/Bragg/', '_trapezandmean_',
                legend=False)
    plt.close('all')

    A.plot_diodes(None, direction='y', plotting_range=(80, 125), inverse=[False, True])
    ax = plt.gca()
    # Find peaks in the signal and select the 2 highest
    peaks = find_peaks(sig_mean, prominence=0)
    peaks = peaks[0]
    sorting = np.argsort(sig_mean[peaks])[-2:]

    for i, gr in enumerate(pos_mean[peaks][sorting]):
        c = sns.color_palette("tab10")[i]
        ax.axvline(gr, c='r')
        print('Peak at position', gr, 'with material thickness', shape.calculate_value(gr))
        print(shape.get_plot_value(shape.calculate_value(gr), 0, 0.5))
        ax.axhline(shape.get_plot_value(shape.calculate_value(gr), 0, 0.5), c=c, ls='-', alpha=1, lw=1)
    shape.add_to_plot(0.0, 0.5, color='grey', alpha=0.7, zorder=-1, edgecolor='k')
    ax.set_xlabel(r'Calculated real position of diodes (mm)')
    ax.set_ylabel(r'Signal Amplitude')
    ax.set_ylim(-100, ax.get_ylim()[1])
    format_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/Bragg/', '_peak_',
                legend=False)