import pandas as pd

from EvaluationSoftware.main import *
from EvaluationSoftware.readout_modules import ams_channel_assignment_readout
from EvaluationSoftware.normalization_modules import normalization_from_translated_array

mapping = Path('../../Files/mapping.xlsx')
data = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:])-1 for k in data['direction_2']]

readout, position_parser = lambda x, y: ams_channel_assignment_readout(x, y, channel_assignment=channel_assignment), standard_position

folder_path = Path('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/matrix_230924/')
new_measurements = ['round_aperture_2_3scans', 'Logo', 'scan_round_aperture_200um']
new_measurements = ['Array3_Logo', 'Array3_BeamShape', 'BraggPeak', 'MiscShape', 'round_aperture_2_3scans', 'Logo', 'scan_round_aperture_200um', 'BeamScan']
new_measurements = ['scan_round_aperture_200um']

dark_paths_array1 = ['voltage_scan_no_beam_nA_1.8000000000000005_x_20.0_y_70.0.csv',
                     'd2_1n_5s_flat_calib_nA_1.8000000000000007_x_20.0_y_70.0.csv']

dark_paths_array3_1V = ['Array3_VoltageScan_dark_nA_1.0_x_0.0_y_40.0.csv']

dark_paths_array3 = ['Array3_VoltageScan_dark_nA_1.8_x_0.0_y_40.0.csv']

norm_array1 = ['Normalization2']
norm_array1 = ['uniformity_scan_']

norm_array3 = ['Array3_DiffuserYScan']

for k, crit in enumerate(new_measurements):
    print('-'*50)
    print(crit)
    print('-'*50)

    A = Analyzer((1, 128), 0.5, 0.0, readout=readout)

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

    # Normalization - correct assignment
    if 'Array3' in crit:
        norm = norm_array3
    else:
        norm = norm_array1

    A.normalization(folder_path, norm, normalization_module=normalization_from_translated_array_v2)

    A.load_measurement(readout_module=readout)

    A.create_map(inverse=[True, False])

    A.plot_map(None, pixel=False)
    ax = plt.gca()
    ax.axvline(A.maps[0]['x'].mean(), c='b')
    ax.axhline(A.maps[0]['y'].mean(), c='r')

    format_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/BeamProps/', 'DirectionsInImage', legend=False)

    x_line = A.get_signal_xline()
    y_line = A.get_signal_yline()

    fig, ax = plt.subplots()
    ax.plot(A.maps[0]['x'], x_line, color='r', label='cross section x-direction')
    ax.set_xlabel('Position x (mm)')
    ax.set_ylabel('Diode signal (a.u.)')
    format_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/BeamProps/', 'CrossSectionsX', legend=True)

    fig, ax = plt.subplots()
    ax.plot(A.maps[0]['y'], y_line, color='b', label='cross section y-direction')
    ax.set_xlabel('Position y (mm)')
    ax.set_ylabel('Diode signal (a.u.)')
    format_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/BeamProps/', 'CrossSectionsY', legend=True)

    fig, ax = plt.subplots()
    ax.plot(A.maps[0]['x']-A.maps[0]['x'].min(), x_line, color='r', label='cross section x-direction')
    new_x = A.maps[0]['x']-A.maps[0]['x'].min()
    new_y = A.maps[0]['y']-A.maps[0]['y'].min()
    len_x = new_x.max()
    y1 = np.argmin(np.abs(new_y - (new_y.mean()-new_x.max()/2)))
    y2 = np.argmin(np.abs(new_y - (new_y.mean()+new_x.max()/2)))

    ax.plot(new_y[y1:y2]-new_y[y1:y2].min(), y_line[y1:y2], color='b', label='cross section y-direction')
    ax.set_xlabel('Position Overlay (mm)')
    ax.set_ylabel('Diode signal (a.u.)')
    format_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/BeamProps/', 'CrossSections',
                legend=True)


    def norm_to_one(array):
        return (array - np.min(array)) / np.max(array - np.min(array))


    def align_x_axes(array, center_ref=0):
        return array + (-np.mean(array)+center_ref)

    fig, ax = plt.subplots()
    cs1_data = pd.read_csv(Path("../../Files/CS1.csv"), header=0, names=["x", "y"])
    cs2_data = pd.read_csv(Path("../../Files/CS2.csv"), header=0, names=["x", "y"])
    cs1y, cs2y = cs1_data["y"] - np.max(cs1_data["y"]), cs2_data["y"] - np.max(cs2_data["y"])
    cs1y, cs2y = cs1y / np.min(cs1y), cs2y / np.min(cs2y)
    order1, order2 = np.argsort(cs1_data["x"]), np.argsort(cs2_data["x"])
    cs1x, cs2x, cs1y, cs2y = cs1_data["x"][order1], cs2_data["x"][order2], cs1y[order1], cs2y[order2]
    ax.plot(A.maps[0]['x'] - A.maps[0]['x'].min(), norm_to_one(x_line), color='r', label='cross section x-direction')
    new_x = A.maps[0]['x'] - A.maps[0]['x'].min()
    new_y = A.maps[0]['y'] - A.maps[0]['y'].min()
    len_x = new_x.max()
    y1 = np.argmin(np.abs(new_y - (new_y.mean() - new_x.max() / 2)))
    y2 = np.argmin(np.abs(new_y - (new_y.mean() + new_x.max() / 2)))

    ax.plot(new_y[y1:y2] - new_y[y1:y2].min(), norm_to_one(y_line[y1:y2]), color='b', label='cross section y-direction')
    ax.plot(align_x_axes(cs1x*25.4, new_x.mean()), cs1y, ls="--", c="k", label='Gafchromic Direction 1')
    ax.plot(align_x_axes(cs2x*25.4, new_x.mean()), cs2y, ls=":", c="k", label='Gafchromic Direction 2')

    ax.plot()
    ax.set_xlabel('Position Overlay (mm)')
    ax.set_ylabel('Response normed to 1')
    format_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/BeamProps/', 'QuantitativeComparison',
                legend=True)
