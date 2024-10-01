from EvaluationSoftware.main import *
from EvaluationSoftware.readout_modules import ams_channel_assignment_readout
from EvaluationSoftware.normalization_modules import normalization_from_translated_array
from EvaluationSoftware.parameter_parsing_modules import standard_voltage

mapping = Path('../Files/mapping.xlsx')
data = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:])-1 for k in data['direction_2']]

readout, position_parser = lambda x, y: ams_channel_assignment_readout(x, y, channel_assignment=channel_assignment), standard_position

A = Analyzer((1, 128), 0.5, 0.0, readout=readout)


folder_path = Path('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/matrix_230924/')

voltage_array1 = ['d2_1n_5s_flat_calib_nA_', 'voltage_scan_no_beam_nA_', 'voltage_scan_beam_2nA_nA_']
voltage_array3 = ['Array3_VoltageScan_dark_nA_', 'Array3_VoltageScan_200um_2nA_nA_']

measurements = voltage_array1 + voltage_array3

for k, crit in enumerate(measurements):
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

    A.measurement_data = []
    A.set_measurement(folder_path, crit)

    for file in A.measurement_files:
        pos = position_parser(file)
        voltage = standard_voltage(file)
        print(voltage)
        cache = A.readout(file, A)
        cache.update({'position': pos, 'voltage': voltage})
        A.measurement_data.append(cache)

    if 'no_beam' in crit or 'dark' in crit or crit == 'd2_1n_5s_flat_calib_nA_':
        if crit == 'd2_1n_5s_flat_calib_nA_':
            crit = 'voltage_scan_no_beam_reduced_points_'
        fig, ax = plt.subplots()
        for i, data in enumerate(A.measurement_data):
            if i == 0:
                ax.plot(data['voltage'], np.mean(data['signal']), ls='', marker='x', c='k', label='Dark current')
                ax.plot(data['voltage'], np.std(data['signal']), ls='', marker='o', c='k', label='Dark Noise')
            else:
                ax.plot(data['voltage'], np.mean(data['signal']), ls='', marker='x', c='k')
                ax.plot(data['voltage'], np.std(data['signal']), ls='', marker='o', c='k')

        ax.set_xlabel('Voltage AMS circuit (V)')
        ax.set_ylabel('Diode signal (a.u.)')
        ax.set_yscale('log')
        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())
        just_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/Voltage/',
                    crit, legend=True)

        fig, ax = plt.subplots()
        for i, data in enumerate(A.measurement_data):
            ax.plot(data['voltage'], np.mean(data['signal'])/np.std(data['signal']), ls='', marker='x', c='k')

        ax.set_xlabel('Voltage AMS circuit (V)')
        ax.set_ylabel('Signal to noise ratio')
        ax.set_yscale('log')
        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())
        just_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/Voltage/',
                  crit+'_SnoRatio_', legend=True)
    else:
        fig, ax = plt.subplots()
        for i, data in enumerate(A.measurement_data):
            threshold = ski_threshold_otsu(data['signal'])
            print(data['voltage'] , threshold)
            signal = np.mean(data['signal'][data['signal'] > threshold])
            dark = np.mean(data['signal'][data['signal'] < threshold])
            signal_std = np.std(data['signal'][data['signal'] > threshold])
            dark_std = np.std(data['signal'][data['signal'] < threshold])
            if i == 0:
                ax.plot(data['voltage'], signal, ls='', marker='x', c='r', label='Signal current')
                ax.plot(data['voltage'], signal_std, ls='', marker='o', c='r', label='Signal noise')
                ax.plot(data['voltage'], dark, ls='', marker='x', c='b', label='Dark current')
                ax.plot(data['voltage'], dark_std, ls='', marker='o', c='b', label='Dark noise')
            else:
                ax.plot(data['voltage'], signal, ls='', marker='x', c='r')
                ax.plot(data['voltage'], signal_std, ls='', marker='o', c='r')
                ax.plot(data['voltage'], dark, ls='', marker='x', c='b')
                ax.plot(data['voltage'], dark_std, ls='', marker='o', c='b')

        ax.set_xlabel('Voltage AMS circuit (V)')
        ax.set_ylabel('Diode signal (a.u.)')
        ax.set_yscale('log')
        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())
        just_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/Voltage/',
                  crit, legend=True)

