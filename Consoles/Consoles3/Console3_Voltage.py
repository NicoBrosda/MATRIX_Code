from EvaluationSoftware.main import *
from EvaluationSoftware.readout_modules import ams_channel_assignment_readout
from EvaluationSoftware.normalization_modules import normalization_from_translated_array
from EvaluationSoftware.parameter_parsing_modules import standard_voltage, first_measurement_voltage

mapping = Path('../../Files/mapping.xlsx')
data = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:])-1 for k in data['direction_2']]

readout, position_parser = lambda x, y: ams_channel_assignment_readout(x, y, channel_assignment=channel_assignment), standard_position

A = Analyzer((1, 128), 0.5, 0.0, readout=readout)
voltage_parser = standard_voltage

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_230924/')

voltage_array1 = ['d2_1n_5s_flat_calib_nA_', 'voltage_scan_no_beam_nA_', 'voltage_scan_beam_2nA_nA_']
voltage_array3 = ['Array3_VoltageScan_dark_nA_', 'Array3_VoltageScan_200um_2nA_nA_']

folder_path_old_d2 = Path('/Users/nico_brosda/Cyrce_Messungen/iphc_python_misc/matrix_27052024/d2/')
folder_path_old_e1 = Path('/Users/nico_brosda/Cyrce_Messungen/iphc_python_misc/matrix_27052024/e1/')
voltage_old_d2 = ['0_nA_']
voltage_old_e1 = ['e1_300p_nA_', 'e1_2000p_lin_nA_']

measurements = voltage_array1 + voltage_array3 + voltage_old_d2 + voltage_old_e1

c1 = sns.color_palette("bright")
c2 = sns.color_palette("tab10")
c3 = sns.color_palette("dark")

array_names = [r'2$\,$nA 128 x 0.5 x 0.5$\,$mm$^2$', r'2$\,$nA 128 x 0.25 x 0.5$\,$mm$^2$',
               r'd2 0$\,$nA 64 x 0.5 x 0.5$\,$mm$^2$', r'e1 2$\,$nA 64 x 0.5 x 0.5$\,$mm$^2$']

measurement_cache = []
new_arrays_cache = []
for k, crit in enumerate(measurements):
    print('-'*50)
    print('-'*50)
    print(crit)
    print('-'*50)
    if k < len(voltage_array1):
        ck = 0
    elif len(voltage_array1+voltage_array3) > k >= len(voltage_array1):
        ck = 1
    elif len(voltage_array1+voltage_array3) <= k < len(voltage_array1+voltage_array3+voltage_old_d2):
        ck = 2
    else:
        ck = 3
    name = array_names[ck]

    if k >= len(voltage_array1+voltage_array3):
        voltage_parser = first_measurement_voltage
        if 'e1' in crit:
            folder_path = folder_path_old_e1
            A = Analyzer((1, 64), 0.42, 0.08, readout=ams_otsus_readout, position_parser=first_measurements_position)
        else:
            folder_path = folder_path_old_d2
            A = Analyzer((1, 64), 0.42, 0.08, readout=ams_constant_signal_readout, position_parser=first_measurements_position)

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
        voltage = voltage_parser(file)
        cache = A.readout(file, A)
        cache.update({'position': pos, 'voltage': voltage})
        A.measurement_data.append(cache)

    if 'no_beam' in crit or 'dark' in crit or crit == 'd2_1n_5s_flat_calib_nA_' or crit == '0_nA_':
        if crit == 'd2_1n_5s_flat_calib_nA_':
            crit = 'voltage_scan_no_beam_reduced_points_'
        elif crit == '0_nA_':
            crit = 'voltage_scan_old_array_d2_'
        fig, ax = plt.subplots()
        for i, data in enumerate(A.measurement_data):
            if i == 0:
                ax.plot(data['voltage'], np.mean(data['signal']), ls='', marker='+', label='Dark current', color=c3[ck])
                ax.plot(data['voltage'], np.std(data['signal']), ls='', marker='^', label='Dark Noise', color=c3[ck])
            else:
                ax.plot(data['voltage'], np.mean(data['signal']), ls='', marker='+', color=c3[ck])
                ax.plot(data['voltage'], np.std(data['signal']), ls='', marker='^', color=c3[ck])

        ax.set_xlabel('Voltage AMS circuit (V)')
        ax.set_ylabel('Diode signal (a.u.)')
        ax.set_yscale('log')
        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())
        just_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/Voltage/',
                    crit, legend=True)

        fig, ax = plt.subplots()
        for i, data in enumerate(A.measurement_data):
            ax.plot(data['voltage'], np.mean(data['signal'])/np.std(data['signal']), ls='', marker='*', color=c3[ck])

        ax.set_xlabel('Voltage AMS circuit (V)')
        ax.set_ylabel('Signal to noise ratio')
        ax.set_yscale('log')
        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())
        just_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/Voltage/',
                  crit+'_SnoRatio_', legend=True)
        plt.close('all')
    else:
        if crit == 'e1_300p_nA_':
            crit = 'voltage_scan_old_array_e1_'
        fig, ax = plt.subplots()
        for i, data in enumerate(A.measurement_data):
            threshold = ski_threshold_otsu(data['signal'])
            signal = np.mean(data['signal'][data['signal'] > threshold])
            dark = np.mean(data['signal'][data['signal'] < threshold])
            signal_std = np.std(data['signal'][data['signal'] > threshold])
            dark_std = np.std(data['signal'][data['signal'] < threshold])
            if i == 0:
                ax.plot(data['voltage'], signal, ls='', marker='x', color=c1[ck], label='Signal current')
                ax.plot(data['voltage'], signal_std, ls='', marker='o', color=c2[ck], label='Signal noise')
                ax.plot(data['voltage'], dark, ls='', marker='+', color=c3[ck], label='Dark current')
                ax.plot(data['voltage'], dark_std, ls='', marker='^', color=c3[ck], label='Dark noise')
            else:
                ax.plot(data['voltage'], signal, ls='', marker='x', color=c1[ck])
                ax.plot(data['voltage'], signal_std, ls='', marker='o', color=c2[ck])
                ax.plot(data['voltage'], dark, ls='', marker='+', color=c3[ck])
                ax.plot(data['voltage'], dark_std, ls='', marker='^', color=c3[ck])

        ax.set_xlabel('Voltage AMS circuit (V)')
        ax.set_ylabel('Diode signal (a.u.)')
        ax.set_yscale('log')
        just_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/Voltage/',
                  crit, legend=True)
        if ax.get_xlim()[0] < 1:
            ax.set_xlim(0.85, 2.1)
            just_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/Voltage/',
                      crit+'no_saturation', legend=True)
        plt.close('all')

        if crit == 'voltage_scan_old_array_e1_':
            continue
        measurement_cache.append([name, A.measurement_data])
        if k < len(voltage_array1+voltage_array3):
            new_arrays_cache.append([name, A.measurement_data])

# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots()
for k, measurement_data in enumerate(measurement_cache):
    ck = k

    crit = measurement_data[0]
    measurement_data = measurement_data[1]
    for i, data in enumerate(measurement_data):
        threshold = ski_threshold_otsu(data['signal'])
        signal = np.mean(data['signal'][data['signal'] > threshold])
        dark = np.mean(data['signal'][data['signal'] < threshold])
        signal_std = np.std(data['signal'][data['signal'] > threshold])
        dark_std = np.std(data['signal'][data['signal'] < threshold])
        if i == 0 and k == 0:
            ax.plot(data['voltage'], signal, ls='', marker='x', color='k', label='Signal current')
            ax.plot(data['voltage'], dark, ls='', marker='+', color='k', label='Dark current')
        if i == 0:
            ax.plot(data['voltage'], signal, ls='', marker='x', color=c1[ck], label=crit)
        ax.plot(data['voltage'], signal, ls='', marker='x', color=c1[ck])
        ax.plot(data['voltage'], dark, ls='', marker='+', color=c3[ck])

ax.set_xlabel('Voltage AMS circuit (V)')
ax.set_ylabel('Diode signal (a.u.)')
ax.set_yscale('log')
just_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/Voltage/',
          'All_comparison', legend=True)
if ax.get_xlim()[0] < 1:
    ax.set_xlim(0.85, 2.1)
    just_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/Voltage/',
              'All_zoomed', legend=True)
plt.close('all')

# -----------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots()
for k, measurement_data in enumerate(measurement_cache):
    ck = k

    crit = measurement_data[0]
    measurement_data = measurement_data[1]
    for i, data in enumerate(measurement_data):
        threshold = ski_threshold_otsu(data['signal'])
        signal = np.mean(data['signal'][data['signal'] > threshold])
        dark = np.mean(data['signal'][data['signal'] < threshold])
        signal_std = np.std(data['signal'][data['signal'] > threshold])
        dark_std = np.std(data['signal'][data['signal'] < threshold])
        if i == 0 and k == 0:
            ax.plot(data['voltage'], signal_std, ls='', marker='o', color='k', label='Signal noise')
            ax.plot(data['voltage'], dark_std, ls='', marker='^', color='k', label='Dark noise')
        if i == 0:
            ax.plot(data['voltage'], signal_std, ls='', marker='x', color=c2[ck], label=crit)
        ax.plot(data['voltage'], signal_std, ls='', marker='o', color=c2[ck])
        ax.plot(data['voltage'], dark_std, ls='', marker='^', color=c3[ck])

ax.set_xlabel('Voltage AMS circuit (V)')
ax.set_ylabel('Diode signal (a.u.)')
ax.set_yscale('log')
just_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/Voltage/',
          'All_noise_comparison', legend=True)
if ax.get_xlim()[0] < 1:
    ax.set_xlim(0.85, 2.1)
    just_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/Voltage/',
              'All_noise_zoomed', legend=True)
plt.close('all')

# -----------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots()
for k, measurement_data in enumerate(measurement_cache):
    ck = k

    crit = measurement_data[0]
    measurement_data = measurement_data[1]
    for i, data in enumerate(measurement_data):
        threshold = ski_threshold_otsu(data['signal'])
        signal = np.mean(data['signal'][data['signal'] > threshold])
        dark = np.mean(data['signal'][data['signal'] < threshold])
        signal_std = np.std(data['signal'][data['signal'] > threshold])
        dark_std = np.std(data['signal'][data['signal'] < threshold])
        if i == 0 and k == 0:
            ax.plot(data['voltage'], signal/signal_std, ls='', marker='*', color='k', label='Signal SNR')
            ax.plot(data['voltage'], dark/dark_std, ls='', marker='v', color='k', label='Dark SNR')
            # ax.plot(data['voltage'], signal/dark, ls='', marker='s', color='k', label='Signal to Dark ratio')
        if i == 0:
            ax.plot(data['voltage'], signal/signal_std, ls='', marker='*', color=c1[ck], label=crit)
        ax.plot(data['voltage'], signal/signal_std, ls='', marker='*', color=c1[ck])
        ax.plot(data['voltage'], dark/dark_std, ls='', marker='v', color=c3[ck])
        # ax.plot(data['voltage'], signal/dark, ls='', marker='s', color=c2[ck])

ax.set_xlabel('Voltage AMS circuit (V)')
ax.set_ylabel('Signal to noise ratio')
ax.set_yscale('log')
just_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/Voltage/',
          'All_ratios', legend=True)
if ax.get_xlim()[0] < 1:
    ax.set_xlim(0.85, 2.1)
    just_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/Voltage/',
              'All_zoomed_ratios', legend=True)
plt.close('all')

# -----------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots()
for k, measurement_data in enumerate(measurement_cache):
    ck = k

    crit = measurement_data[0]
    measurement_data = measurement_data[1]
    for i, data in enumerate(measurement_data):
        threshold = ski_threshold_otsu(data['signal'])
        signal = np.mean(data['signal'][data['signal'] > threshold])
        dark = np.mean(data['signal'][data['signal'] < threshold])
        signal_std = np.std(data['signal'][data['signal'] > threshold])
        dark_std = np.std(data['signal'][data['signal'] < threshold])
        if i == 0:
            ax.plot(data['voltage'], signal-dark, ls='', marker='*', color=c1[ck], label=crit)
        ax.plot(data['voltage'], signal - dark, ls='', marker='*', color=c1[ck])

ax.set_xlabel('Voltage AMS circuit (V)')
ax.set_ylabel('Difference signal - dark (a.u.)')
ax.set_yscale('log')
just_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/Voltage/',
          'All_netto_signal', legend=True)
if ax.get_xlim()[0] < 1:
    ax.set_xlim(0.85, 2.1)
    just_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/Voltage/',
              'All_zoomed_netto_signal', legend=True)
plt.close('all')

# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots()
for k, measurement_data in enumerate(new_arrays_cache):
    ck = k

    crit = measurement_data[0]
    measurement_data = measurement_data[1]
    for i, data in enumerate(measurement_data):
        threshold = ski_threshold_otsu(data['signal'])
        signal = np.mean(data['signal'][data['signal'] > threshold])
        dark = np.mean(data['signal'][data['signal'] < threshold])
        signal_std = np.std(data['signal'][data['signal'] > threshold])
        dark_std = np.std(data['signal'][data['signal'] < threshold])
        if i == 0 and k == 0:
            ax.plot(data['voltage'], signal, ls='', marker='x', color='k', label='Signal current')
            ax.plot(data['voltage'], dark, ls='', marker='+', color='k', label='Dark current')
        if i == 0:
            ax.plot(data['voltage'], signal, ls='', marker='x', color=c1[ck], label=crit)
        ax.plot(data['voltage'], signal, ls='', marker='x', color=c1[ck])
        ax.plot(data['voltage'], dark, ls='', marker='+', color=c3[ck])

ax.set_xlabel('Voltage AMS circuit (V)')
ax.set_ylabel('Diode signal (a.u.)')
ax.set_yscale('log')
just_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/Voltage/',
          'New_comparison', legend=True)
if ax.get_xlim()[0] < 1:
    ax.set_xlim(0.85, 2.1)
    just_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/Voltage/',
              'New_zoomed', legend=True)
plt.close('all')

# -----------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots()
for k, measurement_data in enumerate(new_arrays_cache):
    ck = k

    crit = measurement_data[0]
    measurement_data = measurement_data[1]
    for i, data in enumerate(measurement_data):
        threshold = ski_threshold_otsu(data['signal'])
        signal = np.mean(data['signal'][data['signal'] > threshold])
        dark = np.mean(data['signal'][data['signal'] < threshold])
        signal_std = np.std(data['signal'][data['signal'] > threshold])
        dark_std = np.std(data['signal'][data['signal'] < threshold])
        if i == 0 and k == 0:
            ax.plot(data['voltage'], signal_std, ls='', marker='o', color='k', label='Signal noise')
            ax.plot(data['voltage'], dark_std, ls='', marker='^', color='k', label='Dark noise')
        if i == 0:
            ax.plot(data['voltage'], signal_std, ls='', marker='x', color=c2[ck], label=crit)
        ax.plot(data['voltage'], signal_std, ls='', marker='o', color=c2[ck])
        ax.plot(data['voltage'], dark_std, ls='', marker='^', color=c3[ck])

ax.set_xlabel('Voltage AMS circuit (V)')
ax.set_ylabel('Diode signal (a.u.)')
ax.set_yscale('log')

just_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/Voltage/',
          'New_noise_comparison', legend=True)
if ax.get_xlim()[0] < 1:
    ax.set_xlim(0.85, 2.1)
    just_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/Voltage/',
              'New_noise_zoomed', legend=True)
plt.close('all')

# -----------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots()
for k, measurement_data in enumerate(new_arrays_cache):
    ck = k

    crit = measurement_data[0]
    measurement_data = measurement_data[1]
    for i, data in enumerate(measurement_data):
        threshold = ski_threshold_otsu(data['signal'])
        signal = np.mean(data['signal'][data['signal'] > threshold])
        dark = np.mean(data['signal'][data['signal'] < threshold])
        signal_std = np.std(data['signal'][data['signal'] > threshold])
        dark_std = np.std(data['signal'][data['signal'] < threshold])
        if i == 0 and k == 0:
            ax.plot(data['voltage'], signal/signal_std, ls='', marker='*', color='k', label='Signal SNR')
            ax.plot(data['voltage'], dark/dark_std, ls='', marker='v', color='k', label='Dark SNR')
            # ax.plot(data['voltage'], signal/dark, ls='', marker='s', color='k', label='Signal to Dark ratio')
        if i == 0:
            ax.plot(data['voltage'], signal/signal_std, ls='', marker='*', color=c1[ck], label=crit)
        ax.plot(data['voltage'], signal/signal_std, ls='', marker='*', color=c1[ck])
        ax.plot(data['voltage'], dark/dark_std, ls='', marker='v', color=c3[ck])
        # ax.plot(data['voltage'], signal/dark, ls='', marker='s', color=c2[ck])

ax.set_xlabel('Voltage AMS circuit (V)')
ax.set_ylabel('Signal to noise ratio')
ax.set_yscale('log')

just_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/Voltage/',
          'New_ratios', legend=True)
if ax.get_xlim()[0] < 1:
    ax.set_xlim(0.85, 2.1)
    just_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/Voltage/',
              'New_zoomed_ratios', legend=True)
plt.close('all')

# -----------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots()
for k, measurement_data in enumerate(new_arrays_cache):
    ck = k

    crit = measurement_data[0]
    measurement_data = measurement_data[1]
    for i, data in enumerate(measurement_data):
        threshold = ski_threshold_otsu(data['signal'])
        signal = np.mean(data['signal'][data['signal'] > threshold])
        dark = np.mean(data['signal'][data['signal'] < threshold])
        signal_std = np.std(data['signal'][data['signal'] > threshold])
        dark_std = np.std(data['signal'][data['signal'] < threshold])
        if i == 0:
            ax.plot(data['voltage'], signal-dark, ls='', marker='*', color=c1[ck], label=crit)
        ax.plot(data['voltage'], signal - dark, ls='', marker='*', color=c1[ck])

ax.set_xlabel('Voltage AMS circuit (V)')
ax.set_ylabel('Difference signal - dark (a.u.)')
ax.set_yscale('log')

just_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/Voltage/',
          'New_netto_signal', legend=True)
if ax.get_xlim()[0] < 1:
    ax.set_xlim(0.85, 2.1)
    just_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/Voltage/',
              'New_zoomed_netto_signal', legend=True)
plt.close('all')

# -----------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots()
cache = []
for k, measurement_data in enumerate(new_arrays_cache):
    ck = k

    crit = measurement_data[0]
    measurement_data = measurement_data[1]
    inner_cache = []
    for i, data in enumerate(measurement_data):
        threshold = ski_threshold_otsu(data['signal'])
        signal = np.mean(data['signal'][data['signal'] > threshold])
        dark = np.mean(data['signal'][data['signal'] < threshold])
        signal_std = np.std(data['signal'][data['signal'] > threshold])
        dark_std = np.std(data['signal'][data['signal'] < threshold])

        inner_cache.append([data['voltage'], signal])

    sorting = np.argsort([i[0] for i in inner_cache])
    cache.append(np.array(inner_cache)[sorting])

ratios = [i[1]/j[1] for i, j in zip(cache[0], cache[1])]
ax.plot([i[0] for i in cache[0]], ratios, ls='', marker='*', c='k')
ax.axhline(2.352941176470588, c='grey', zorder=-1)

ax.set_xlabel('Voltage AMS circuit (V)')
ax.set_ylabel('Signal ratio Array1 / Array3')

just_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/Voltage/',
          'Array_ratios', legend=True)
if ax.get_xlim()[0] < 1:
    ax.set_xlim(0.85, 2.1)
    just_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/Voltage/',
              'Array_ratios_zoomed', legend=True)
plt.close('all')