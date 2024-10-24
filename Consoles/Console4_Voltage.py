from EvaluationSoftware.standard_processes import *

results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Results_111024/')
# Defining the voltage measurements done so far
store = []

folder_path2 = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_230924/')
mapping = Path('../Files/mapping.xlsx')
data = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:])-1 for k in data['direction_2']]
readout2, position_parser, voltage_parser = lambda x, y: ams_channel_assignment_readout(x, y, channel_assignment=channel_assignment), standard_position, standard_voltage
store.append([[r'128x0.5x0.5$\,$mm$^2$ line array $\#1$', '128x0.5x0.5line array_1'], folder_path2, 'voltage_scan_no_beam_nA_', 'voltage_scan_beam_2nA_nA_', ((1, 128), (0.4, 0.4), (0.1, 0.1), readout2, position_parser, voltage_parser)])
store.append([[r'128x0.25x0.5$\,$mm$^2$ line array $\#1$', '128x0.5x0.25line array_1'], folder_path2, 'Array3_VoltageScan_dark_nA_', 'Array3_VoltageScan_200um_2nA_nA_', ((1, 128), (0.17, 0.4), (0.08, 0.1), readout2, position_parser, voltage_parser)])

# '''
mapping = Path('../Files/mapping.xlsx')
direction1 = pd.read_excel(mapping, header=1)
direction1 = np.array([int(k[-3:]) for k in direction1['direction_1']])
direction2 = pd.read_excel(mapping, header=1)
direction2 = np.array([int(k[-3:]) for k in direction2['direction_2']])
mapping = Path('../Files/Mapping_MatrixArray.xlsx')
data2 = pd.read_excel(mapping, header=None)
mapping_map = data2.to_numpy().flatten()

translated_mapping = np.array([direction2[np.argwhere(direction1 == i)[0][0]]-1 for i in mapping_map])
readout, position_parser, voltage_parser = lambda x, y: ams_2D_assignment_readout(x, y, channel_assignment=translated_mapping), standard_position, standard_voltage

A = ((11, 11), (0.8, 0.8), (0.2, 0.2), readout, position_parser, voltage_parser)
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_111024/')
volt_dark = ['2DLarge_dark_']
volt_signal = ['2DLarge_200_um_2_nA_']
store.append([[r'11x11 1x1$\,$mm$^2$ matrix $\#1$', '11x11_1x1matrix_1'], folder_path, volt_dark, volt_signal, A])
# '''

# Some colour palettes
c1 = sns.color_palette("bright")
c2 = sns.color_palette("tab10")
c3 = sns.color_palette("dark")

# Get the data:
for i, st in enumerate(store):
    data_dark, data_signal = voltage_analysis(st[1], st[2], st[3], Analyzer(*st[4]))
    store[i].append(data_dark), store[i].append(data_signal)
    st = store[i]
    fig, ax = plt.subplots()

    ax.plot(st[5][0], st[5][1], ls='-', marker='+', label='Dark current', color=c3[i])
    ax.plot(st[5][0], st[5][2], ls='-', marker='^', label='Dark Noise', color=c3[i])

    ax.plot(st[6][0], st[6][1], ls='-', marker='+', label='Signal current', color=c1[i])
    ax.plot(st[6][0], st[6][2], ls='-', marker='^', label='Signal Noise', color=c1[i])

    ax.set_xlabel('Voltage AMS circuit (V)')
    ax.set_ylabel('Diode signal (a.u.)')
    ax.set_yscale('log')
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    ax.set_title(st[0][0]+r' at 2$\,$nA')
    just_save(results_path / 'Voltage/', st[0][1], legend=True)

# (dark_voltage, dark_current, dark_std), (signal_voltage, signal_current, signal_std)
# Start the plotting:
fig, ax = plt.subplots()
for i, st in enumerate(store):
    ax.plot(st[5][0], st[5][1], ls='-', marker='+', color=c3[i])
    ax.plot(st[5][0], st[5][2], ls='-', marker='^', color=c3[i])

    ax.plot(st[6][0], st[6][1], ls='-', marker='+', label=st[0][0], color=c1[i])
    ax.plot(st[6][0], st[6][2], ls='-', marker='^', color=c1[i])

ax.set_xlabel('Voltage AMS circuit (V)')
ax.set_ylabel('Diode signal (a.u.)')
ax.set_yscale('log')
ax.set_ylim(ax.get_ylim())
ax.set_title('Comparison of signal / dark currents and noise'+r' at 2$\,$nA')
just_save(results_path / 'Voltage/', '_comparison_', legend=True)

fig, ax = plt.subplots()
for i, st in enumerate(store):
    ax.plot(st[5][0], st[5][1], ls='-', marker='+', color=c3[i])
    ax.plot(st[5][0], st[5][2], ls='-', marker='^', color=c3[i])

    ax.plot(st[6][0], st[6][1], ls='-', marker='+', label=st[0][0], color=c1[i])
    ax.plot(st[6][0], st[6][2], ls='-', marker='^', color=c1[i])

ax.set_xlabel('Voltage AMS circuit (V)')
ax.set_ylabel('Diode signal (a.u.)')
ax.set_yscale('log')
ax.set_ylim(ax.get_ylim())
ax.set_xlim(ax.get_xlim())

just_save(results_path / 'Voltage/', '_comparison_', legend=True)

fig, ax = plt.subplots()
ax.set_xlim(0.6, 2.05)
for i, st in enumerate(store):
    if (st[5][0]-st[6][0]).all() == 0:
        ax.plot(st[5][0], st[6][1] - st[5][1], ls='-', marker='+', label=st[0][0], color=c2[i])

ax.set_xlabel('Voltage AMS circuit (V)')
ax.set_ylabel('Diode signal (a.u.)')
ax.set_yscale('log')
ax.set_ylim(10**2, ax.get_ylim()[1])
ax.set_title('Comparison of netto signal'+r' at 2$\,$nA')
just_save(results_path / 'Voltage/', '_netto_', legend=True)


fig, ax = plt.subplots()
ax.set_xlim(0.6, 2.05)
for i, st in enumerate(store):
    if (st[5][0]-st[6][0]).all() == 0:
        ax.plot(st[5][0], (st[6][1] - st[5][1]) / st[5][2], ls='-', marker='+', label=st[0][0], color=c2[i])

ax.set_xlabel('Voltage AMS circuit (V)')
ax.set_ylabel('Ratio of netto signal and dark noise')
ax.set_yscale('log')
ax.set_ylim(10**0, ax.get_ylim()[1])
ax.set_title('Comparison of sensitivity'+r' at 2$\,$nA')
just_save(results_path / 'Voltage/', '_sensitivity_', legend=True)

fig, ax = plt.subplots()
ax.set_xlim(0.6, 2.05)
for i, st in enumerate(store):
    if (st[5][0]-st[6][0]).all() == 0:
        ax.plot(st[5][0], (st[6][1] - st[5][1]) / st[6][2], ls='-', marker='+', label=st[0][0], color=c2[i])

ax.set_xlabel('Voltage AMS circuit (V)')
ax.set_ylabel('Ratio of netto signal and signal noise')
ax.set_yscale('log')
ax.set_ylim(10**0, ax.get_ylim()[1])
ax.set_title('Comparison of signal quality'+r' at 2$\,$nA')
just_save(results_path / 'Voltage/', '_signalquality_', legend=True)

# Ratio to smallest signal
fig, ax = plt.subplots()
ax.set_xlim(0.6, 2.05)
for i, st in enumerate(store):
    if i == 1:
        continue
    if (st[5][0]-st[6][0]).all() == 0 and (st[6][0]-store[1][6][0]).all() == 0:
        if i == 2:
            ax.plot(st[5][0], (st[6][1] - st[5][1]) / 0.86 / (store[1][6][1]-store[1][5][1]), ls='-', marker='+', label=st[0][0], color=c2[i])
        else:
            ax.plot(st[5][0], (st[6][1] - st[5][1]) / (store[1][6][1]-store[1][5][1]), ls='-', marker='+', label=st[0][0], color=c2[i])
        ax.axhline(np.multiply(*st[4][1])/np.multiply(*store[1][4][1]), zorder=-2, ls='--', color=c2[i], alpha=0.5)

ax.set_xlabel('Voltage AMS circuit (V)')
ax.set_ylabel('Ratio of netto signal '+r'(128x0.25x0.5$\,$mm$^2$ line array $\#1$)')
# ax.set_yscale('log')
ax.set_ylim(10**0, 10)
ax.set_title('Ratio between netto signals'+r' at 2$\,$nA')
just_save(results_path / 'Voltage/', '_ratios_', legend=True)