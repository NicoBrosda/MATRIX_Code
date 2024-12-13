import numpy as np

from EvaluationSoftware.main import *
from scipy.optimize import curve_fit


def filter_signal_diodes(signal_voltage, signal_current, signal_std, dark_voltage, dark_current, dark_std, limit=3,
                         return_mean: str or bool = True):
    # Introduce a minimum threshold for the signal to exceed
    cache = []
    for i, v in enumerate(dark_voltage):
        for j, v2 in enumerate(signal_voltage):
            if dark_voltage[i] == signal_voltage[j]:
                diff = signal_current[j] - dark_current[i]
                cache.append(diff)
                break
    cache = np.array(cache)

    track = np.zeros_like(cache[0])
    for i, arr in enumerate(cache):

        for j, diff in enumerate(arr):
            if diff < np.mean(arr) - 2.5 * np.mean(signal_std[i]) and np.std(arr) > np.mean(dark_std[i]):
                track[j] += 1
    # print(track)
    if return_mean == 'track':
        return track < limit
    elif return_mean:
        return np.array([np.mean(i[track < limit]) for i in signal_current]), \
            np.array([np.mean(i[track < limit]) for i in signal_std]), \
            np.array([np.mean(i[track < limit]) for i in dark_current]), \
            np.array([np.mean(i[track < limit]) for i in dark_std])
    else:
        return np.array([i[track < limit] for i in signal_current]), np.array([i[track < limit] for i in signal_std]), \
            np.array([i[track < limit] for i in dark_current]), np.array([i[track < limit] for i in dark_std])


def voltage_analysis(folder_path, dark_crit, signal_crit, instance, filter_working_diodes=True):
    instance.set_measurement(folder_path, dark_crit)
    instance.load_measurement()
    dark = instance.measurement_data

    instance.set_measurement(folder_path, signal_crit)
    instance.load_measurement()
    signal = instance.measurement_data

    dark_voltage = np.array([i['voltage'] for i in dark])
    sorting_d = np.argsort(dark_voltage)

    signal_voltage = np.array([i['voltage'] for i in signal])
    sorting_s = np.argsort(signal_voltage)

    if filter_working_diodes:
        dark_current = np.array([i['signal'].flatten() for i in dark])
        signal_current = np.array([i['signal'].flatten() for i in signal])

        dark_std = np.array([i['std'].flatten() for i in dark])
        signal_std = np.array([i['std'].flatten() for i in signal])

        dark_voltage, dark_current, dark_std = dark_voltage[sorting_d], dark_current[sorting_d], dark_std[sorting_d]
        signal_voltage, signal_current, signal_std = signal_voltage[sorting_s], signal_current[sorting_s], signal_std[
            sorting_s]

        signal_current, signal_std, dark_current, dark_std = filter_signal_diodes(signal_voltage, signal_current,
                                                                                  signal_std, dark_voltage, dark_current,
                                                                                  dark_std, limit=3)
    else:
        dark_current = np.array([np.mean(i['signal']) for i in dark])
        dark_std = np.array([np.mean(i['std']) for i in dark])
        dark_voltage, dark_current, dark_std = dark_voltage[sorting_d], dark_current[sorting_d], dark_std[sorting_d]

        signal_current = np.array([np.mean(i['signal']) for i in signal])
        signal_std = np.array([np.mean(i['std']) for i in signal])
        signal_voltage, signal_current, signal_std = signal_voltage[sorting_s], signal_current[sorting_s], signal_std[
            sorting_s]

    return (np.array(dark_voltage), np.array(dark_current), np.array(dark_std)), \
        (np.array(signal_voltage), np.array(signal_current), np.array(signal_std))


def linearity(folder_path, results_path, crit, dark_crit, instance, voltage_dependent=True):
    def linear_func2(x, a, b):
        return a*x + b

    def linear_func(x, a):
        return a*x

    def exp_func(x, a, b):
        return a * (x**1/2) + b

    # Load in all the data for the linearity measurement
    instance.set_measurement(folder_path, crit)
    instance.load_measurement()
    signal = instance.measurement_data

    # Find the different currents available from this data
    currents = np.sort(np.array(list(set([i['current'] for i in signal]))))

    cache_currents = [[] for i in currents]
    for sig in signal:
        cache_currents[np.argwhere(currents == sig['current']).flatten()[0]].append(sig)

    # Load and sort the dark measurements
    instance.set_measurement(folder_path, dark_crit)
    instance.load_measurement()
    dark = instance.measurement_data

    _dark_voltage = np.array([i['voltage'] for i in dark])
    sorting_d = np.argsort(_dark_voltage)

    _dark_current = np.array([i['signal'].flatten() for i in dark])
    _dark_std = np.array([i['std'].flatten() for i in dark])
    _dark_voltage, _dark_current, _dark_std = _dark_voltage[sorting_d], _dark_current[sorting_d], _dark_std[sorting_d]

    # For each current filter the signal and dark current for the signal diodes and matching voltages
    for j, sig in enumerate(cache_currents[::-1]):
        if currents[len(cache_currents)-1-j] is None:
            continue
        signal = sig
        signal_voltage = np.array([i['voltage'] for i in signal])
        sorting_s = np.argsort(signal_voltage)
        signal_current = np.array([i['signal'].flatten() for i in signal])
        signal_std = np.array([i['std'].flatten() for i in signal])
        signal_voltage, signal_current, signal_std = signal_voltage[sorting_s], signal_current[sorting_s], signal_std[
            sorting_s]

        if len(_dark_voltage) < len(signal_voltage):
            dark_voltage, dark_current, dark_std = signal_voltage, np.zeros_like(signal_current), np.zeros_like(signal_std)
            linear_func = linear_func2
        else:
            fvp = [np.argwhere(_dark_voltage == i)[0][0] for i in signal_voltage]
            dark_voltage, dark_current, dark_std = _dark_voltage[fvp], _dark_current[fvp], _dark_std[fvp]

        if j == 0:
            track = filter_signal_diodes(signal_voltage, signal_current, signal_std, dark_voltage, dark_current,
                                         dark_std, return_mean='track')

        cache_currents[len(cache_currents)-1-j] = [[signal_voltage[i], signal_current[i][track], signal_std[i][track],
                                                    dark_current[i][track],
                                                    dark_std[i][track]] for i in range(len(signal_voltage))]

    # Reduce the used voltages to the minimum intersection
    sets = [set([cache_currents[i][j][0] for j, voltage in enumerate(cache_currents[i])]) for i, current in enumerate(currents)]
    intersection = set.intersection(*sets)
    # print(intersection)
    for i in range(len(cache_currents)):
        if len(cache_currents[i][0]) > len(intersection):
            cache = []
            for j in range(len(cache_currents[i])):
                if cache_currents[i][j][0] in intersection:
                    cache.append(cache_currents[i][j])
            cache_currents[i] = cache

    # -----------------------------------------------------------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------------------------------------------------------
    # Plot netto signal mean - signal std mean - dark std mean vs proton current / logarithmic scale per voltage
    r2s_lin = []
    r2s_std = []
    spread = []
    for j, voltage in enumerate(cache_currents[0]):
        voltage = voltage[0]
        fig, ax = plt.subplots()
        netto_signals = np.array([cache_currents[i][j][1]-cache_currents[i][j][3] for i, cur in enumerate(currents)])
        signal_std = np.array([np.mean(cache_currents[i][j][2]) for i, cur in enumerate(currents)])
        dark_std = np.array([np.mean(cache_currents[i][j][4]) for i, cur in enumerate(currents)])

        # linear fit for each single diode
        r2s = []
        params = []
        for k in range(np.shape(netto_signals)[1]):
            y_data = netto_signals[:, k]
            popt, pcov = curve_fit(linear_func, currents, y_data)
            residuals = y_data - linear_func(currents, *popt)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            if ss_tot == 0:
                continue
            r_squared = 1 - (ss_res / ss_tot)
            r2s.append(r_squared)
            params.append([popt, pcov])

        # print(np.mean([i[0][0] for i in params]), np.std([i[0][0] for i in params]))
        print(np.mean([i[0] for i in params], axis=0))
        print(np.shape([i[0] for i in params]))
        ax.plot(currents, linear_func(currents, *np.mean([i[0] for i in params], axis=0)), ls='--', c='r', zorder=3)
        # ax.plot(currents, linear_func(currents, *np.mean([i[0][0] for i in params])), ls='--', c='r', zorder=3)
        # print(np.mean([i[0][1] for i in params]), np.std([i[0][1] for i in params]))

        # print('-'*50)
        # print(dark_std)

        r2s_st = []
        params_std = []
        for k in range(np.shape(netto_signals)[1]):
            y_data = np.array(signal_std)
            popt, pcov = curve_fit((lambda x, a: exp_func(x, a, b=dark_std)), currents, y_data)
            residuals = y_data - exp_func(currents, *popt, b=dark_std)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            if ss_tot == 0:
                continue
            r_squared = 1 - (ss_res / ss_tot)
            r2s_st.append(r_squared)
            params_std.append([popt, pcov])

        # print(np.mean([i[0][0] for i in params_std]), np.std([i[0][0] for i in params_std]))
        ax.plot(currents, exp_func(currents, np.mean([i[0][0] for i in params_std]), dark_std), ls='--', c='r', zorder=3)

        ax.plot(currents, np.mean(netto_signals, axis=1), c='k', marker='x')
        ax.plot(currents, [np.sqrt(signal_std[i]**2+np.std(netto_signals[i])**2) for i, cur in enumerate(currents)], c='k', marker='^')
        ax.plot(currents, signal_std, c='b', ls='-', marker='^')
        ax.plot(currents, [np.std(netto_signals[i]) for i, cur in enumerate(currents)], c='g', ls='-', marker='^')
        ax.plot(currents, dark_std, c='k', marker='v')
        [ax.plot(currents, netto_signals[:, k], c='grey', alpha=0.6, zorder=-1, marker='x') for k in range(np.shape(netto_signals)[1])]

        ax.set_xlabel('Proton current at Faraday cup (nA)')
        ax.set_ylabel('Netto signal (a.u.)')
        ax.set_yscale('log')  
        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())
        ax.text(*transform_axis_to_data_coordinates(ax, [0.04, 0.93]), r'Signal linear fit $\bar{\mathrm{R}}^2$'+' = {x:.5f}'.format(x=np.mean(r2s)), fontsize=12)
        ax.text(*transform_axis_to_data_coordinates(ax, [0.04, 0.88]), r'Std sqrt fit $\bar{\mathrm{R}}^2$'+' = {x:.5f}'.format(x=np.mean(r2s_st)), fontsize=12)
        ax.text(*transform_axis_to_data_coordinates(ax, [0.04, 0.81]), r'{v:.1f}$\,$V'.format(v=voltage), fontsize=15)
        just_save(results_path / ('Linearity/' + str(crit) + '/'), 'Linearity_' + str(voltage) + 'V_', legend=False)
        plt.close('all')
        r2s_lin.append(np.mean(r2s))
        r2s_std.append(np.mean(r2s_st))
        spread.append([np.std(netto_signals[i])/np.mean(netto_signals[i]) if np.mean(netto_signals[i]) != 0 else 1 for i, cur in enumerate(currents)])

    # Plot: spread of diode response (np.std(m) of linear fit) vs voltage
    voltages = [i[0] for i in cache_currents[0]]
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    print(np.shape(voltages))
    print(np.shape(r2s_lin))
    print(np.shape(r2s_std))
    print(np.shape(spread))

    r2s_lin = np.array(r2s_lin)
    r2s_lin[((1 < r2s_lin) | (r2s_lin < 0))] = 0
    r2s_std = np.array(r2s_std)
    r2s_std[((1 < r2s_std) | (r2s_std < 0))] = 0
    ax.plot(voltages, r2s_lin, c='b', ls='-', marker='x', label='Linear fit signal level')
    ax.plot(voltages, r2s_std, c='b', ls=':', marker='^', label='Sqrt fit signal std')
    ax2.plot(voltages, np.mean(np.array(spread)*100, axis=1), c='r', label='Inhomogeneity of diode response', ls='-', marker='*')

    ax.set_xlabel('Voltage AMS circuit (V)')
    ax.set_ylabel(r'Agreement with fit model $\bar{\mathrm{R}}^2$')
    ax2.set_ylabel(r'Std / signal level ($\%$)')
    ax.set_xlim(max(0.66, min(voltages)), 2.02)
    ax.set_ylim([0.95, 1.008])
    ax2.set_ylim(5, 15)
    ax.set_title('Accuracy of linear (sqrt) growth of diode signal (std) \n and inhomogeneity of diodes response', fontsize=12)
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    just_save(results_path / ('Linearity/'+str(crit)+'/'), 'VoltageComp', legend=True)
    plt.close('all')