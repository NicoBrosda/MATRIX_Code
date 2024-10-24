import numpy as np

from EvaluationSoftware.main import *


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

        dark_std = np.array([np.mean(i['std']) for i in dark])
        signal_std = np.array([np.mean(i['std']) for i in signal])

        dark_voltage, dark_current, dark_std = dark_voltage[sorting_d], dark_current[sorting_d], dark_std[sorting_d]
        signal_voltage, signal_current, signal_std = signal_voltage[sorting_s], signal_current[sorting_s], signal_std[
            sorting_s]

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
            # print('-'*50)
            # print(np.mean(arr))
            # print(np.std(arr))
            # print(dark_std[i])
            # print(signal_std[i])

            for j, diff in enumerate(arr):
                if diff < np.mean(arr) - 2.5*signal_std[i] and np.std(arr) > dark_std[i]:
                    track[j] += 1
        track = np.reshape(track, instance.diode_dimension)
        # print(track)
        dark_current = np.array([np.mean(i['signal'][track < 3]) for i in dark])
        signal_current = np.array([np.mean(i['signal'][track < 3]) for i in signal])

        dark_std = np.array([np.mean(i['std'][track < 3]) for i in dark])
        signal_std = np.array([np.mean(i['std'][track < 3]) for i in signal])
        dark_current, dark_std = dark_current[sorting_d], dark_std[sorting_d]
        signal_current, signal_std = signal_current[sorting_s], signal_std[sorting_s]
    else:
        dark_current = np.array([np.mean(i['signal']) for i in dark])
        dark_std = np.array([np.mean(i['std']) for i in dark])
        dark_voltage, dark_current, dark_std = dark_voltage[sorting_d], dark_current[sorting_d], dark_std[sorting_d]

        signal_current = np.array([np.mean(i['signal']) for i in signal])
        signal_std = np.array([np.mean(i['std']) for i in signal])
        signal_voltage, signal_current, signal_std = signal_voltage[sorting_s], signal_current[sorting_s], signal_std[
            sorting_s]

    return (np.array(dark_voltage), np.array(dark_current), np.array(dark_std)), (np.array(signal_voltage), np.array(signal_current), np.array(signal_std))