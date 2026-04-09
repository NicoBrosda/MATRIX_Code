from copy import deepcopy

import numpy as np
import pandas as pd

# This is the standard blueprint of a readout module (in terms of inputs and outputs)
def readout_example(path_to_data_file, instance):
    signal = np.zeros(instance.diodes_dimension)
    std = np.zeros(instance.diodes_dimension)
    dict_for_other_params = dict()
    return {'signal': signal, 'std': std, 'dict': dict_for_other_params}


def ams_otsus_readout(path_to_data_file, instance, subtract_background=True):
    def threshold_otsu(x, *args, **kwargs) -> float or None:
        # hist, bins, range=None
        """Find the threshold value for a bimodal histogram using the Otsu method.

      If you have a distribution that is bimodal (AKA with two peaks, with a valley
      between them), then you can use this to find the location of that valley, that
      splits the distribution into two.

      From the SciKit Image threshold_otsu implementation:
      https://github.com/scikit-image/scikit-image/blob/70fa904eee9ef370c824427798302551df57afa1/skimage/filters/thresholding.py#L312
      """
        if np.nan in x or pd.NA in x:
            return None
        counts, bin_edges = np.histogram(x, *args, **kwargs)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

        # class probabilities for all possible thresholds
        weight1 = np.cumsum(counts)
        weight2 = np.cumsum(counts[::-1])[::-1]
        # class means for all possible thresholds
        with np.errstate(divide='ignore', invalid='ignore'):
            mean1 = np.cumsum(counts * bin_centers) / weight1
            mean2 = (np.cumsum((counts * bin_centers)[::-1]) / weight2[::-1])[::-1]

        # Clip ends to align class 1 and class 2 variables:
        # The last value of ``weight1``/``mean1`` should pair with zero values in
        # ``weight2``/``mean2``, which do not exist.
        variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

        idx = np.argmax(variance12)
        threshold = bin_centers[idx]
        return threshold

    el = instance.diode_dimension[0] * instance.diode_dimension[1]
    columns_used = el
    if columns_used > 128:
        columns_used = 128
        if columns_used - (instance.excluded == True).sum() > 128:
            print('The ams_readout module is not suitable for arrays with more than 128 diodes, because only 128 channels '
                  'are existent. Check if the array structure was inserted correctly and if the correct readout module was '
                  'picked')
    detect = len(pd.read_csv(path_to_data_file, delimiter=',', nrows=1, header=None).columns)
    data1 = pd.read_csv(path_to_data_file, delimiter=',', nrows=1, header=None, usecols=range(130, detect))
    data2 = pd.read_csv(path_to_data_file, delimiter=',', usecols=range(columns_used+2))
    if len(data2.columns) - len(data1.columns) > 0:
        for i in range(len(data2.columns) - len(data1.columns)):
            data1['Add' + str(i)] = np.nan
    data1 = data1.set_axis(list(data2.columns), axis=1)
    data = pd.concat([data1, data2])

    thresholds = []
    signals = []
    signal_std = []
    darks = []
    dark_std = []
    for i, col in enumerate(data):
        # Column 0 is voltage information
        if i == 0:
            continue
        # Column 1 is sample number
        if i == 1:
            continue
        # Column 3 = measurement channel 1 = array reference 0
        j = i - 2
        if instance.excluded.flatten()[j]:
            signals.append(0)
            darks.append(0)
            signal_std.append(0)
            dark_std.append(0)
            thresholds.append(None)
        try:
            threshold = threshold_otsu(data[col])
        except ValueError:
            threshold = None
        thresholds.append(threshold)

        if threshold is not None:
            dark_std.append(np.std(data[col][data[col] < threshold]))
            sig = np.mean(data[col][data[col] > threshold])
            dar = np.mean(data[col][data[col] < threshold])
            std = np.std(data[col][data[col] > threshold])
            if not np.isnan(sig) and not np.isnan(dar):
                signals.append(sig - dar)
                signal_std.append(std)
                darks.append(dar)
            elif sig is not np.nan and dar is np.nan:
                signals.append(sig)
                signal_std.append(std)
                darks.append(0)
            else:
                signals.append(0)
                signal_std.append(0)
                darks.append(0)
        else:
            signals.append(0)
            darks.append(0)
            signal_std.append(0)
            dark_std.append(0)
    # Now the data needs to be filled in an array structure that resembles: instance.diode_dimension
    # Check if there are enough values to fill the instance.diode_dimension array, and append 0 otherwise
    if len(signals) < el:
        while len(signals) < el:
            signals.append(0)
            darks.append(0)
            signal_std.append(0)
            dark_std.append(0)
            thresholds.append(None)
    signals, darks, signal_std, dark_std, thresholds = np.array(signals)[:el], np.array(darks)[:el], np.array(signal_std)[:el], \
        np.array(dark_std)[:el], np.array(thresholds)[:el]
    return {'signal': np.reshape(signals, instance.diode_dimension),
            'std': np.reshape(signal_std, instance.diode_dimension),
            'dict': {'thresholds': np.reshape(thresholds, instance.diode_dimension),
                     'dark': np.reshape(darks, instance.diode_dimension),
                     'dark_std': np.reshape(dark_std, instance.diode_dimension)}}


def ams_constant_signal_readout(path_to_data_file, instance):
    el = instance.diode_dimension[0] * instance.diode_dimension[1]
    columns_used = el
    if columns_used > 128:
        columns_used = 128
        if columns_used - (instance.excluded == True).sum() > 128:
            print(
                'The ams_readout module is not suitable for arrays with more than 128 diodes, because only 128 channels '
                'are existent. Check if the array structure was inserted correctly and if the correct readout module was '
                'picked')
    detect = len(pd.read_csv(path_to_data_file, delimiter=',', nrows=1, header=None).columns)
    data1 = pd.read_csv(path_to_data_file, delimiter=',', nrows=1, header=None, usecols=range(130, detect))
    data2 = pd.read_csv(path_to_data_file, delimiter=',', usecols=range(columns_used+2))
    if len(data2.columns) - len(data1.columns) > 0:
        for i in range(len(data2.columns) - len(data1.columns)):
            data1['Add' + str(i)] = np.nan
    data1 = data1.set_axis(list(data2.columns), axis=1)
    data = pd.concat([data1, data2])

    signals = []
    signal_std = []

    '''
    # Load in one or multiple dark, measurements - calculate their mean - subtract from the signal
    dark = []
    if subtract_background and np.shape(instance.dark_files)[0] > 0:
        for file2 in instance.dark_files:
            dark.append(ams_constant_signal_readout(file2, instance, subtract_background=False)['signal'].flatten())
        dark = np.mean(np.array(dark), axis=0)
    

    if np.shape(dark)[0] == 0:
        dark = np.zeros(columns_used)
    '''

    for i, col in enumerate(data):
        # Column 0 is voltage information
        if i == 0:
            continue
        # Column 1 is sample number
        if i == 1:
            continue
        # Column 3 = measurement channel 1 = array reference 0
        j = i - 2
        if instance.excluded.flatten()[j]:
            signals.append(0)
            signal_std.append(0)
        try:
            cache = np.mean(data[col])  # - dark[j]
            cache_std = np.std(data[col])
            if not np.isnan(cache):
                signals.append(cache)
                signal_std.append(cache_std)
            else:
                signals.append(0)
                signal_std.append(0)
        except ValueError:
            signals.append(0)
            signal_std.append(0)
    # Now the data needs to be filled in an array structure that resembles: instance.diode_dimension
    # Check if there are enough values to fill the instance.diode_dimension array, and append 0 otherwise
    if len(signals) < el:
        while len(signals) < el:
            signals.append(0)
            signal_std.append(0)
            # dark = np.append(dark, 0)
    signals, signal_std = np.array(signals)[:el], np.array(signal_std)[:el]
    # dark = dark[:el]
    return {'signal': np.reshape(signals, instance.diode_dimension),
            'std': np.reshape(signal_std, instance.diode_dimension),
            'dict': {}}
            # 'dict': {'dark': np.reshape(dark, instance.diode_dimension)}}


def design_readout(path_to_data_file, instance):
    el = instance.diode_dimension[0] * instance.diode_dimension[1]
    columns_used = el
    if columns_used > 128:
        columns_used = 128
        if columns_used - (instance.excluded == True).sum() > 128:
            print(
                'The ams_readout module is not suitable for arrays with more than 128 diodes, because only 128 channels '
                'are existent. Check if the array structure was inserted correctly and if the correct readout module was '
                'picked')
    data = pd.read_csv(path_to_data_file, delimiter=',', usecols=range(columns_used + 1))

    signals = []
    signal_std = []

    for i, col in enumerate(data):
        # Column 0 is voltage information
        if i == 0:
            continue
        # Column 2 = measurement channel 1 = array reference 0
        j = i - 1
        if instance.excluded.flatten()[j]:
            signals.append(0)
            signal_std.append(0)
        try:
            cache = np.mean(data[col])
            cache_std = np.std(data[col])
            if not np.isnan(cache):
                signals.append(cache)
                signal_std.append(cache_std)
            else:
                signals.append(0)
                signal_std.append(0)
        except ValueError:
            signals.append(0)
            signal_std.append(0)
    # Now the data needs to be filled in an array structure that resembles: instance.diode_dimension
    # Check if there are enough values to fill the instance.diode_dimension array, and append 0 otherwise
    if len(signals) < el:
        while len(signals) < el:
            signals.append(0)
            signal_std.append(0)
    signals, signal_std = np.array(signals)[:el], np.array(signal_std)[:el]
    return {'signal': np.reshape(signals, instance.diode_dimension),
            'std': np.reshape(signal_std, instance.diode_dimension),
            'dict': {}}


def ams_channel_assignment_readout(path_to_data_file, instance, channel_assignment=None, sample_size=None):
    el = instance.diode_dimension[0] * instance.diode_dimension[1]
    columns_used = el
    # '''
    if channel_assignment is None:
        channel_assignment = [i for i in range(el)]
    # ordering = np.argsort([channel_assignment[i] for i in channel_assignment])
    # '''
    if columns_used > 128:
        columns_used = 128
        if columns_used - (instance.excluded == True).sum() > 128:
            print(
                'The ams_readout module is not suitable for arrays with more than 128 diodes, because only 128 channels '
                'are existent. Check if the array structure was inserted correctly and if the correct readout module was '
                'picked')
    detect = len(pd.read_csv(path_to_data_file, delimiter=',', nrows=1, header=None).columns)
    data1 = pd.read_csv(path_to_data_file, delimiter=',', nrows=1, header=None, usecols=range(130, detect))
    data2 = pd.read_csv(path_to_data_file, delimiter=',', usecols=range(columns_used+2))
    if len(data2.columns) - len(data1.columns) > 0:
        for i in range(len(data2.columns) - len(data1.columns)):
            data1['Add' + str(i)] = np.nan
    data1 = data1.set_axis(list(data2.columns), axis=1)
    data = pd.concat([data1, data2])

    signals = []
    signal_std = []

    '''
    # Load in one or multiple dark, measurements - calculate their mean - subtract from the signal
    dark = []
    if subtract_background and np.shape(instance.dark_files)[0] > 0:
        for file2 in instance.dark_files:
            dark.append(ams_constant_signal_readout(file2, instance, subtract_background=False)['signal'].flatten())
        dark = np.mean(np.array(dark), axis=0)

    if np.shape(dark)[0] == 0:
        dark = np.zeros(columns_used)
    '''

    for i, col in enumerate(data):
        # Column 0 is voltage information
        if i == 0:
            continue
        # Column 1 is sample number
        if i == 1:
            continue
        # Column 3 = measurement channel 1 = array reference 0
        j = i - 2
        if instance.excluded.flatten()[j]:
            signals.append(0)
            signal_std.append(0)
        try:
            if sample_size is None:
                cache = np.mean(data[col])  # - dark[j]
                cache_std = np.std(data[col])
            else:
                if sample_size > len(data[col]):
                    sample_size = len(data[col])
                cache = np.mean(data[col][0:sample_size])  # - dark[j]
                cache_std = np.std(data[col][0:sample_size])
            if not np.isnan(cache):
                signals.append(cache)
                signal_std.append(cache_std)
            else:
                signals.append(0)
                signal_std.append(0)
        except ValueError:
            signals.append(0)
            signal_std.append(0)
    # Now the data needs to be filled in an array structure that resembles: instance.diode_dimension
    # Check if there are enough values to fill the instance.diode_dimension array, and append 0 otherwise
    if len(signals) < el:
        while len(signals) < el:
            signals.append(0)
            signal_std.append(0)
            # dark = np.append(dark, 0)
    signals, signal_std = np.array(signals)[:el][channel_assignment], np.array(signal_std)[:el][channel_assignment]

    return {'signal': np.reshape(signals, instance.diode_dimension),
            'std': np.reshape(signal_std, instance.diode_dimension),
            'dict': {}}


def ams_2line_readout(path_to_data_file, instance, channel_assignment=None, sample_size=None):
    values = ams_channel_assignment_readout(path_to_data_file, instance, channel_assignment, sample_size)
    signal, std = values['signal'].flatten(), values['std'].flatten()
    signal, std = np.concatenate((signal[1::2], signal[0::2])), np.concatenate((std[1::2], std[0::2]))
    # signal, std = signal[0::2], std[0::2]
    # return {'signal': np.reshape(signal, (instance.diode_dimension[0], instance.diode_dimension[1]//2)), 'std': np.reshape(std, (instance.diode_dimension[0], instance.diode_dimension[1]//2)), 'dict': values['dict']}
    return {'signal': np.reshape(signal, (instance.diode_dimension[0], instance.diode_dimension[1])),
            'std': np.reshape(std, (instance.diode_dimension[0], instance.diode_dimension[1])),
            'dict': values['dict']}


def ams_2D_assignment_readout(path_to_data_file, instance, channel_assignment=None, sample_size=None):
    el = instance.diode_dimension[0] * instance.diode_dimension[1]
    columns_used = el
    # '''
    if channel_assignment is None:
        channel_assignment = [i for i in range(el)]
    if not isinstance(channel_assignment, np.ndarray):
        channel_assignment = np.array(channel_assignment)

    channel_assignment = channel_assignment.flatten()

    # ordering = np.argsort([channel_assignment[i] for i in channel_assignment])
    # '''
    detect = len(pd.read_csv(path_to_data_file, delimiter=',', nrows=1, header=None).columns)

    not_in_assignment = []
    if np.shape(channel_assignment)[0] < detect - 132:
        not_in_assignment = [i for i in range(detect - 132) if i not in channel_assignment]
        columns_used += np.shape(not_in_assignment)[0]

    if columns_used > 128:
        columns_used = 128
        if columns_used - (instance.excluded == True).sum() > 128:
            print(
                'The ams_readout module is not suitable for arrays with more than 128 diodes, because only 128 channels '
                'are existent. Check if the array structure was inserted correctly and if the correct readout module was '
                'picked')

    data1 = pd.read_csv(path_to_data_file, delimiter=',', nrows=1, header=None, usecols=range(130, detect))
    data2 = pd.read_csv(path_to_data_file, delimiter=',', usecols=range(columns_used + 2))
    if len(data2.columns) - len(data1.columns) > 0:
        for i in range(len(data2.columns) - len(data1.columns)):
            data1['Add' + str(i)] = np.nan
    data1 = data1.set_axis(list(data2.columns), axis=1)
    data = pd.concat([data1, data2])

    signals = []
    signal_std = []

    '''
    # Load in one or multiple dark, measurements - calculate their mean - subtract from the signal
    dark = []
    if subtract_background and np.shape(instance.dark_files)[0] > 0:
        for file2 in instance.dark_files:
            dark.append(ams_constant_signal_readout(file2, instance, subtract_background=False)['signal'].flatten())
        dark = np.mean(np.array(dark), axis=0)

    if np.shape(dark)[0] == 0:
        dark = np.zeros(columns_used)
    '''

    for i, col in enumerate(data):
        # Column 0 is voltage information
        if i == 0:
            continue
        # Column 1 is sample number
        if i == 1:
            continue
        # Column 3 = measurement channel 1 = array reference 0
        j = i - 2

        if instance.excluded.flatten()[j-np.shape(not_in_assignment)[0]]:
            signals.append(0)
            signal_std.append(0)
        try:
            if sample_size is None:
                cache = np.mean(data[col])  # - dark[j]
                cache_std = np.std(data[col])
            else:
                if sample_size > len(data[col]):
                    sample_size = len(data[col])
                cache = np.mean(data[col][0:sample_size])  # - dark[j]
                cache_std = np.std(data[col][0:sample_size])
            if not np.isnan(cache):
                signals.append(cache)
                signal_std.append(cache_std)
            else:
                signals.append(0)
                signal_std.append(0)
        except ValueError:
            signals.append(0)
            signal_std.append(0)
    # Now the data needs to be filled in an array structure that resembles: instance.diode_dimension
    # Check if there are enough values to fill the instance.diode_dimension array, and append 0 otherwise
    if len(signals) < columns_used:
        while len(signals) < columns_used:
            signals.append(0)
            signal_std.append(0)
            # dark = np.append(dark, 0)
    signals, signal_std = np.array(signals)[:columns_used][channel_assignment], np.array(signal_std)[:columns_used][channel_assignment]

    return {'signal': np.reshape(signals, instance.diode_dimension),
            'std': np.reshape(signal_std, instance.diode_dimension),
            'dict': {}}


def ams_sample_noise_readout(path_to_data_file, instance, channel_assignment=None, sample_size=None):
    el = instance.diode_dimension[0] * instance.diode_dimension[1]
    columns_used = el
    # '''
    if channel_assignment is None:
        channel_assignment = [i for i in range(el)]
    # ordering = np.argsort([channel_assignment[i] for i in channel_assignment])
    # '''
    if columns_used > 128:
        columns_used = 128
        if columns_used - (instance.excluded == True).sum() > 128:
            print(
                'The ams_readout module is not suitable for arrays with more than 128 diodes, because only 128 channels '
                'are existent. Check if the array structure was inserted correctly and if the correct readout module was '
                'picked')
    detect = len(pd.read_csv(path_to_data_file, delimiter=',', nrows=1, header=None).columns)
    data1 = pd.read_csv(path_to_data_file, delimiter=',', nrows=1, header=None, usecols=range(130, detect))
    data2 = pd.read_csv(path_to_data_file, delimiter=',', usecols=range(columns_used+2))
    if len(data2.columns) - len(data1.columns) > 0:
        for i in range(len(data2.columns) - len(data1.columns)):
            data1['Add' + str(i)] = np.nan
    data1 = data1.set_axis(list(data2.columns), axis=1)
    data = pd.concat([data1, data2])

    signals = []
    signal_std = []

    '''
    # Load in one or multiple dark, measurements - calculate their mean - subtract from the signal
    dark = []
    if subtract_background and np.shape(instance.dark_files)[0] > 0:
        for file2 in instance.dark_files:
            dark.append(ams_constant_signal_readout(file2, instance, subtract_background=False)['signal'].flatten())
        dark = np.mean(np.array(dark), axis=0)

    if np.shape(dark)[0] == 0:
        dark = np.zeros(columns_used)
    '''

    for i, col in enumerate(data):
        # Column 0 is voltage information
        if i == 0:
            continue
        # Column 1 is sample number
        if i == 1:
            continue
        # Column 3 = measurement channel 1 = array reference 0
        j = i - 2
        if instance.excluded.flatten()[j]:
            signals.append(0)
            signal_std.append(0)

        try:
            if sample_size is None:
                cache = np.array(data[col])  # - dark[j]
                cache_std = np.std(data[col])
            else:
                if sample_size > len(data[col]):
                    sample_size = len(data[col])
                cache = np.array(data[col][0:sample_size])  # - dark[j]
                cache_std = np.std(data[col][0:sample_size])
            if not np.isnan(cache.any()):
                signals.append(cache)
                signal_std.append(cache_std)
            else:
                if sample_size is None:
                    signals.append(np.zeros(len(data[col])))
                else:
                    signals.append(np.zeros(sample_size))
                signal_std.append(0)
        except ValueError:
            signals.append(0)
            signal_std.append(0)
    # Now the data needs to be filled in an array structure that resembles: instance.diode_dimension
    # Check if there are enough values to fill the instance.diode_dimension array, and append 0 otherwise
    if len(signals) < el:
        while len(signals) < el:
            signals.append(0)
            signal_std.append(0)
            # dark = np.append(dark, 0)
    signals, signal_std = np.array(signals)[:el][channel_assignment], np.array(signal_std)[:el][channel_assignment]
    return {'signal': signals,
            'std': np.reshape(signal_std, instance.diode_dimension),
            'dict': {}}


def ams_2D_assignment_readout_WPE(path_to_data_file, instance, channel_assignment=None, sample_size=None):
    el = instance.diode_dimension[0] * instance.diode_dimension[1]
    columns_used = el
    # '''
    if channel_assignment is None:
        channel_assignment = [i for i in range(el)]
    if not isinstance(channel_assignment, np.ndarray):
        channel_assignment = np.array(channel_assignment)

    channel_assignment = channel_assignment.flatten()

    # ordering = np.argsort([channel_assignment[i] for i in channel_assignment])
    # '''
    detect = len(pd.read_csv(path_to_data_file, delimiter=',', nrows=1, header=None).columns)

    not_in_assignment = []
    if np.shape(channel_assignment)[0] < detect - 132:
        not_in_assignment = [i for i in range(detect - 132) if i not in channel_assignment]
        columns_used += np.shape(not_in_assignment)[0]

    if columns_used > 128:
        columns_used = 128
        if columns_used - (instance.excluded == True).sum() > 128:
            print(
                'The ams_readout module is not suitable for arrays with more than 128 diodes, because only 128 channels '
                'are existent. Check if the array structure was inserted correctly and if the correct readout module was '
                'picked')

    data1 = pd.read_csv(path_to_data_file, delimiter=',', nrows=1, header=None, usecols=range(130, detect))
    data2 = pd.read_csv(path_to_data_file, delimiter=',', usecols=range(columns_used + 2))
    if len(data2.columns) - len(data1.columns) > 0:
        for i in range(len(data2.columns) - len(data1.columns)):
            data1['Add' + str(i)] = np.nan
    data1 = data1.set_axis(list(data2.columns), axis=1)
    data = pd.concat([data1, data2])

    signals = []
    signal_std = []


    # Load in one or multiple dark, measurements - calculate their mean - subtract from the signal
    dark = []
    '''
    instance2 = deepcopy(instance)
    instance2.diode_dimension = [1, 128]
    instance2.excluded = np.full(instance2.diode_dimension, False)
    for file2 in instance.dark_files:
        dark.append(ams_constant_signal_readout(file2, instance)['signal'].flatten())
    dark = np.mean(np.array(dark), axis=0)
    # '''
    if np.shape(dark)[0] == 0:
        dark = np.zeros(columns_used)


    for i, col in enumerate(data):
        # Column 0 is voltage information
        if i == 0:
            continue
        # Column 1 is sample number
        if i == 1:
            continue
        # Column 3 = measurement channel 1 = array reference 0
        j = i - 2

        if instance.excluded.flatten()[j-np.shape(not_in_assignment)[0]]:
            signals.append(0)
            signal_std.append(0)
        try:
            if sample_size is None:
                cache = np.mean(data[col]-dark[j])  # - dark[j]
                cache_std = np.std(data[col])
            elif isinstance(sample_size, list):
                if sample_size[0] < 0:
                    sample_size = 0
                if sample_size[1] > len(data[col]):
                    sample_size = len(data[col])
                cache = np.mean(data[col][sample_size[0]:sample_size[1]]-dark[j])  # - dark[j]
                cache_std = np.std(data[col][sample_size[0]:sample_size[1]])
            else:
                if sample_size > len(data[col]):
                    sample_size = len(data[col])
                cache = np.mean(data[col][0:sample_size]-dark[j])  # - dark[j]
                cache_std = np.std(data[col][0:sample_size])
            if not np.isnan(cache):
                signals.append(cache)
                signal_std.append(cache_std)
            else:
                signals.append(0)
                signal_std.append(0)
        except ValueError:
            signals.append(0)
            signal_std.append(0)
    # Now the data needs to be filled in an array structure that resembles: instance.diode_dimension
    # Check if there are enough values to fill the instance.diode_dimension array, and append 0 otherwise
    if len(signals) < columns_used:
        while len(signals) < columns_used:
            signals.append(0)
            signal_std.append(0)
            # dark = np.append(dark, 0)
    signals, signal_std = np.array(signals)[:columns_used][channel_assignment], np.array(signal_std)[:columns_used][channel_assignment]

    return {'signal': np.reshape(signals, instance.diode_dimension),
            'std': np.reshape(signal_std, instance.diode_dimension),
            'dict': {}}


def ams_2D_assignment_readout_WPE2(path_to_data_file, instance, channel_assignment=None):
    el = instance.diode_dimension[0] * instance.diode_dimension[1]
    columns_used = el
    # '''
    if channel_assignment is None:
        channel_assignment = [i for i in range(el)]
    if not isinstance(channel_assignment, np.ndarray):
        channel_assignment = np.array(channel_assignment)

    channel_assignment = channel_assignment.flatten()

    # ordering = np.argsort([channel_assignment[i] for i in channel_assignment])
    # '''
    detect = len(pd.read_csv(path_to_data_file, delimiter=',', nrows=1, header=None).columns)

    not_in_assignment = []
    if np.shape(channel_assignment)[0] < detect - 132:
        not_in_assignment = [i for i in range(detect - 132) if i not in channel_assignment]
        columns_used += np.shape(not_in_assignment)[0]

    if columns_used > 128:
        columns_used = 128
        if columns_used - (instance.excluded == True).sum() > 128:
            print(
                'The ams_readout module is not suitable for arrays with more than 128 diodes, because only 128 channels '
                'are existent. Check if the array structure was inserted correctly and if the correct readout module was '
                'picked')

    data1 = pd.read_csv(path_to_data_file, delimiter=',', nrows=1, header=None, usecols=range(130, detect))
    data2 = pd.read_csv(path_to_data_file, delimiter=',', usecols=range(columns_used + 2))
    if len(data2.columns) - len(data1.columns) > 0:
        for i in range(len(data2.columns) - len(data1.columns)):
            data1['Add' + str(i)] = np.nan
    data1 = data1.set_axis(list(data2.columns), axis=1)
    data = pd.concat([data1, data2])

    signals = []
    signal_std = []

    # '''
    # Load in one or multiple dark, measurements - calculate their mean - subtract from the signal
    dark = []
    instance2 = deepcopy(instance)
    instance2.diode_dimension = [1, 128]
    instance2.excluded = np.full(instance2.diode_dimension, False)
    for file2 in instance.dark_files:
        dark.append(ams_channel_assignment_readout(file2, instance2)['signal'].flatten())
    dark = np.mean(np.array(dark), axis=0)

    if np.shape(dark)[0] == 0:
        dark = np.zeros(columns_used)
    # '''

    for i, col in enumerate(data):
        # Column 0 is voltage information
        if i == 0:
            continue
        # Column 1 is sample number
        if i == 1:
            continue
        # Column 3 = measurement channel 1 = array reference 0
        j = i - 2

        if instance.excluded.flatten()[j-np.shape(not_in_assignment)[0]]:
            signals.append(0)
            signal_std.append(0)
        try:
            cache = data[col] - dark[j]
            signals.append(cache)

        except ValueError:
            signals.append(0)
            signal_std.append(0)
    # Now the data needs to be filled in an array structure that resembles: instance.diode_dimension
    # Check if there are enough values to fill the instance.diode_dimension array, and append 0 otherwise
    if len(signals) < columns_used:
        while len(signals) < columns_used:
            signals.append(0)
            signal_std.append(0)
            # dark = np.append(dark, 0)
    signals = np.array(signals)[:columns_used]

    return signals


def ams_2D_assignment_readout_WPE3(path_to_data_file, instance, channel_assignment=None, sample_size=None):
    el = instance.diode_dimension[0] * instance.diode_dimension[1]
    columns_used = el
    # '''
    if channel_assignment is None:
        channel_assignment = [i for i in range(el)]
    if not isinstance(channel_assignment, np.ndarray):
        channel_assignment = np.array(channel_assignment)

    channel_assignment = channel_assignment.flatten()

    # ordering = np.argsort([channel_assignment[i] for i in channel_assignment])
    # '''
    detect = len(pd.read_csv(path_to_data_file, delimiter=',', nrows=1, header=None).columns)

    not_in_assignment = []
    if np.shape(channel_assignment)[0] < detect - 132:
        not_in_assignment = [i for i in range(detect - 132) if i not in channel_assignment]
        columns_used += np.shape(not_in_assignment)[0]

    if columns_used > 128:
        columns_used = 128
        if columns_used - (instance.excluded == True).sum() > 128:
            print(
                'The ams_readout module is not suitable for arrays with more than 128 diodes, because only 128 channels '
                'are existent. Check if the array structure was inserted correctly and if the correct readout module was '
                'picked')

    data1 = pd.read_csv(path_to_data_file, delimiter=',', nrows=1, header=None, usecols=range(130, detect))
    data2 = pd.read_csv(path_to_data_file, delimiter=',', usecols=range(columns_used + 2))
    if len(data2.columns) - len(data1.columns) > 0:
        for i in range(len(data2.columns) - len(data1.columns)):
            data1['Add' + str(i)] = np.nan
    data1 = data1.set_axis(list(data2.columns), axis=1)
    data = pd.concat([data1, data2])

    signals = []
    signal_std = []

    # '''
    # Load in one or multiple dark, measurements - calculate their mean - subtract from the signal
    dark = []
    instance2 = deepcopy(instance)
    instance2.diode_dimension = [1, 128]
    instance2.excluded = np.full(instance2.diode_dimension, False)
    for file2 in instance.dark_files:
        dark.append(ams_channel_assignment_readout(file2, instance2)['signal'].flatten())
    dark = np.mean(np.array(dark), axis=0)

    if np.shape(dark)[0] == 0:
        dark = np.zeros(columns_used)
    # '''
    instance2.diode_dimension = [11, 11]
    instance2.excluded = np.full(instance2.diode_dimension, False)

    for i, col in enumerate(data):
        # Column 0 is voltage information
        if i == 0:
            continue
        # Column 1 is sample number
        if i == 1:
            continue
        # Column 3 = measurement channel 1 = array reference 0
        j = i - 2

        if instance.excluded.flatten()[j-np.shape(not_in_assignment)[0]]:
            signals.append(0)
            signal_std.append(0)
        try:
            if sample_size is None:
                cache = np.sum(data[col]-dark[j])  # - dark[j]
                cache_std = np.std(data[col])
            elif isinstance(sample_size, list):
                if sample_size[0] < 0:
                    sample_size = 0
                if sample_size[1] > len(data[col]):
                    sample_size = len(data[col])
                cache = np.sum(data[col][sample_size[0]:sample_size[1]]-dark[j])  # - dark[j]
                cache_std = np.std(data[col][sample_size[0]:sample_size[1]])
            else:
                if sample_size > len(data[col]):
                    sample_size = len(data[col])
                cache = np.sum(data[col][0:sample_size]-dark[j])  # - dark[j]
                cache_std = np.std(data[col][0:sample_size])
            if not np.isnan(cache):
                signals.append(cache)
                signal_std.append(cache_std)
            else:
                signals.append(0)
                signal_std.append(0)
        except ValueError:
            signals.append(0)
            signal_std.append(0)
    # Now the data needs to be filled in an array structure that resembles: instance.diode_dimension
    # Check if there are enough values to fill the instance.diode_dimension array, and append 0 otherwise
    if len(signals) < columns_used:
        while len(signals) < columns_used:
            signals.append(0)
            signal_std.append(0)
            # dark = np.append(dark, 0)

    signals, signal_std = np.array(signals)[:columns_used][channel_assignment], np.array(signal_std)[:columns_used][channel_assignment]

    return {'signal': np.reshape(signals, instance.diode_dimension),
            'std': np.reshape(signal_std, instance.diode_dimension),
            'dict': {}}


def ams_2D_assignment_readout_WPE_choice(path_to_data_file, instance, channel_assignment=None, sample_size=None, version='sum', step=100):
    el = instance.diode_dimension[0] * instance.diode_dimension[1]
    columns_used = el
    # '''
    if channel_assignment is None:
        channel_assignment = [i for i in range(el)]
    if not isinstance(channel_assignment, np.ndarray):
        channel_assignment = np.array(channel_assignment)

    channel_assignment = channel_assignment.flatten()

    # ordering = np.argsort([channel_assignment[i] for i in channel_assignment])
    # '''
    detect = len(pd.read_csv(path_to_data_file, delimiter=',', nrows=1, header=None).columns)

    not_in_assignment = []
    if np.shape(channel_assignment)[0] < detect - 132:
        not_in_assignment = [i for i in range(detect - 132) if i not in channel_assignment]
        columns_used += np.shape(not_in_assignment)[0]

    if columns_used > 128:
        columns_used = 128
        if columns_used - (instance.excluded == True).sum() > 128:
            print(
                'The ams_readout module is not suitable for arrays with more than 128 diodes, because only 128 channels '
                'are existent. Check if the array structure was inserted correctly and if the correct readout module was '
                'picked')

    data1 = pd.read_csv(path_to_data_file, delimiter=',', nrows=1, header=None, usecols=range(130, detect))
    data2 = pd.read_csv(path_to_data_file, delimiter=',', usecols=range(columns_used + 2))
    if len(data2.columns) - len(data1.columns) > 0:
        for i in range(len(data2.columns) - len(data1.columns)):
            data1['Add' + str(i)] = np.nan
    data1 = data1.set_axis(list(data2.columns), axis=1)
    data = pd.concat([data1, data2])

    signals = []
    signal_std = []

    # '''
    # Load in one or multiple dark, measurements - calculate their mean - subtract from the signal
    dark = []
    instance2 = deepcopy(instance)
    instance2.diode_dimension = [1, 128]
    instance2.excluded = np.full(instance2.diode_dimension, False)
    for file2 in instance.dark_files:
        dark.append(ams_channel_assignment_readout(file2, instance2)['signal'].flatten())
    dark = np.mean(np.array(dark), axis=0)

    if np.shape(dark)[0] == 0:
        dark = np.zeros(columns_used)
    # '''
    instance2.diode_dimension = [11, 11]
    instance2.excluded = np.full(instance2.diode_dimension, False)

    for i, col in enumerate(data):
        # Column 0 is voltage information
        if i == 0:
            continue
        # Column 1 is sample number
        if i == 1:
            continue
        # Column 3 = measurement channel 1 = array reference 0
        j = i - 2

        if instance.excluded.flatten()[j-np.shape(not_in_assignment)[0]]:
            signals.append(0)
            signal_std.append(0)
        try:
            if sample_size is None:
                if version == 'sum':
                    cache = np.sum(data[col]-dark[j])  # - dark[j]
                    cache = np.sum(data[col]-np.mean(data[col][0:1000]))  # - dark[j]

                    if np.mean(data[col][0:1000] - dark[j]) > 10:
                        print(col, '-' * 30)
                        print('Sum', np.sum(data[col][0:1000] - dark[j]))
                        print('Dark', dark[j])
                        print('Dark in Measurement', np.mean(data[col][0:1000]))
                        print('Dark after original dark subtraction', np.mean(data[col][0:1000] - dark[j]))
                        print('-' * 30)

                elif version == 'mean':
                    cache = np.mean(data[col] - dark[j])
                elif version == 'mean_filtered':
                    max_index = np.argmax(data[col] - dark[j])
                    cache = np.mean(data[col][max_index-step:max_index+step] - dark[j])
                elif version == 'max':
                    cache = np.max(data[col] - dark[j])
                elif version == 'n_max':
                    cache = np.mean(np.sort(data[col] - dark[j], axis=None)[-step:])
                cache_std = np.std(data[col])
            elif isinstance(sample_size, list):
                if sample_size[0] < 0:
                    sample_size = 0
                if sample_size[1] > len(data[col]):
                    sample_size = len(data[col])
                if version == 'sum':
                    cache = np.sum(data[col][sample_size[0]:sample_size[1]]-dark[j])  # - dark[j]
                    print('-' * 30)
                    print('Sum', np.sum(data[col][0:1000] - dark[j]))
                    print('Dark', dark[j])
                    print('Dark in Measurement', np.mean(data[col][0:1000]))
                    print('Dark after original dark subtraction', np.mean(data[col][0:1000] - dark[j]))
                    print('-' * 30)

                elif version == 'mean':
                    cache = np.mean(data[col][sample_size[0]:sample_size[1]]-dark[j])
                elif version == 'mean_filtered':
                    max_index = np.argmax(data[col][sample_size[0]:sample_size[1]]-dark[j])
                    cache = np.mean(data[col][sample_size[0]:sample_size[1]][max_index-step:max_index+step] - dark[j])
                elif version == 'max':
                    cache = np.max(data[col][sample_size[0]:sample_size[1]]-dark[j])
                elif version == 'n_max':
                    cache = np.mean(np.sort(data[col][sample_size[0]:sample_size[1]]-dark[j], axis=None)[-step:])
                cache_std = np.std(data[col][sample_size[0]:sample_size[1]])
            else:
                if sample_size > len(data[col]):
                    sample_size = len(data[col])
                if version == 'sum':
                    cache = np.sum(data[col][0:sample_size]-dark[j])  # - dark[j]
                    print('-' * 30)
                    print('Sum', np.sum(data[col][0:1000] - dark[j]))
                    print('Dark', dark[j])
                    print('Dark in Measurement', np.mean(data[col][0:1000]))
                    print('Dark after original dark subtraction', np.mean(data[col][0:1000] - dark[j]))
                    print('-' * 30)

                elif version == 'mean':
                    cache = np.mean(data[col][0:sample_size]-dark[j])
                elif version == 'mean_filtered':
                    max_index = np.argmax(data[col][0:sample_size]-dark[j])
                    cache = np.mean(data[col][0:sample_size][max_index-step:max_index+step] - dark[j])
                elif version == 'max':
                    cache = np.max(data[col][0:sample_size]-dark[j])
                elif version == 'n_max':
                    cache = np.mean(np.sort(data[col][0:sample_size]-dark[j], axis=None)[-step:])
                cache_std = np.std(data[col][0:sample_size])
            if not np.isnan(cache):
                signals.append(cache)
                signal_std.append(cache_std)
            else:
                signals.append(0)
                signal_std.append(0)
        except ValueError:
            signals.append(0)
            signal_std.append(0)
    # Now the data needs to be filled in an array structure that resembles: instance.diode_dimension
    # Check if there are enough values to fill the instance.diode_dimension array, and append 0 otherwise
    if len(signals) < columns_used:
        while len(signals) < columns_used:
            signals.append(0)
            signal_std.append(0)
            # dark = np.append(dark, 0)

    signals, signal_std = np.array(signals)[:columns_used][channel_assignment], np.array(signal_std)[:columns_used][channel_assignment]

    return {'signal': np.reshape(signals, instance.diode_dimension),
            'std': np.reshape(signal_std, instance.diode_dimension),
            'dict': {}}


def ams_2D_assignment_WPE_fast(path_to_data_file, instance, channel_assignment=None,
                           sample_size=None, version='mean', step=100,
                           keep_first_row=False):
    """
    Fast AMS 2D readout with vectorized operations.

    Parameters
    ----------
    path_to_data_file : str
        Path to CSV file.
    instance : object
        Analyzer instance containing diode_dimension and excluded info.
    channel_assignment : list or ndarray, optional
        List of active channels. Defaults to all.
    sample_size : list or int, optional
        Range of samples to average [start, end] or single integer.
    version : str
        'sum', 'mean', 'mean_filtered', 'max', 'n_max'
    step : int
        Step size for 'mean_filtered' or 'n_max'
    keep_first_row : bool
        Keep the first malformed measurement row.

    Returns
    -------
    dict
        {'signal': 2D array, 'std': 2D array, 'dict': {}}
    """
    el = np.prod(instance.diode_dimension)
    if channel_assignment is None:
        channel_assignment = np.arange(el)
    channel_assignment = np.array(channel_assignment).flatten()

    # ---------------------------
    # Load CSV once, handle first row
    # ---------------------------
    if keep_first_row:
        # read header + first row manually
        with open(path_to_data_file, 'r') as f:
            header = f.readline().strip().split(',')
            first_row = f.readline().strip().split(',')
        # convert first_row to float
        first_row = np.array(first_row, dtype=float)
        # read remaining rows
        data_rest = pd.read_csv(path_to_data_file, delimiter=',', header=0, skiprows=[1])
        data_rest = data_rest.to_numpy()
        # combine
        data = np.vstack([first_row, data_rest])
        columns_used = data.shape[1]
    else:
        # skip malformed first row
        data = pd.read_csv(path_to_data_file, delimiter=',', header=0, skiprows=[1])
        data = data.to_numpy()
        columns_used = data.shape[1]

    # ---------------------------
    # Handle dark subtraction
    # ---------------------------
    dark = []
    from copy import deepcopy
    instance2 = deepcopy(instance)
    instance2.diode_dimension = [1, 128]
    instance2.excluded = np.full(instance2.diode_dimension, False)

    if hasattr(instance, 'dark_files') and instance.dark_files:
        for file2 in instance.dark_files:
            # simple mean over dark file
            dark_data = ams_channel_assignment_readout(file2, instance2)['signal'].flatten()
            dark.append(dark_data)
        dark = np.mean(np.array(dark), axis=0)
    else:
        dark = np.zeros(columns_used)

    # ---------------------------
    # Determine sample slice
    # ---------------------------
    if sample_size is None:
        s_start, s_end = 0, data.shape[0]
    elif isinstance(sample_size, list) and len(sample_size) == 2:
        s_start = max(0, sample_size[0])
        s_end = min(data.shape[0], sample_size[1])
    else:
        s_start, s_end = 0, min(data.shape[0], int(sample_size))

    data_slice = data[s_start:s_end, 2:]  # skip DAC + Sample columns
    dark_vec = dark[:data_slice.shape[1]]

    # ---------------------------
    # Excluded mask
    # ---------------------------
    excluded_flat = instance.excluded.flatten()
    mask = np.ones(data_slice.shape[1], dtype=bool)
    mask[excluded_flat] = False
    mask[channel_assignment] = True  # ensure channel assignment

    # ---------------------------
    # Vectorized signal computation
    # ---------------------------
    # subtract dark
    data_slice -= dark_vec

    signals = np.zeros(data_slice.shape[1])
    signal_std = np.zeros(data_slice.shape[1])

    if version == 'sum':
        signals[mask] = np.sum(data_slice[:, mask], axis=0)
        signal_std[mask] = np.std(data_slice[:, mask], axis=0)
    elif version == 'mean':
        signals[mask] = np.mean(data_slice[:, mask], axis=0)
        signal_std[mask] = np.std(data_slice[:, mask], axis=0)
    elif version == 'max':
        signals[mask] = np.max(data_slice[:, mask], axis=0)
        signal_std[mask] = np.std(data_slice[:, mask], axis=0)
    elif version == 'n_max':
        # use partial sort for speed
        sorted_part = np.partition(data_slice[:, mask], -step, axis=0)[-step:]
        signals[mask] = np.mean(sorted_part, axis=0)
        signal_std[mask] = np.std(data_slice[:, mask], axis=0)
    elif version == 'mean_filtered':
        # find column-wise argmax
        max_idx = np.argmax(data_slice[:, mask], axis=0)
        # compute mean around max index (with clipping)
        for i, idx in enumerate(max_idx):
            s = max(idx - step, 0)
            e = min(idx + step + 1, data_slice.shape[0])
            signals[np.where(mask)[0][i]] = np.mean(data_slice[s:e, np.where(mask)[0][i]])
            signal_std[np.where(mask)[0][i]] = np.std(data_slice[s:e, np.where(mask)[0][i]])

    # reshape to diode array
    signals = signals[:el][channel_assignment]
    signal_std = signal_std[:el][channel_assignment]

    return {'signal': np.reshape(signals, instance.diode_dimension),
            'std': np.reshape(signal_std, instance.diode_dimension),
            'dict': {}}


def ams_2D_assignment_fast_avg(path_to_data_file, instance,
                               channel_assignment=None,
                               sample_size=None,
                               keep_first_row=True):
    """
    Fast AMS 2D readout for averaging only, ignoring exclusions.
    Handles malformed CSVs where the first measurement row is appended to the header.
    """
    el = np.prod(instance.diode_dimension)

    # ---------------------------
    # Setup channel assignment
    # ---------------------------
    if channel_assignment is None:
        channel_assignment = np.arange(el)
    else:
        channel_assignment = np.array(channel_assignment).flatten()

    # ---------------------------
    # Read CSV properly
    # ---------------------------
    # Skip header, read DAC + Sample + 128 channels
    data = pd.read_csv(path_to_data_file, delimiter=',', header=None,
                       skiprows=1, usecols=range(2 + 128)).to_numpy()

    if keep_first_row:
        with open(path_to_data_file, 'r') as f:
            header_line = f.readline().strip().split(',')  # full header + first row appended
            # extract only the first measurement values (columns 2..2+128)
            first_row = np.array(header_line[2 + 128:], dtype=float)
        # prepend to the rest of the CSV data
        data = np.vstack([first_row[np.newaxis, :], data])

    # ---------------------------
    # Sample slicing
    # ---------------------------
    if sample_size is None:
        s_start, s_end = 0, data.shape[0]
    elif isinstance(sample_size, list) and len(sample_size) == 2:
        s_start = max(0, sample_size[0])
        s_end = min(data.shape[0], sample_size[1])
    else:
        s_start = 0
        s_end = min(data.shape[0], int(sample_size))

    data_slice = data[s_start:s_end, 2:]  # skip DAC + Sample

    # ---------------------------
    # Average & std (vectorized)
    # ---------------------------
    signals_full = np.mean(data_slice, axis=0) # shape: 128
    signal_std_full = np.std(data_slice, axis=0)

    # ---------------------------
    # Apply channel_assignment and reshape
    # ---------------------------
    signals = signals_full[channel_assignment]
    signal_std = signal_std_full[channel_assignment]

    return {'signal': np.reshape(signals, instance.diode_dimension),
            'std': np.reshape(signal_std, instance.diode_dimension),
            'dict': {}}


def ams_2D_assignment_frame(path_to_data_file, instance,
                               channel_assignment=None,
                               keep_first_row=True,
                               ) -> np.ndarray:
    """
    Fast AMS 2D readout for averaging only, ignoring exclusions.
    Handles malformed CSVs where the first measurement row is appended to the header.
    """
    el = np.prod(instance.diode_dimension)

    # ---------------------------
    # Setup channel assignment
    # ---------------------------
    if channel_assignment is None:
        channel_assignment = np.arange(el)
    else:
        channel_assignment = np.array(channel_assignment).flatten()

    # ---------------------------
    # Read CSV properly
    # ---------------------------
    # Skip header, read DAC + Sample + 128 channels
    data = pd.read_csv(path_to_data_file, delimiter=',', header=None,
                       skiprows=1, usecols=range(2 + 128)).to_numpy()

    # print(np.shape(data))
    if keep_first_row:
        with open(path_to_data_file, 'r') as f:
            header_line = f.readline().strip().split(',')  # full header + first row appended
            # extract only the first measurement values (columns 2..2+128)
            first_row = np.array(header_line[2 + 128:], dtype=float)
        # prepend to the rest of the CSV data
        data = np.vstack([first_row[np.newaxis, :], data])

    # ---------------------------
    # Sample slicing
    # ---------------------------
    frame_bunch_size = FrameReadoutConfig.bunch_size
    if frame_bunch_size is None:
        frame_bunch_size = 1
    elif frame_bunch_size < 1:
        frame_bunch_size = 1
    elif frame_bunch_size > data.shape[0]:
        frame_bunch_size = data.shape[0]
    elif not isinstance(frame_bunch_size, int):
        frame_bunch_size = int(frame_bunch_size)

    N = data.shape[0]
    indices = np.arange(N)
    groups = np.array_split(indices, frame_bunch_size)

    return np.array([data[g].mean(axis=0)[2:][channel_assignment] for g in groups])


class FrameReadoutConfig:
    bunch_size = 1


def ams_fast_time_profiles(path_to_data_file, instance,
                               channel_assignment=None,
                               sample_size=None,
                               keep_first_row=True,
                                excluded=0,
                           ):
    """
    Fast AMS 2D readout for averaging only, ignoring exclusions.
    Handles malformed CSVs where the first measurement row is appended to the header.
    """
    el = np.prod(instance.diode_dimension) - excluded

    # ---------------------------
    # Setup channel assignment
    # ---------------------------
    if channel_assignment is None:
        channel_assignment = np.arange(el+excluded)
    else:
        channel_assignment = np.array(channel_assignment).flatten()

    # ---------------------------
    # Read CSV properly
    # ---------------------------

    # Skip header, read DAC + Sample + 128 channels
    data = pd.read_csv(path_to_data_file, delimiter=',', header=None,
                       skiprows=1, usecols=range(2 + 128)).to_numpy()

    if keep_first_row:
        with open(path_to_data_file, 'r') as f:
            header_line = f.readline().strip().split(',')  # full header + first row appended
            # extract only the first measurement values (columns 2..2+128)
            first_row = np.array(header_line[2 + 128:], dtype=float)
        # prepend to the rest of the CSV data
        data = np.vstack([first_row[np.newaxis, :], data])

    # ---------------------------
    # Sample slicing
    # ---------------------------
    if sample_size is None:
        s_start, s_end = 0, data.shape[0]
    elif isinstance(sample_size, list) and len(sample_size) == 2:
        s_start = max(0, sample_size[0])
        s_end = min(data.shape[0], sample_size[1])
    else:
        s_start = 0
        s_end = min(data.shape[0], int(sample_size))

    data_slice = data[s_start:s_end, 2:]  # skip DAC + Sample

    if excluded > 0:
        data_slice = np.concatenate((data_slice, np.ones((data_slice.shape[0], excluded))), axis=1)

    # ---------------------------
    # Apply channel_assignment and reshape
    # ---------------------------
    signals = data_slice[:, channel_assignment].T

    return {'signal': np.reshape(signals, (instance.diode_dimension[0], instance.diode_dimension[1], len(data_slice))), 'std': {}, 'dict': {}}


def ams_2line_fast_avg(path_to_data_file, instance,
                               channel_assignment=None,
                               sample_size=None,
                               keep_first_row=True):
    """
    Fast AMS 2D readout for averaging only, ignoring exclusions.
    Handles malformed CSVs where the first measurement row is appended to the header.
    """
    el = np.prod(instance.diode_dimension)

    # ---------------------------
    # Setup channel assignment
    # ---------------------------
    if channel_assignment is None:
        channel_assignment = np.arange(el)
    else:
        channel_assignment = np.array(channel_assignment).flatten()

    # ---------------------------
    # Read CSV properly
    # ---------------------------
    # Skip header, read DAC + Sample + 128 channels
    data = pd.read_csv(path_to_data_file, delimiter=',', header=None,
                       skiprows=1, usecols=range(2 + 128)).to_numpy()

    if keep_first_row:
        with open(path_to_data_file, 'r') as f:
            header_line = f.readline().strip().split(',')  # full header + first row appended
            # extract only the first measurement values (columns 2..2+128)
            first_row = np.array(header_line[2 + 128:], dtype=float)
        # prepend to the rest of the CSV data
        data = np.vstack([first_row[np.newaxis, :], data])

    # ---------------------------
    # Sample slicing
    # ---------------------------
    if sample_size is None:
        s_start, s_end = 0, data.shape[0]
    elif isinstance(sample_size, list) and len(sample_size) == 2:
        s_start = max(0, sample_size[0])
        s_end = min(data.shape[0], sample_size[1])
    else:
        s_start = 0
        s_end = min(data.shape[0], int(sample_size))

    data_slice = data[s_start:s_end, 2:]  # skip DAC + Sample

    # ---------------------------
    # Average & std (vectorized)
    # ---------------------------
    signals_full = np.mean(data_slice, axis=0) # shape: 128
    signal_std_full = np.std(data_slice, axis=0)

    # ---------------------------
    # Apply channel_assignment and reshape
    # ---------------------------
    signals = signals_full[channel_assignment]
    signal_std = signal_std_full[channel_assignment]

    signals, signal_std = np.concatenate((signals[1::2], signals[0::2])), np.concatenate((signal_std[1::2], signal_std[0::2]))

    return {'signal': np.reshape(signals, instance.diode_dimension),
            'std': np.reshape(signal_std, instance.diode_dimension),
            'dict': {}}


def ams_fast_avg(path_to_data_file, instance,
                               channel_assignment=None,
                               sample_size=None,
                               keep_first_row=True):
    """
    Fast AMS 2D readout for averaging only, ignoring exclusions.
    Handles malformed CSVs where the first measurement row is appended to the header.
    """
    el = np.prod(instance.diode_dimension)

    # ---------------------------
    # Setup channel assignment
    # ---------------------------
    if channel_assignment is None:
        channel_assignment = np.arange(el)
    else:
        channel_assignment = np.array(channel_assignment).flatten()

    # ---------------------------
    # Read CSV properly
    # ---------------------------
    # Skip header, read DAC + Sample + 128 channels
    data = pd.read_csv(path_to_data_file, delimiter=',', header=None,
                       skiprows=1, usecols=range(2 + 128)).to_numpy()

    if keep_first_row:
        with open(path_to_data_file, 'r') as f:
            header_line = f.readline().strip().split(',')  # full header + first row appended
            # extract only the first measurement values (columns 2..2+128)
            first_row = np.array(header_line[2 + 128:], dtype=float)
        # prepend to the rest of the CSV data
        data = np.vstack([first_row[np.newaxis, :], data])

    # ---------------------------
    # Sample slicing
    # ---------------------------
    if sample_size is None:
        s_start, s_end = 0, data.shape[0]
    elif isinstance(sample_size, list) and len(sample_size) == 2:
        s_start = max(0, sample_size[0])
        s_end = min(data.shape[0], sample_size[1])
    else:
        s_start = 0
        s_end = min(data.shape[0], int(sample_size))

    data_slice = data[s_start:s_end, 2:]  # skip DAC + Sample

    # ---------------------------
    # Average & std (vectorized)
    # ---------------------------
    signals_full = np.mean(data_slice, axis=0) # shape: 128
    signal_std_full = np.std(data_slice, axis=0)

    # ---------------------------
    # Apply channel_assignment and reshape
    # ---------------------------
    signals = signals_full[channel_assignment]
    signal_std = signal_std_full[channel_assignment]

    return {'signal': np.reshape(signals, instance.diode_dimension),
            'std': np.reshape(signal_std, instance.diode_dimension),
            'dict': {}}