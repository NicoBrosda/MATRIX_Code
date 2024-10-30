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
        if np.NaN in x or pd.NA in x:
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
            data1['Add' + str(i)] = np.NaN
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
            data1['Add' + str(i)] = np.NaN
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
            data1['Add' + str(i)] = np.NaN
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
    # signal, std = np.concatenate((signal[1::2], signal[0::2])), np.concatenate((std[1::2], std[0::2]))
    signal, std = signal[0::2], std[0::2]
    return {'signal': np.reshape(signal, (instance.diode_dimension[0], instance.diode_dimension[1]//2)),
            'std': np.reshape(std, (instance.diode_dimension[0], instance.diode_dimension[1]//2)),
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
            data1['Add' + str(i)] = np.NaN
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

