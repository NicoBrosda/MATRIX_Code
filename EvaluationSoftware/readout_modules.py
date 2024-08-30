import numpy as np
import pandas as pd


# This is the standard blueprint of a readout module (in terms of inputs and outputs)
def readout_example(path_to_data_file, instance):
    signal = np.zeros(instance.diodes_dimension)
    std = np.zeros(instance.diodes_dimension)
    position = (0, 0)
    voltage = 0
    return {'signal': signal, 'std': std, 'pos': position, 'voltage': voltage}


def ams_otsus_readout(path_to_data_file, instance):
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

    detect = len(pd.read_csv(path_to_data_file, delimiter=',', nrows=1, header=None).columns)
    data1 = pd.read_csv(path_to_data_file, delimiter=',', nrows=1, header=None, usecols=range(130, detect))
    data2 = pd.read_csv(path_to_data_file, delimiter=',', usecols=range(66))
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
        if instance.excluded[i]:
            signals.append(0)
            darks.append(0)
            signal_std.append(0)
            dark_std.append(0)
        if i == 0:
            voltage = np.mean(data[col])
        if i == 1:
            continue
        try:
            threshold = threshold_otsu(data[col])
        except ValueError:
            threshold = None
        thresholds.append(threshold)

        if threshold is not None:
            # dark_std.append(np.std(data[col][data[col] < threshold]))
            sig = np.mean(data[col][data[col] > threshold])
            dar = np.mean(data[col][data[col] < threshold])
            std = np.std(data[col][data[col] > threshold])
            if not np.isnan(sig) and not np.isnan(dar):
                signals.append(sig - dar)
                signal_std.append(std)
            elif sig is not np.nan and dar is np.nan:
                signals.append(sig)
                signal_std.append(std)
            else:
                signals.append(0)
                signal_std.append(0)
        else:
            signals.append(0)
            darks.append(0)
            signal_std.append(0)
    return None
    if advanced_output:
        return [np.array(signals)[:len(signals) - len(excluded_channel)],
                np.array(signal_std)[:len(signal_std) - len(excluded_channel)],
                np.array(darks)[:len(darks) - len(excluded_channel)],
                np.array(dark_std)[:len(dark_std) - len(excluded_channel)],
                np.array(thresholds)[:len(thresholds) - len(excluded_channel)]]
    else:
        return np.array(signals)[:len(signals) - len(excluded_channel)]