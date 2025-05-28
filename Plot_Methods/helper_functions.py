import pandas as pd
import scipy.constants as spc  # v. 1.11.1
from pathlib import Path
import pathlib
import os
import numpy as np  # v. 1.25.1


def wv_to_e(wv):
    # Takes wv in nm and returns E in meV
    return spc.h * spc.c * 1000 / (wv*10**(-9) * spc.e)


def e_to_wv(e):
    # Takes E in eV and returns wv in nm
    return spc.h * spc.c * 10**9 / (e * spc.e)


# Function that specifies the conversion function applied to x-ticks for getting the ticks of the second x-axis. In this
# case I just use the wavelength in nm to e in meV conversion func defined above.
def x_conversion(x):
    return wv_to_e(x)


def find_10power(array) -> int:
    # Returns the power of base 10 of the mean of the given array
    if not isinstance(array, np.ndarray):
        array = np.array(array)

    mean = np.abs(np.mean(array))
    i = 0
    while mean // 10**i > 9:
        i += 1

    return i


def remove_outliers(array, rolling_median=3, deviation=2, others=None) -> np.ndarray or list:
    # Automatically scans for single data points that are not fitting other measurement points
    if not isinstance(array, pd.DataFrame):
        array = pd.DataFrame(array)

    mean = array.rolling(window=rolling_median, center=True, min_periods=1).median()
    std = np.mean(array.rolling(window=rolling_median, center=True, min_periods=1).std())
    drop = np.where(np.abs(array - mean) > deviation * std, 1, 0)

    if others is None:
        return array[drop != 1]
    else:
        for i in range(len(others)):
            if not isinstance(others[i], pd.DataFrame):
                others[i] = pd.DataFrame(others[i])
            others[i] = others[i][drop != 1]
        return array[drop != 1], *others


def is_colorbar(ax):
    """
    Guesses whether a set of Axes is home to a colorbar

    :param ax: Axes instance

    :return: bool
        True if the x xor y axis satisfies all of the following and thus looks like it's probably a colorbar:
            No ticks, no tick labels, no axis label, and range is (0, 1)
    """
    xcb = (len(ax.get_xticks()) == 0) and (len(ax.get_xticklabels()) == 0) and (len(ax.get_xlabel()) == 0) and \
          (ax.get_xlim() == (0, 1))
    ycb = (len(ax.get_yticks()) == 0) and (len(ax.get_yticklabels()) == 0) and (len(ax.get_ylabel()) == 0) and \
          (ax.get_ylim() == (0, 1))
    return xcb != ycb  # != is effectively xor in this case, since xcb and ycb are both bool


def interpolate(x_given, x_inter, y_given) -> np.array:
    def line_segment(x, x1, x2, y1, y2):
        return ((x * (y2 - y1)) / (x2 - x1)) - ((x1 * (y2 - y1)) / (x2 - x1)) + y1

    counts_inter = []
    for wv in x_inter:
        if wv in x_given:
            counts_inter.append(y_given[np.where(x_given == wv)[0][0]])
        else:
            diff = wv - x_given
            if wv >= x_given[-1]:
                key2 = len(x_given)-1
                key1 = key2 - 1
            elif wv <= x_given[0]:
                key1 = np.argsort(diff[diff > 0])[0]
                key2 = key1 + 1
            else:
                key1 = np.argsort(diff[diff > 0])[0]
                key2 = key1 + 1
            x1, x2 = x_given[key1], x_given[key2]
            y1, y2 = y_given[key1], y_given[key2]
            counts_inter.append(line_segment(wv, x1, x2, y1, y2))
    return np.array(counts_inter)


def transform_data_to_axis_coordinates(axis, data_coordinates) -> 'coordinates referenced to axis':
    if isinstance(data_coordinates[0], (float, int)):
        return axis.transAxes.inverted().transform(axis.transData.transform((data_coordinates[0], data_coordinates[1])))


def transform_axis_to_data_coordinates(axis, axis_coordinates: list) -> 'coordinates referenced to data':
    if isinstance(axis_coordinates[0], (float, int)):
        return axis.transData.inverted().transform(axis.transAxes.transform((axis_coordinates[0], axis_coordinates[1])))


def list_check(name, list_):
    value = False
    for i in list_:
        if i in name:
            value = True
    return value


def array_txt_file_search(array, blacklist=[], searchlist=None, txt_file=True, file_suffix=None):
    txt_files = []
    for i in array:
        if isinstance(i, pathlib.PurePath):
            i = str(i)
        if txt_file:
            if '.TXT' in i or '.txt' in i or '.npz' in i:
                if not list_check(i, blacklist):
                    if searchlist is None:
                        txt_files.append(i)
                    else:
                        if list_check(i, searchlist):
                            txt_files.append(i)
        elif file_suffix is not None:
            if file_suffix in i:
                if not list_check(i, blacklist):
                    if searchlist is None:
                        txt_files.append(i)
                    else:
                        if list_check(i, searchlist):
                            txt_files.append(i)
        else:
            if not list_check(i, blacklist):
                if searchlist is None:
                    txt_files.append(i)
                else:
                    if list_check(i, searchlist):
                        txt_files.append(i)
    return txt_files


def group(input_list, group_range):
    class Group:
        def __init__(self, value):
            self.mean = value
            self.start_value = value
            self.members = 1

        def add(self, value):
            self.mean = (self.mean * self.members + value)/(self.members + 1)
            self.members += 1

        def check_add(self, value):
            return (self.mean * self.members + value)/(self.members + 1)

    if not isinstance(input_list, (np.ndarray, pd.DataFrame)):
        input_list = np.array(input_list)
    input_list = input_list.flatten()

    groups = []
    for j in input_list:
        if len(groups) == 0:
            groups.append(Group(j))
            continue
        in_group = False
        for group in groups:
            if group.mean - group_range <= j <= group.mean + group_range:
                # check for drift of the group:
                if group.start_value - group_range < group.check_add(j) < group.start_value + group_range:
                    group.add(j)
                    in_group = True
                    break

        if not in_group:
            groups.append(Group(j))

    return [g.mean for g in groups if g.members != 0]


