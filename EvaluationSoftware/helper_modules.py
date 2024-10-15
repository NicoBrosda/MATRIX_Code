import pathlib
from pathlib import Path
import os
import codecs
from copy import deepcopy
import numpy as np
from Plot_Methods.helper_functions import transform_axis_to_data_coordinates, transform_data_to_axis_coordinates
import matplotlib.pyplot as plt
import pandas as pd


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


def path_check(path):
    # Abfangen eines Problems, wenn Pfad nicht mit "/" beendet:
    back = False
    if not isinstance(path, str):
        path = str(path)
        back = True
    '''
    if not path[-1] == '/':
        path = path + '/'
    '''
    path = str(Path(path) / ' ')[:-1]
    if back:
        return Path(path)
    else:
        return path


def save_text(txt, save_path, save_name, newline=False):
    save_path = path_check(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    path = Path(save_path) / save_name
    with codecs.open(path, 'w', 'utf-8', 'strict') as fh:
        for i in txt:
            if newline:
                fh.write(i+'\n')
            else:
                fh.write(i)


class LineShape:
    def __init__(self, list_of_points, distance_mode=False):
        print('A shape with connection lines between the given points will be initiated! Note that for positioning of '
              'the shape the x-positions can be scaled by setting a reference point later on.')
        self.points = np.array(list_of_points)
        self.distance_mode = distance_mode
        if distance_mode:
            cache = deepcopy(self.points)
            for i, point in enumerate(self.points):
                cache[i, 0] += np.sum(self.points[:i, 0])
            self.points = cache
        else:
            self.sort()

    def sort(self):
        self.points = self.points[np.argsort(self.points[:, 0])]

    def print_shape(self):
        print('The points of the current shape will be converted to a printout')
        for i, point in enumerate(self.points):
            print('Point', i, ':', point)

    def mirror(self):
        cache = []
        distance = 0
        for j in range(np.shape(self.points)[0]):
            cache.append([self.points[0, 0] + distance, self.points[::-1][j, 1]])
            if j < np.shape(self.points)[0]-1:
                distance += self.points[::-1][j, 0] - self.points[::-1][j+1, 0]
        self.points = np.array(cache)

    def add_points(self, list_of_points):
        self.points = np.concatenate((self.points, np.array(list_of_points)), axis=0)
        self.sort()

    def position(self, x_middle=None, reference_x=None, y_diff=0):
        if x_middle is not None:
            if reference_x is None:
                np.mean(self.points[:, 1])
            self.points[:, 0] += (x_middle - reference_x)
        self.points[:, 1] += y_diff

    def add_to_plot(self, y_min=0.2, y_max=0.8, ax=None, *args, **kwargs):
        if ax is None:
            ax = plt.gca()
        if ax.get_xlim()[0] > min(min(self.points[:, 0]), ax.get_xlim()[0]) or ax.get_xlim()[1] < max(max(self.points[:, 0]), ax.get_xlim()[1]):
            ax.set_xlim(min(min(self.points[:, 0]), ax.get_xlim()[0]), max(max(self.points[:, 0]), ax.get_xlim()[1]))
        ax.set_ylim(ax.get_ylim())

        y = self.points[:, 1]
        y -= min(y)
        y = y * (y_max-y_min) / max(y)
        y += y_min
        y = [transform_axis_to_data_coordinates(ax, [0.5, j])[1] for j in y]
        y_min = transform_axis_to_data_coordinates(ax, [0.5, y_min])[1]
        fill = ax.fill_between(self.points[:, 0], y, y_min, *args, **kwargs)
        return fill

    def calculate_value(self, x_coordinate):
        def line_segment(x, x1, y1, x2, y2):
            return ((x * (y2-y1)) / (x2-x1)) - ((x1 * (y2-y1)) / (x2 - x1)) + y1
        value = 0
        if self.points[0, 0] < x_coordinate < self.points[-1, 0]:
            distance = x_coordinate - self.points[:, 0]
            index = np.argmin(distance[distance >= 0])
            value = line_segment(x_coordinate, *self.points[index], *self.points[index+1])
        elif self.points[0, 0] >= x_coordinate:
            value = self.points[0, 1]
        else:
            value = self.points[-1, 1]
        return value

    def get_plot_value(self, value, y_min=0.2, y_max=0.8, ax=None):
        if ax is None:
            ax = plt.gca()
        y = self.points[:, 1]
        if value < min(y):
            value = min(y)
        elif value > max(y):
            value = max(y)
        y -= min(y)
        value = (value-min(y)) * (y_max - y_min) / max(y) + y_min
        return transform_axis_to_data_coordinates(ax, [0.5, value])[1]


def group(input_list, group_range):
    class Group:
        def __init__(self, value):
            self.mean = value
            self.start_value = value
            self.members = 1

        def add(self, value):
            self.mean = (self.mean * self.members + value) / (self.members + 1)
            self.members += 1

        def check_add(self, value):
            return (self.mean * self.members + value) / (self.members + 1)

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


def mean_diodes(position_arrays, signal_arrays, instance, direction_switch=0, threshold=0):
    # Group the positions after their recalculation to gain a grid, from which the mean calculation is meaningful
    group_distance = instance.diode_size[direction_switch]
    if len(np.shape(position_arrays)) > 2:
        position_arrays = position_arrays[:, :, direction_switch]
    groups = group(position_arrays.flatten(), group_distance)

    # Calculate the mean for each grouped position, consider only the diode signals that were close to this position
    mean_new = []
    mean_x_new = []
    groups = np.sort(groups)
    for mean in groups:
        indices = []
        for k, channel in enumerate(position_arrays):
            index_min = np.argsort(np.abs(channel - mean))[0]
            if np.abs(channel[index_min] - mean) <= group_distance:
                indices.append(index_min)
            else:
                indices.append(None)
        cache_new = 0
        j_new = 0
        for i in range(len(indices)):
            if indices[i] is not None and signal_arrays[indices[i]][i] >= threshold:
                cache_new += signal_arrays[indices[i]][i]
                j_new += 1
        if j_new > 0:
            mean_new.append(cache_new / j_new)
            mean_x_new.append(mean)
    mean_x_new, mean_new = np.array(mean_x_new), np.array(mean_new)
    return mean_x_new, mean_new


def rename_files(file_path, crit, rename='_{file_name}', file_suffix=None):
    if not isinstance(file_path, pathlib.Path):
        file_path = Path(file_path)
    for file_name in array_txt_file_search(os.listdir(file_path), searchlist=[crit], txt_file=False,
                                           file_suffix=file_suffix):
        os.rename(file_path / file_name, file_path / rename.format(file_name=file_name))
