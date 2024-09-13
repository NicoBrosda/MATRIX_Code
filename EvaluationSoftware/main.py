import os
import pathlib
from tqdm import tqdm
import numpy as np
from pathlib import Path
from helper_modules import array_txt_file_search
from readout_modules import *
from position_parsing_modules import *
from Plot_Methods.plot_standards import *
from normalization_modules import *


class DiodeGeometry:
    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        return self.instance

    def __set__(self, instance, value):
        if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 2:
            self.instance = (value[0], value[1])
        elif isinstance(value, (list, tuple, np.ndarray)) and len(value) == 1:
            self.instance = (value[0], value[0])
        elif isinstance(value, (float, int, complex)) and not isinstance(value, bool):
            self.instance = (value, value)
        else:
            print('No suited input value given for the '+str(self.name)+'. Before further proceeding is possible set '
                                                                        'a regular value!')
            self.instance = None


class Analyzer:

    diode_dimension = DiodeGeometry()
    diode_size = DiodeGeometry()
    diode_spacing = DiodeGeometry()

    def __init__(self, diode_dimension, diode_size, diode_spacing, readout='AMS_evaluation'):
        """
        This is an analyzer class for the readout and data analysis from measurements with diode arrays in the MATRIX
        project. This class aims to be as general as possible to adapt easily to changing diode geometries,
        readout circuits, data treatment methods and data usage. The idea is to pack the critical specific elements into
        several modules and allowing a quick adaptation by choosing the correct module combination for the present
        experimental situation. Thus, other software parts like the image generation can be kept the same.
        This class will cover a rectangular diode pattern with equally sized diodes, which should be sufficient for the
        MATRIX project - three parameters of the diodes are initialized, because they define how e.g. the readout and
        image construction work
        :param diode_dimension: The dimension of the diode array as (n, m) array-like input. n defines the columns (x)
        of the array, m (y) the rows. Note that the choice of n, m defines the axis for following parts. If only one
        float-like input value is given, the array is set as (1, value).
        :param diode_size: Size of one diode as (len_x, len_y) array-like input. If only one float-like input value
        is given, the dimensions are set as quadratic.
        :param diode_spacing: Spacing between diodes as (spacing_x, spacing_y) array-like input. If only one float-like
        input value is given, both spacings will be set equally.
        :param readout: Option to specify the readout circuit used in measurements. Standard is the first read_out used,
        the evaluation kit of the AMS evaluation kit.
        """
        self.diode_dimension = diode_dimension
        self.diode_size = diode_size
        self.diode_spacing = diode_spacing

        self.readout = ams_constant_signal_readout
        self.pos_parser = standard_position
        self.measurement_files = []
        self.measurement_data = []
        self.dark_files = []
        self.norm_factor = np.ones(diode_dimension)
        self.name = ''
        self.map = {'x': [], 'y': [], 'z': []}
        self.excluded = np.full(self.diode_dimension, False)

    def set_measurement(self, path_to_folder, filter_criterion, file_format='.csv', blacklist=['.png', '.pdf', '.jpg']):
        self.name = filter_criterion
        if blacklist is None:
            blacklist = []
        if not isinstance(path_to_folder, pathlib.PurePath):
            path_to_folder = Path(path_to_folder)
        files = os.listdir(path_to_folder)
        measurement_files = array_txt_file_search(files, searchlist=[filter_criterion], blacklist=blacklist,
                                                  file_suffix=file_format, txt_file=False)
        print(len(measurement_files), ' files found in the folder: ', path_to_folder, ' under the search criterion: ',
              filter_criterion)
        self.measurement_files = [Path(path_to_folder) / i for i in measurement_files]

    def set_dark_measurement(self, path_to_folder, filter_criterion='dark', file_format='.csv', blacklist=['.png', '.pdf', '.jpg']):
        if blacklist is None:
            blacklist = []
        if not isinstance(path_to_folder, pathlib.PurePath):
            path_to_folder = Path(path_to_folder)
        files = os.listdir(path_to_folder)
        dark_files = array_txt_file_search(files, searchlist=[filter_criterion], blacklist=blacklist,
                                                  file_suffix=file_format, txt_file=False)
        print(len(dark_files), ' files for background correction found in the folder: ', path_to_folder,
              ' under the search criterion: ', filter_criterion)
        self.dark_files = [Path(path_to_folder) / i for i in dark_files]

    def normalization(self, path_to_folder, filter_criterion, file_format='.csv', blacklist=['.png', '.pdf', '.jpg'],
                      normalization_module=None, cache_save=True, factor_limits=(0, 3), norm_factor=True):
        # Check if factor is already saved and is not needed to be recalculated:
        # if cache_save and os.path.isfile(path_to_folder / (filter_criterion+'_normalization_factor.npy')):
        if cache_save and os.path.isfile(path_to_folder / 'normalization_factor.npy'):
            try:
                # factor = np.load(path_to_folder / (filter_criterion+'_normalization_factor.npy'))
                factor = np.load(path_to_folder / 'normalization_factor.npy')
                factor[((factor < factor_limits[0]) | (factor > factor_limits[1]))] = 0
                if norm_factor:
                    factor = factor / np.mean(factor[factor != 0])
                self.norm_factor = factor.reshape(self.diode_dimension)
                return factor.reshape(self.diode_dimension)
            except ValueError:
                print('Error while loading the factor - note that another file in the given folder might collide with '
                      'the utilized save name "normalization_factor.npy". For safety reasons the calculated factor will'
                      ' not be saved')
                cache_save = False

        if blacklist is None:
            blacklist = []
        if not isinstance(path_to_folder, pathlib.PurePath):
            path_to_folder = Path(path_to_folder)
        files = os.listdir(path_to_folder)
        files = array_txt_file_search(files, searchlist=[filter_criterion], blacklist=blacklist,
                                                  file_suffix=file_format, txt_file=False)
        print(len(files), ' files for normalization found in the folder: ', path_to_folder,
              ' under the search criterion: ', filter_criterion)
        files = [Path(path_to_folder) / i for i in files]

        factor = normalization_module(files, self)

        if cache_save:
            np.save(path_to_folder / (filter_criterion+'_normalization_factor.npy'), factor)

        factor[((factor < factor_limits[0]) | (factor > factor_limits[1]))] = 0

        if norm_factor:
            factor = factor / np.mean(factor[factor != 0])

        self.norm_factor = factor

    def load_measurement(self, readout_module=None, position_parser=None):
        self.measurement_data = []
        if readout_module is None:
            readout_module = self.readout
        else:
            self.readout = readout_module

        if position_parser is None:
            position_parser = self.pos_parser
        else:
            self.pos_parser = position_parser
        for file in tqdm(self.measurement_files):
            pos = position_parser(file)
            cache = readout_module(file, self)
            cache['signal'] = cache['signal']*self.norm_factor
            cache.update({'position': pos})
            self.measurement_data.append(cache)

    def create_map(self, overlay='ignore'):
        x = []
        y = []
        z = []
        for data in self.measurement_data:
            pos = data['position']
            if np.isnan(pos[0] or np.isnan(pos[1])):
                continue
            signal = data['signal']
            for i, column in enumerate(signal):
                x.append(pos[0] + i * (self.diode_size[0] + self.diode_spacing[0]))
                cache_column = []
                for j, row in enumerate(column):
                    y.append(pos[1] + j * (self.diode_size[1] + self.diode_spacing[1]))
                    cache_column.append(row)
                z.append(cache_column)

        # Sort the signals in to an array with sorted and distinct position values
        x, y, z = np.array(x), np.array(y), np.array(z)

        sorting = np.argsort(x)
        x = x[sorting]
        z = z[sorting]

        distinct_x = sorted(set(x))
        distinct_y = sorted(set(y))
        image = np.zeros((len(distinct_x), len(distinct_y)))

        for i, column in enumerate(z):
            for j, row in enumerate(column):
                image[i, distinct_y.index(y[i*np.shape(z)[1]+j])] = row

        z = image.T

        if overlay == 'ignore':
            self.map.update({'x': distinct_x, 'y': distinct_y, 'z': z})
        else:
            pass

    def plot_map(self, save_path=None, contour=True, *args, **kwargs):
        fig, ax = plt.subplots()
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black", "red", "yellow"])
        intensity_limits = [0, np.abs(np.max(self.map['z']) * 0.9)]
        intensity_limits2 = intensity_limits
        print(intensity_limits)
        levels = np.linspace(intensity_limits2[0], intensity_limits2[1], 100)
        if not contour:
            norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
            color_map = ax.pcolormesh(self.map['x'], self.map['y'], self.map['z'], cmap=cmap, norm=norm, shading='flat')
            norm = matplotlib.colors.Normalize(vmin=intensity_limits2[0], vmax=intensity_limits2[1])
            sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map.cmap)
            sm.set_array([])
            bar = fig.colorbar(sm, ax=ax, extend='max')
        else:
            # color_map = ax.contourf(X, Y, Z, cmap=cmap, extend='neither', levels=levels, *args, **kwargs)
            if np.min(self.map['z']) < intensity_limits[0] and np.max(self.map['z']) > intensity_limits[1]:
                color_map = ax.contourf(self.map['x'], self.map['y'], self.map['z'], cmap=cmap, extend='both', levels=levels)
            elif np.min(self.map['z']) < intensity_limits[0]:
                color_map = ax.contourf(self.map['x'], self.map['y'], self.map['z'], cmap=cmap, extend='min', levels=levels)
            elif np.max(self.map['z']) > intensity_limits[1]:
                color_map = ax.contourf(self.map['x'], self.map['y'], self.map['z'], cmap=cmap, extend='max', levels=levels)
            else:
                color_map = ax.contourf(self.map['x'], self.map['y'], self.map['z'], cmap=cmap, extend='neither', levels=levels)
            # '''
            norm = matplotlib.colors.Normalize(vmin=intensity_limits2[0], vmax=intensity_limits2[1])
            sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map.cmap)
            sm.set_array([])
            bar = fig.colorbar(sm, ax=ax, extend='max', ticks=color_map.levels)
        ax.set_xlabel(r'Position x (mm)')
        ax.set_ylabel(r'Position y (mm)')
        bar.set_label('Measured Signal (a.u.)')

        save_name = self.name + '_map'
        if contour:
            save_name += '_contour'
        if save_path is not None:
            format_save(save_path=save_path, save_name=save_name)
        else:
            plt.show()

    def overview(self):
        pass

    def plot_parameter(self, parameter):
        pass


A = Analyzer((1, 64), 0.5, 0.08)
folder_path = Path('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/matrix_19062024/')
# A.set_dark_measurement(folder_path, 'd2_1n_3s_beam_all_without_diffuser_dark.csv')
A.set_measurement(folder_path, 'd2_1n_3s_beam_all_without_diffuser_dark.csv')
A.load_measurement()
fig, ax = plt.subplots()
signal = A.measurement_data[0]['signal'][0]
signal[60] /= 2000
ax.plot(signal)
print(A.measurement_data)
plt.show()
'''
A.normalization(folder_path, '5s_flat_calib_', normalization_module=normalization_from_translated_array)
for crit in ['10s_iphcmatrixcrhea_', '5s_misc_shapes_']:
    A.set_measurement(folder_path, crit)
    A.load_measurement(ams_constant_signal_readout, standard_position)
    A.create_map()
    A.plot_map('/Users/nico_brosda/Desktop/NewMaps/')
'''