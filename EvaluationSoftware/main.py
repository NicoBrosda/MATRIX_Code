import numpy as np
from tqdm import tqdm
from EvaluationSoftware.helper_modules import array_txt_file_search
from EvaluationSoftware.readout_modules import *
from EvaluationSoftware.position_parsing_modules import *
from Plot_Methods.plot_standards import *
from EvaluationSoftware.normalization_modules import *
from EvaluationSoftware.filter_modules import *
from EvaluationSoftware.parameter_parsing_modules import *


class DiodeGeometry:
    def __set_name__(self, owner, name):
        self.name = name
        self.private_name = f'_{name}'

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__[self.private_name]

    def __set__(self, instance, value):
        if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 2:
           val = (value[0], value[1])
        elif isinstance(value, (list, tuple, np.ndarray)) and len(value) == 1:
            val = (value[0], value[0])
        elif isinstance(value, (float, int, complex)) and not isinstance(value, bool):
            val = (value, value)
        else:
            print('No suited input value given for the '+str(self.private_name)+'. Before further proceeding is possible set '
                                                                        'a regular value!')
            val = None
        instance.__dict__[self.private_name] = val


scale_dict = {'tera': [1e12, 'T'],
              'giga': [1e9, 'G'],
              'mega': [1e6, 'M'],
              'kilo': [1e3, 'k'],
              'non': [1, ''],
              'milli': [1e-3, 'm'],
              'micro': [1e-6, r'$\micro$'],
              'nano': [1e-9, 'n'],
              'pico': [1e-12, 'p'],
              'femto': [1e-15, 'f'],
              'atto': [1e-18, 'a']}


class Analyzer:

    diode_dimension = DiodeGeometry()
    diode_size = DiodeGeometry()
    diode_spacing = DiodeGeometry()

    def __init__(self, diode_dimension, diode_size, diode_spacing, readout=ams_constant_signal_readout,
                 position_parser=standard_position, voltage_parser=None, current_parser=None, diode_offset=None):
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
        if diode_offset is None:
            self.diode_offset = [np.zeros(self.diode_dimension[0]), np.zeros(self.diode_dimension[1])]
        elif np.shape(diode_offset[0])[0] != self.diode_dimension[0] or \
                np.shape(diode_offset[1])[0] != self.diode_dimension[1]:
            print('Diode offset was inserted in the wrong dimensions - recheck your input!')
            self.diode_offset = [np.zeros(self.diode_dimension[0]), np.zeros(self.diode_dimension[1])]
        else:
            self.diode_offset = diode_offset

        self.readout = readout
        self.pos_parser = position_parser
        self.voltage_parser = voltage_parser
        self.current_parser = current_parser
        self.measurement_files = []
        self.measurement_data = []
        self.dark_files = []
        self.dark_measurements = []
        self.dark = np.zeros(self.diode_dimension)
        self.norm_factor = np.ones(diode_dimension)
        self.name = ''
        self.maps = [{'x': np.array([]), 'y': np.array([]), 'z': np.array([]), 'position': ''}]
        self.excluded = np.full(self.diode_dimension, False)

        self.clock = 3e6
        self.gain = 1e-6
        self.scale = 'pico'

    def signal_conversion(self, signal):
        scale = scale_dict[self.scale][0]
        return self.gain * self.clock / 3e6 * signal / 2 ** 26 / scale

    def set_measurement(self, path_to_folder, filter_criterion, file_format='.csv', blacklist=['.png', '.pdf', '.jpg']):
        self.measurement_files = []
        self.name = str(filter_criterion)
        if blacklist is None:
            blacklist = []
        if not isinstance(path_to_folder, pathlib.PurePath):
            path_to_folder = Path(path_to_folder)
        files = os.listdir(path_to_folder)
        if not isinstance(filter_criterion, (tuple, list)):
            filter_criterion = [filter_criterion]

        for i, crit in enumerate(filter_criterion):
            if str(path_to_folder) in crit:
                filter_criterion[i] = crit[len(str(path_to_folder))+1:]

        measurement_files = array_txt_file_search(files, searchlist=filter_criterion, blacklist=blacklist,
                                                  file_suffix=file_format, txt_file=False)
        print(len(measurement_files), ' files found in the folder: ', path_to_folder, ' under the search criterion: ',
              filter_criterion)
        self.measurement_files = [Path(path_to_folder) / i for i in measurement_files]

    def choose_dark_param(self, parameter=None):
        cache = []
        for measurement in self.dark_measurements:
            if parameter is None:
                cache.append(measurement['signal'])
            else:
                try:
                    if measurement[parameter[0]] == parameter[1]:
                        cache.append(measurement['signal'])
                except (KeyError, ValueError):
                    pass
        if len(cache) > 0:
            self.dark = np.mean(np.array(cache), axis=0)

    def set_dark_measurement(self, path_to_folder, filter_criterion: list = ['dark'], file_format: str = '.csv',
                             blacklist: list = ['.png', '.pdf', '.jpg'], readout_module: any = None, parameter=None):
        if readout_module is None:
            readout_module = self.readout
        if blacklist is None:
            blacklist = []
        if not isinstance(path_to_folder, pathlib.PurePath):
            path_to_folder = Path(path_to_folder)
        files = os.listdir(path_to_folder)
        if not isinstance(filter_criterion, (tuple, list)):
            filter_criterion = [filter_criterion]
        dark_files = array_txt_file_search(files, searchlist=filter_criterion, blacklist=blacklist,
                                                  file_suffix=file_format, txt_file=False)
        print(len(dark_files), ' files for background correction found in the folder: ', path_to_folder,
              ' under the search criterion: ', filter_criterion)
        self.dark_files = [Path(path_to_folder) / i for i in dark_files]

        for file in self.dark_files:
            self.dark_measurements.append(readout_module(path_to_folder / file, self))
            if self.voltage_parser is not None:
                self.dark_measurements[-1].update({'voltage': self.voltage_parser(file)})
            if self.current_parser is not None:
                self.dark_measurements[-1].update({'current': self.current_parser(file)})

        self.choose_dark_param(parameter)

    def normalization(self, path_to_folder, filter_criterion, file_format='.csv', blacklist=['.png', '.pdf', '.jpg'],
                      normalization_module=None, cache_save=True, factor_limits=(0, 3), norm_factor=True):
        # Check if factor is already saved and is not needed to be recalculated:
        if not isinstance(filter_criterion, (tuple, list)):
            filter_criterion = [filter_criterion]
        if cache_save and os.path.isfile(path_to_folder / (str(filter_criterion)+'_normalization_factor.npy')):
            try:
                factor = np.load(path_to_folder / (str(filter_criterion)+'_normalization_factor.npy'))
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
        files = array_txt_file_search(files, searchlist=filter_criterion, blacklist=blacklist,
                                                  file_suffix=file_format, txt_file=False)
        print(len(files), ' files for normalization found in the folder: ', path_to_folder,
              ' under the search criterion: ', filter_criterion)
        files = [Path(path_to_folder) / i for i in files]

        factor = normalization_module(files, self)

        if cache_save:
            np.save(path_to_folder / (str(filter_criterion)+'_normalization_factor.npy'), factor)

        factor[((factor < factor_limits[0]) | (factor > factor_limits[1]))] = 0

        if norm_factor:
            factor = factor / np.mean(factor[factor != 0])

        self.norm_factor = factor

    def load_measurement(self, readout_module=None, position_parser=None, progress_bar=True):
        self.measurement_data = []
        if readout_module is None:
            readout_module = self.readout
        else:
            self.readout = readout_module

        if position_parser is None:
            position_parser = self.pos_parser
        else:
            self.pos_parser = position_parser
        if progress_bar:
            for file in tqdm(self.measurement_files):
                pos = position_parser(file)
                cache = readout_module(file, self)
                cache['signal'] = (cache['signal']-self.dark)*self.norm_factor
                cache.update({'position': pos})
                if self.voltage_parser is not None:
                    cache.update({'voltage': self.voltage_parser(file)})
                if self.current_parser is not None:
                    cache.update({'current': self.current_parser(file)})
                self.measurement_data.append(cache)
        else:
            for file in self.measurement_files:
                pos = position_parser(file)
                cache = readout_module(file, self)
                cache['signal'] = (cache['signal']-self.dark)*self.norm_factor
                cache.update({'position': pos})
                if self.voltage_parser is not None:
                    cache.update({'voltage': self.voltage_parser(file)})
                if self.current_parser is not None:
                    cache.update({'current': self.current_parser(file)})
                self.measurement_data.append(cache)

    def load_measurement_time(self, readout_module=None, position_parser=None, progress_bar=False):
        self.measurement_data = []
        if readout_module is None:
            readout_module = self.readout
        else:
            self.readout = readout_module

        if position_parser is None:
            position_parser = self.pos_parser
        else:
            self.pos_parser = position_parser
        if progress_bar:
            for file in tqdm(self.measurement_files):
                pos = position_parser(file)
                cache = readout_module(file, self)
                cache['signal'] = (cache['signal']-self.dark)*self.norm_factor
                cache.update({'position': pos})
                if self.voltage_parser is not None:
                    cache.update({'voltage': self.voltage_parser(file)})
                if self.current_parser is not None:
                    cache.update({'current': self.current_parser(file)})
                self.measurement_data.append(cache)
        else:
            for file in self.measurement_files:
                pos = position_parser(file)
                cache = readout_module(file, self)
                cache['signal'] = (cache['signal']-self.dark)*self.norm_factor
                cache.update({'position': pos})
                if self.voltage_parser is not None:
                    cache.update({'voltage': self.voltage_parser(file)})
                if self.current_parser is not None:
                    cache.update({'current': self.current_parser(file)})
                self.measurement_data.append(cache)

    def update_measurement(self, dark=True, factor=True):
        if dark and factor:
            for i in tqdm(range(len(self.measurement_data))):
                self.measurement_data[i]['signal'] = (self.measurement_data[i]['signal']-self.dark)*self.norm_factor
        elif dark:
            for i in tqdm(range(len(self.measurement_data))):
                self.measurement_data[i]['signal'] = self.measurement_data[i]['signal']-self.dark
        elif factor:
            for i in tqdm(range(len(self.measurement_data))):
                self.measurement_data[i]['signal'] = self.measurement_data[i]['signal']*self.norm_factor

    def create_map(self, overlay='interpolate', inverse=[False, False]):
        faster_loop: bool = (self.diode_offset[1].all() == 0)
        faster_loop: bool = False

        self.maps = []

        def mapping(position):
            if isinstance(position, str):
                filter_data = self.measurement_data
            else:
                if self.diode_dimension[1] == 1:
                    switch = 0
                elif self.diode_dimension[0] == 1:
                    switch = 1
                else:
                    print('Under the given diode geometry the map creation for several line array positions is not '
                          'possible.')
                    return None
                filter_data = [i for i in self.measurement_data if position == i['position'][switch]]
            x = []
            y = []
            z = []
            for data in filter_data:
                pos = data['position']
                # print(pos, '#'*50)
                if None in pos or np.isnan(pos[0]) or np.isnan(pos[1]):
                    continue
                signal = data['signal']
                if faster_loop:
                    for i, column in enumerate(signal):
                        # print('column', i, '-' * 50)
                        # print(pos[0] + i * (self.diode_size[0] + self.diode_spacing[0]))
                        x.append(pos[0] + (i - (self.diode_dimension[0]-1)/2) * (self.diode_size[0] + self.diode_spacing[0]))
                        cache_column = []
                        for j, row in enumerate(column):
                            # if j == 0:
                                # print('row', i, '.' * 50)
                                # print(pos[1] + j * (self.diode_size[1] + self.diode_spacing[1]) + self.diode_offset[0][i])
                            y.append(pos[1] + (j - self.diode_dimension[1]/2) * (self.diode_size[1] + self.diode_spacing[1]) + self.diode_offset[0][i])
                            cache_column.append(row)
                        z.append(cache_column)
                else:
                    for i, column in enumerate(signal):
                        for j, row in enumerate(column):
                            x.append(pos[0] + (i - (self.diode_dimension[0]-1)/2) * (self.diode_size[0] + self.diode_spacing[0]) + self.diode_offset[1][j])
                            y.append(pos[1] + (j - (self.diode_dimension[1]-1)/2) * (self.diode_size[1] + self.diode_spacing[1]) + self.diode_offset[0][i])
                            z.append(row)

            # Sort the signals in to an array with sorted and distinct position values
            x, y, z = np.array(x), np.array(y), np.array(z)

            sorting = np.argsort(x)
            if not faster_loop:
                y = y[sorting]
            x = x[sorting]
            z = z[sorting]
            distinct_x = sorted(set(x))
            distinct_y = sorted(set(y))
            image = np.full((len(distinct_x), len(distinct_y)), -1)

            if faster_loop:
                for i, column in enumerate(z):
                    for j, row in enumerate(column):
                        image[distinct_x.index(x[i]), distinct_y.index(y[i * np.shape(z)[1] + j])] = row
            else:
                for i, row in enumerate(z):
                    x_index = distinct_x.index(x[i])
                    y_index = distinct_y.index(y[i])
                    if image[x_index, y_index] == -1:
                        image[x_index, y_index] = row
                    elif overlay == 'ignore':
                        image[x_index, y_index] = row
                    elif overlay == 'interpolate':
                        if image[x_index, y_index] == 0:
                            image[x_index, y_index] = row
                        elif row == 0:
                            pass
                        else:
                            image[x_index, y_index] = (row + image[x_index, y_index]) / 2

            z = image.T
            if inverse[0]:
                z = z[::-1]
            if inverse[1]:
                z = z[:, ::-1]

            self.maps.append({'x': np.array(distinct_x), 'y': np.array(distinct_y), 'z': self.signal_conversion(z),
                              'position': str(position)})

        # Create loop to automatize plotting of multiple maps from 1 array (x or y shifted)
        if self.diode_dimension[1] == 1 and len(set([i['position'][0] for i in self.measurement_data])) > 1:
            for posi in set([i['position'][0] for i in self.measurement_data]):
                mapping(posi)
        elif self.diode_dimension[0] == 1 and len(set([i['position'][1] for i in self.measurement_data])) > 1:
            for posi in set([i['position'][1] for i in self.measurement_data]):
                mapping(posi)
        else:
            mapping('')

    def plot_map(self, save_path=None, pixel=True, intensity_limits=None, ax_in=None, fig_in=None, colorbar=True,
                 cmap=matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black", "red", "yellow"]),
                 plot_size=fullsize_plot, imshow=False, dpi=300, save_format='.svg', bbox=None, alpha=1.0,
                 insert_txt: bool or list = False, *args, **kwargs):
        if isinstance(pixel, str):
            pixel = pixel.lower()
        if len(self.maps) == 1:
            print(len(self.maps), ' map will be plotted.')
        else:
            print(len(self.maps), ' maps will be plotted at the positions ', [i['position'] for i in self.maps])
        if intensity_limits is None:
            # intensity_limits = [0, np.abs(np.max(map_el['z']) * 0.9)]
            intensity_limits = [0, np.abs(np.max([np.max(i['z']) for i in self.maps]))]
        for map_el in self.maps:

            if ax_in is None or fig_in is None:
                fig, ax = plt.subplots()
            else:
                fig, ax = fig_in, ax_in
            fig.set_dpi(dpi)

            # levels = np.linspace(intensity_limits[0], intensity_limits[1], 100)
            levels = np.linspace(intensity_limits[0], intensity_limits[1], 256)

            if pixel:
                # Auto-detect step width in x and y:
                x_steps = np.array([map_el['x'][i + 1] - map_el['x'][i] for i in range(np.shape(map_el['x'])[0] - 1)])
                y_steps = np.array([map_el['y'][i + 1] - map_el['y'][i] for i in range(np.shape(map_el['y'])[0] - 1)])

                # Auto-detect if spaces should be inserted in one-direction (> 2 diodes and distances = diodes geometry)
                # if self.diode_dimension[0] <= 2 and x_steps.std() == 0 and x_steps.mean() == self.diode_size[0]+self.diode_spacing[0]:
                if x_steps.std() == 0 and x_steps.mean() == self.diode_size[0] + self.diode_spacing[0]:
                    cache_x = np.array([map_el['x'][0]-(self.diode_spacing[0]+self.diode_size[0])/2])
                    for i in range(np.shape(map_el['x'])[0]):
                        if i == 0:
                            cache_x = np.append(cache_x, cache_x[-1]+self.diode_spacing[0]/2)
                            cache_x = np.append(cache_x, cache_x[-1] + self.diode_size[0])
                        else:
                            cache_x = np.append(cache_x, cache_x[-1] + self.diode_spacing[0])
                            cache_x = np.append(cache_x, cache_x[-1]+self.diode_size[0])
                    cache_z = []
                    for i, row in enumerate(map_el['z'].T):
                        if pixel == 'fill':
                            if i == 0:
                                cache_z.append(map_el['z'].T[i])
                            else:
                                cache_z.append((map_el['z'].T[i]+map_el['z'].T[i-1])/2)
                        else:
                            cache_z.append(np.zeros_like(row))
                        cache_z.append(row)
                    cache_z = np.array(cache_z).T
                # Else insert +1 step in the end of the measurement and do not add white spaces
                else:
                    cache_x = [map_el['x'][i] - x_steps[i-1]/2 for i, el in enumerate(map_el['x']) if i > 0]
                    cache_x = [map_el['x'][0] - x_steps[0]/2] + cache_x + [map_el['x'][-1] + x_steps[-1]/2]
                    # cache_x = np.append(map_el['x'], map_el['x'][-1] + x_steps.mean()) - x_steps.mean()/2
                    cache_z = map_el['z']

                # if self.diode_dimension[1] <= 2 and y_steps.std() == 0 and y_steps.mean() == self.diode_size[1]+self.diode_spacing[1]:
                if y_steps.std() == 0 and y_steps.mean() == self.diode_size[1] + self.diode_spacing[1]:
                    cache_y = np.array([map_el['y'][0]-(self.diode_spacing[1]+self.diode_size[1])/2])
                    for i in range(np.shape(map_el['y'])[0]):
                        if i == 0:
                            cache_y = np.append(cache_y, cache_y[-1] + self.diode_spacing[1] / 2)
                            cache_y = np.append(cache_y, cache_y[-1] + self.diode_size[1])
                        else:
                            cache_y = np.append(cache_y, cache_y[-1] + self.diode_spacing[1])
                            cache_y = np.append(cache_y, cache_y[-1] + self.diode_size[1])

                    cache = []
                    for i, row in enumerate(cache_z):
                        if pixel == 'fill':
                            if i == 0:
                                cache.append(cache_z[i])
                            else:
                                cache.append((cache_z[i]+cache_z[i-1])/2)
                        else:
                            cache.append(np.zeros_like(row))
                        cache.append(row)
                    cache_z = np.array(cache)
                else:
                    cache_y = [map_el['y'][i] - y_steps[i - 1] / 2 for i, el in enumerate(map_el['y']) if i > 0]
                    cache_y = [map_el['y'][0] - y_steps[0] / 2] + cache_y + [map_el['y'][-1] + y_steps[-1] / 2]
                    # cache_y = np.append(map_el['y'], map_el['y'][-1] + y_steps[-1]) - y_steps.mean()/2
                    cache_z = cache_z

                # homogenize_pixel_size({'x': cache_x, 'y': cache_y, 'z': cache_z})
                # print(np.shape(cache_x), np.shape(cache_y), np.shape(cache_z))
                norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
                if not imshow:
                    color_map = ax.pcolormesh(cache_x, cache_y, cache_z, cmap=cmap, norm=norm, shading='flat',
                                              alpha=alpha)
                else:
                    map_x, map_y, map_z = homogenize_pixel_size([cache_x, cache_y, cache_z])

                    pixel_size = map_x[1] - map_x[0]
                    p2 = pixel_size / 2
                    if isinstance(imshow, str):
                        interpolation = imshow
                    else:
                        interpolation = 'antialiased'

                    color_map = ax.imshow(map_z, cmap=cmap, origin='lower', vmin=intensity_limits[0],
                                          vmax=intensity_limits[1], interpolation=interpolation, alpha=alpha,
                                          extent=(map_x[0] - p2, map_x[-1] + p2, map_y[0] - p2, map_y[-1] + p2))
                # '''
                norm = matplotlib.colors.Normalize(vmin=intensity_limits[0], vmax=intensity_limits[1])
                sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map.cmap)
                sm.set_array([])
                if colorbar:
                    bar = fig.colorbar(sm, ax=ax, extend='max')
            else:
                if np.min(map_el['z']) < intensity_limits[0] and np.max(map_el['z']) > intensity_limits[1]:
                    color_map = ax.contourf(map_el['x'], map_el['y'], map_el['z'], cmap=cmap, extend='both', levels=levels, alpha=alpha)
                elif np.min(map_el['z']) < intensity_limits[0]:
                    color_map = ax.contourf(map_el['x'], map_el['y'], map_el['z'], cmap=cmap, extend='min', levels=levels, alpha=alpha)
                elif np.max(map_el['z']) > intensity_limits[1]:
                    color_map = ax.contourf(map_el['x'], map_el['y'], map_el['z'], cmap=cmap, extend='max', levels=levels, alpha=alpha)
                else:
                    color_map = ax.contourf(map_el['x'], map_el['y'], map_el['z'], cmap=cmap, extend='neither', levels=levels, alpha=alpha)
                norm = matplotlib.colors.Normalize(vmin=intensity_limits[0], vmax=intensity_limits[1])
                sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map.cmap)
                sm.set_array([])
                if colorbar:
                    bar = fig.colorbar(sm, ax=ax, extend='max', ticks=color_map.levels)
            ax.set_xlabel(r'Position x (mm)')
            ax.set_ylabel(r'Position y (mm)')

            # ax.set_xlabel(r'Stage position y (mm)')
            # ax.set_ylabel(r'Stage position x (mm)')

            # Scale the axis true to scale
            x_scale = ax.get_xlim()
            y_scale = ax.get_ylim()
            if x_scale[1] - x_scale[0] < y_scale[1] - y_scale[0]:
                ax.set_xlim(x_scale[0] - (y_scale[1] - y_scale[0] - x_scale[1] + x_scale[0])/2,
                            x_scale[1] + (y_scale[1] - y_scale[0] - x_scale[1] + x_scale[0])/2)
            elif x_scale[1] - x_scale[0] > y_scale[1] - y_scale[0]:
                ax.set_ylim(y_scale[0] - (x_scale[1] - x_scale[0] - y_scale[1] + y_scale[0])/2,
                            y_scale[1] + (x_scale[1] - x_scale[0] - y_scale[1] + y_scale[0])/2)

            # x_scale = ax.get_xlim()
            # ax.set_xlim(x_scale[0]-(x_scale[1]-x_scale[0])*0.15, x_scale[1]+(x_scale[1]-x_scale[0])*0.15)

            if colorbar:
                # bar.set_label('Measured Signal (a.u.)')
                bar.set_label(f'Signal Current ({scale_dict[self.scale][1]}A)')

            if insert_txt:
                if len(insert_txt) < 4:
                    ax.text(*transform_axis_to_data_coordinates(ax, insert_txt[0]), insert_txt[1], c='k',
                            fontsize=insert_txt[2], zorder=3, bbox={'facecolor': 'white', 'alpha': 0.3, 'pad': 2})
                elif len(insert_txt) < 5:
                    ax.text(*transform_axis_to_data_coordinates(ax, insert_txt[0]), insert_txt[1], c=insert_txt[3],
                            fontsize=insert_txt[2], zorder=3, bbox={'facecolor': 'white', 'alpha': 0.3, 'pad': 2})
                else:
                    ax.text(*transform_axis_to_data_coordinates(ax, insert_txt[0]), insert_txt[1], c=insert_txt[3],
                            fontsize=insert_txt[2], zorder=3, bbox=insert_txt[4])

            save_name = self.name + '_map' + map_el['position']
            if imshow:
                save_name += '_imshow'
            if not pixel:
                save_name += '_contour'
            if pixel == 'fill':
                save_name += '_fill'
            if save_path is not None:
                format_save(save_path=save_path, save_name=save_name, dpi=dpi, plot_size=plot_size,
                            save_format=save_format, fig=fig, bbox=bbox)

    def overview(self):
        pass

    def plot_for_parameter(self, parameter='voltage', dark=True, map_inverse=[True, False], *args, **kwargs):
        if dark:
            self.dark = np.zeros(self.diode_dimension)
            factor_cache = deepcopy(self.norm_factor)
            self.norm_factor = np.ones(self.diode_dimension)
        self.load_measurement()

        param_list = []
        for measurement in self.measurement_data:
            try:
                param_list.append(measurement[str(parameter)])
            except KeyError:
                print('The given parameter is not given for the chosen measurements!')
                return None
        param_list = set(param_list)
        print(f'For the parameter {parameter} there are {len(param_list)} different values in the chosen measurements. '
              f'A map for each of the values in {param_list} will be plotted!')
        cache = deepcopy(self.measurement_data)
        name = deepcopy(self.name)
        for param in param_list:
            self.measurement_data = [i for i in cache if i[parameter]==param]
            if dark:
                self.choose_dark_param([parameter, param])
                self.norm_factor = factor_cache
                self.update_measurement(dark=True, factor=True)
            self.create_map(inverse=map_inverse)
            self.name = name + f'_{parameter}={param}_'
            self.plot_map(*args, **kwargs)

    def get_signal_xline(self, y_position=None, x_start=None, x_end=None, map_select=None):
        if map_select is None:
            map_select = self.maps[0]
        elif isinstance(map_select, (float, int)):
            map_select = self.maps[map_select]

        # Find the indices to get the signal data - in y, closest to given y_position or middle of y range
        if y_position is None:
            y_position = np.argmin(np.abs(map_select['y'] - map_select['y'].mean()))
        else:
            y_position = np.argmin(np.abs(map_select['y'] - y_position))

        # Index 0 if no value given for x_start, otherwise closest value in map['x'] to x_start
        if x_start is None:
            x_start = 0
        else:
            x_start = np.argmin(np.abs(map_select['x'] - x_start))

        # Index of last value in x if no value given for x_start, otherwise closest value in map['x'] to x_start
        if x_end is None:
            x_end = np.argmax(map_select['x']) + 1
        else:
            x_end = np.argmin(np.abs(map_select['x'] - x_end))

        # Get the signal with the indices
        signal = map_select['z'][y_position, x_start:x_end]
        return signal

    def get_signal_yline(self, x_position=None, y_start=None, y_end=None, map_select=None):
        if map_select is None:
            map_select = self.maps[0]
        elif isinstance(map_select, (float, int)):
            map_select = self.maps[map_select]

        # Find the indices to get the signal data - in x, closest to given x_position or middle of x range
        if x_position is None:
            x_position = np.argmin(np.abs(map_select['x'] - map_select['x'].mean()))
        else:
            x_position = np.argmin(np.abs(map_select['x'] - x_position))

        # Index 0 if no value given for y_start, otherwise closest value in map['y'] to y_start
        if y_start is None:
            y_start = 0
        else:
            y_start = np.argmin(np.abs(map_select['y'] - y_start))

        # Index of last value in y if no value given for y_start, otherwise closest value in map['y'] to y_start
        if y_end is None:
            y_end = np.argmax(map_select['y']) + 1
        else:
            y_end = np.argmin(np.abs(map_select['y'] - y_end))

        # Get the signal with the indices
        signal = map_select['z'][y_start:y_end, x_position]
        return signal

    def get_diodes_signal(self, direction=None, diode_line=None, pos_select=None, inverse=[False, False],):
        if direction == 'x' or direction == 0:
            switch = 1
            direction = 0
            set_positions = set([i['position'][1] for i in self.measurement_data])
        else:
            switch = 0
            direction = 1
            set_positions = set([i['position'][0] for i in self.measurement_data])

        if diode_line is None or not isinstance(diode_line, int) or 0 > diode_line or diode_line > self.diode_dimension[switch] - 1:
            diode_line = int(self.diode_dimension[switch]/2)

        # Loop over set_positions and create a plot at each unique measurement position not in direction
        if pos_select is None:
            pos_select = 0
        print(set_positions)
        set_position = list(set_positions)[pos_select]

        filter_data = [i for i in self.measurement_data if set_position == i['position'][switch]]
        pos_var = [[] for i in range(self.diode_dimension[direction])]
        signal_var = [[] for i in range(self.diode_dimension[direction])]
        for data in filter_data:
            pos = data['position']
            if None in pos or np.isnan(pos[0]) or np.isnan(pos[1]):
                continue
            signal = data['signal']
            if direction == 0:
                signal = signal[:, diode_line]
            else:
                signal = signal[diode_line]

            for i, column in enumerate(signal):
                pos_var[i].append(pos[direction] + i * (self.diode_size[direction] + self.diode_spacing[direction]) +
                                  self.diode_offset[direction][i])
                signal_var[i].append(column)

        # Sort the signals in to an array with sorted and distinct position values
        for diode in range(self.diode_dimension[direction]):
            ordering = np.argsort(pos_var[diode])
            pos_var[diode], signal_var[diode] = np.array(pos_var[diode])[ordering], np.array(signal_var[diode])[ordering]
        if inverse[direction]:
            signal_var = np.array(signal_var)[::-1, ::-1]
        return np.array(pos_var), self.signal_conversion(np.array(signal_var))

    def plot_diodes(self, save_path=None, direction=None, plotting_range=None, diode_line=None, inverse=[False, False],
                    diode_cmap=sns.color_palette("coolwarm", as_cmap=True)):
        """
        Takes a direction and plots diodes signal in this direction in a range given. Meaning the arguments x_position
        and y_position change their meaning depending on the direction given. The parameter aligning with the specified
        direction gives the plotted range in calculated real diode positions, the other parameter is used to specify the
        diode line which is considered. Note that for a line array this parameter has no influence.
        :param direction:
        :param plotting_range:
        :param diode_line:
        :param inverse:
        :param diode_cmap:
        :return:
        """
        if direction == 'x' or direction == 0:
            switch = 1
            direction = 0
            set_positions = set([i['position'][1] for i in self.measurement_data])
        else:
            switch = 0
            direction = 1
            set_positions = set([i['position'][0] for i in self.measurement_data])

        if diode_line is None or not isinstance(diode_line, int) or 0 > diode_line or diode_line > self.diode_dimension[switch] - 1:
            diode_line = int(self.diode_dimension[switch]/2)

        # Loop over set_positions and create a plot at each unique measurement position not in direction
        for set_position in set_positions:
            filter_data = [i for i in self.measurement_data if set_position == i['position'][switch]]
            pos_var = [[] for i in range(self.diode_dimension[direction])]
            signal_var = [[] for i in range(self.diode_dimension[direction])]
            for data in filter_data:
                pos = data['position']
                if None in pos or np.isnan(pos[0]) or np.isnan(pos[1]):
                    continue
                signal = data['signal']
                if direction == 0:
                    signal = signal[:, diode_line]
                else:
                    signal = signal[diode_line]

                for i, column in enumerate(signal):
                    pos_var[i].append(pos[direction] + i * (self.diode_size[direction]+self.diode_spacing[direction]) +
                                      self.diode_offset[direction][i])
                    signal_var[i].append(column)

            # Sort the signals in to an array with sorted and distinct position values
            for diode in range(self.diode_dimension[direction]):
                ordering = np.argsort(pos_var[diode])
                pos_var[diode], signal_var[diode] = np.array(pos_var[diode])[ordering], np.array(signal_var[diode])[ordering]
            if inverse[direction]:
                signal_var = np.array(signal_var)[::-1, ::-1]
            signal_var = self.signal_conversion(signal_var)
            diode_colormapper = lambda diode: color_mapper(diode, 0, self.diode_dimension[direction])
            diode_color = lambda diode: diode_cmap(diode_colormapper(diode))

            fig, ax = plt.subplots()
            if range is not None and isinstance(plotting_range, (tuple, list, np.ndarray)):
                ax.set_xlim(*plotting_range)
            for diode in range(self.diode_dimension[direction]):
                ax.plot(pos_var[diode], signal_var[diode], color=diode_color(diode), alpha=0.5)

            ax.set_xlabel('Real measurement position of diode (mm)')
            # ax.set_ylabel('Measured Signal (a.u.)')
            ax.set_ylabel(f'Signal Current ({scale_dict[self.scale][1]}A)')
            ax.set_xlim(ax.get_xlim())
            ax.set_ylim(ax.get_ylim())
            gradient_arrow(ax, transform_axis_to_data_coordinates(ax, [0.11, 0.92]),
                           transform_axis_to_data_coordinates(ax, [0.11, 0.79]), cmap=diode_cmap, lw=10, zorder=5)
            ax.text(*transform_axis_to_data_coordinates(ax, [0.035, 0.94]), r'Diode $\#$1', fontsize=13,
                    c=diode_color(0), zorder=3)  # , bbox={'facecolor': freq_colour(1033), 'alpha': 0.2, 'pad': 2})
            ax.text(*transform_axis_to_data_coordinates(ax, [0.02, 0.71]),
                    r'Diode $\#$' + str(self.diode_dimension[direction]),
                    fontsize=13, c=diode_color(self.diode_dimension[direction]),
                    zorder=3)  # , bbox={'facecolor': freq_colour(32033), 'alpha': 0.2, 'pad': 2})

            if direction == 0:
                save_name = 'DiodeScan_XDirection_YMeasurement' + str(set_position)
            else:
                save_name = self.name + 'DiodeScan_YDirection_XMeasurement' + str(set_position)
            if save_path is not None:
                format_save(save_path=save_path, save_name=save_name)

    def rescale_maps(self, extend=None, standard_value=0.0):
        # Determine the global extent if not provided
        if extend is None:
            extend = [
                [np.min([np.min(i['x']) for i in self.maps]), np.max([np.max(i['x']) for i in self.maps])],
                [np.min([np.min(i['y']) for i in self.maps]), np.max([np.max(i['y']) for i in self.maps])]
            ]

        # Iterate through all maps with index k
        for k, map_data in enumerate(self.maps):
            # Get original map data
            x = map_data['x']
            y = map_data['y']
            z = map_data['z']

            # Initialize the new x and y ranges
            x_min, x_max = extend[0]
            y_min, y_max = extend[1]

            # Create new x and y arrays considering the padding
            # For padding, we need to extend the boundaries with the original steps at the edges
            new_x = list(x)
            new_y = list(y)

            # Padding for x (extend boundaries with original steps at the edges)
            if x_min < x[0]:
                # Add padding to the left
                while new_x[0] > x_min:
                    new_x.insert(0, new_x[0] - (x[1] - x[0]))

            if x_max > x[-1]:
                # Add padding to the right
                while new_x[-1] < x_max:
                    new_x.append(new_x[-1] + (x[-1] - x[-2]))

            # Padding for y (extend boundaries with original steps at the edges)
            if y_min < y[0]:
                # Add padding to the top
                while new_y[0] > y_min:
                    new_y.insert(0, new_y[0] - (y[1] - y[0]))

            if y_max > y[-1]:
                # Add padding to the bottom
                while new_y[-1] < y_max:
                    new_y.append(new_y[-1] + (y[-1] - y[-2]))

            # Now, we need to adjust the z array to match the new x and y grids
            # Create a new z array that has the same dimensions as the padded x and y arrays
            new_z = np.full((len(new_y), len(new_x)), standard_value)

            # Get the indices where original x and y fit into the new grid
            x_start = np.searchsorted(new_x, x[0])
            x_end = np.searchsorted(new_x, x[-1], side='right')
            y_start = np.searchsorted(new_y, y[0])
            y_end = np.searchsorted(new_y, y[-1], side='right')

            # Copy the original z values into the correct locations in the new_z array
            new_z[y_start:y_end, x_start:x_end] = z

            # Update map with rescaled x, y, z (using the index k)
            self.maps[k]['x'] = np.array(new_x)
            self.maps[k]['y'] = np.array(new_y)
            self.maps[k]['z'] = new_z

