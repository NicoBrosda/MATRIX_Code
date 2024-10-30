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
        if diode_offset is None or np.shape(diode_offset[0])[0] != self.diode_dimension[0] or \
                np.shape(diode_offset[1])[0] != self.diode_dimension[1]:
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
        self.dark = np.zeros(self.diode_dimension)
        self.norm_factor = np.ones(diode_dimension)
        self.name = ''
        self.maps = [{'x': np.array([]), 'y': np.array([]), 'z': np.array([]), 'position': ''}]
        self.excluded = np.full(self.diode_dimension, False)

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
        measurement_files = array_txt_file_search(files, searchlist=filter_criterion, blacklist=blacklist,
                                                  file_suffix=file_format, txt_file=False)
        print(len(measurement_files), ' files found in the folder: ', path_to_folder, ' under the search criterion: ',
              filter_criterion)
        self.measurement_files = [Path(path_to_folder) / i for i in measurement_files]

    def set_dark_measurement(self, path_to_folder, filter_criterion: list = ['dark'], file_format: str = '.csv',
                             blacklist: list = ['.png', '.pdf', '.jpg'], readout_module: any = None):
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

        cache = []
        for file in self.dark_files:
            cache.append(readout_module(path_to_folder / file, self)['signal'])
        self.dark = np.mean(np.array(cache), axis=0)

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

    def create_map(self, overlay='ignore', inverse=[False, False]):
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
                if None in pos or np.isnan(pos[0]) or np.isnan(pos[1]):
                    continue
                signal = data['signal']
                for i, column in enumerate(signal):
                    x.append(pos[0] + i * (self.diode_size[0] + self.diode_spacing[0]) + self.diode_offset[0][i])
                    cache_column = []
                    for j, row in enumerate(column):
                        y.append(pos[1] + j * (self.diode_size[1] + self.diode_spacing[1]) + self.diode_offset[1][j])
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
                    image[distinct_x.index(x[i]), distinct_y.index(y[i * np.shape(z)[1] + j])] = row

            z = image.T
            if inverse[0]:
                z = z[::-1]
            if inverse[1]:
                z = z[:, ::-1]
            if overlay == 'ignore':
                self.maps.append({'x': np.array(distinct_x), 'y': np.array(distinct_y), 'z': z, 'position': str(position)})
            else:
                pass

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
                 plot_size=fullsize_plot, *args, **kwargs):
        if isinstance(pixel, str):
            pixel = pixel.lower()
        print(len(self.maps), ' map created at positions ', [i['position'] for i in self.maps])
        for map_el in self.maps:
            if ax_in is None or fig_in is None:
                fig, ax = plt.subplots()
            else:
                fig, ax = fig_in, ax_in
            if intensity_limits is None:
                intensity_limits = [0, np.abs(np.max(map_el['z']) * 0.9)]
                intensity_limits = [0, np.abs(np.max(map_el['z']))]

            # print(intensity_limits)
            levels = np.linspace(intensity_limits[0], intensity_limits[1], 100)
            if pixel:
                # Auto-detect step width in x and y:
                x_steps = np.array([map_el['x'][i + 1] - map_el['x'][i] for i in range(np.shape(map_el['x'])[0] - 1)])
                y_steps = np.array([map_el['y'][i + 1] - map_el['y'][i] for i in range(np.shape(map_el['y'])[0] - 1)])

                # Auto-detect if spaces should be inserted in one-direction (>1 diode and distances = diodes geometry)
                if self.diode_dimension[0] > 1 and x_steps.std() == 0 and \
                        x_steps.mean() == self.diode_size[0]+self.diode_spacing[0]:
                    cache_x = np.array([map_el['x'][0]])
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
                    cache_x = np.append(map_el['x'], map_el['x'][-1] + x_steps.mean())
                    cache_z = map_el['z']

                if self.diode_dimension[1] > 1 and y_steps.std() == 0 and \
                        y_steps.mean() == self.diode_size[1]+self.diode_spacing[1]:
                    cache_y = np.array([map_el['y'][0]])
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
                    cache_y = np.append(map_el['y'], map_el['y'][-1] + y_steps.mean())
                    cache_z = cache_z

                # print(np.shape(cache_x), np.shape(cache_y), np.shape(cache_z))
                norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
                color_map = ax.pcolormesh(cache_x, cache_y, cache_z, cmap=cmap, norm=norm, shading='flat')
                norm = matplotlib.colors.Normalize(vmin=intensity_limits[0], vmax=intensity_limits[1])
                sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map.cmap)
                sm.set_array([])
                if colorbar:
                    bar = fig.colorbar(sm, ax=ax, extend='max')
            else:
                # color_map = ax.contourf(X, Y, Z, cmap=cmap, extend='neither', levels=levels, *args, **kwargs)
                if np.min(map_el['z']) < intensity_limits[0] and np.max(map_el['z']) > intensity_limits[1]:
                    color_map = ax.contourf(map_el['x'], map_el['y'], map_el['z'], cmap=cmap, extend='both', levels=levels)
                elif np.min(map_el['z']) < intensity_limits[0]:
                    color_map = ax.contourf(map_el['x'], map_el['y'], map_el['z'], cmap=cmap, extend='min', levels=levels)
                elif np.max(map_el['z']) > intensity_limits[1]:
                    color_map = ax.contourf(map_el['x'], map_el['y'], map_el['z'], cmap=cmap, extend='max', levels=levels)
                else:
                    color_map = ax.contourf(map_el['x'], map_el['y'], map_el['z'], cmap=cmap, extend='neither', levels=levels)
                # '''
                norm = matplotlib.colors.Normalize(vmin=intensity_limits[0], vmax=intensity_limits[1])
                sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map.cmap)
                sm.set_array([])
                if colorbar:
                    bar = fig.colorbar(sm, ax=ax, extend='max', ticks=color_map.levels)
            ax.set_xlabel(r'Position x (mm)')
            ax.set_ylabel(r'Position y (mm)')

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
                bar.set_label('Measured Signal (a.u.)')

            save_name = self.name + '_map' + map_el['position']
            if not pixel:
                save_name += '_contour'
            if pixel == 'fill':
                save_name += '_fill'
            if save_path is not None:
                if pixel:
                    save_format = '.png'
                    dpi = 300
                else:
                    save_format = '.png'
                    dpi = 300
                format_save(save_path=save_path, save_name=save_name, dpi=dpi, format=save_format)

    def overview(self):
        pass

    def plot_parameter(self, parameter):
        pass

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
        return np.array(pos_var), np.array(signal_var)

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
            diode_colormapper = lambda diode: color_mapper(diode, 0, self.diode_dimension[direction])
            diode_color = lambda diode: diode_cmap(diode_colormapper(diode))

            fig, ax = plt.subplots()
            if range is not None and isinstance(plotting_range, (tuple, list, np.ndarray)):
                ax.set_xlim(*plotting_range)
            for diode in range(self.diode_dimension[direction]):
                ax.plot(pos_var[diode], signal_var[diode], color=diode_color(diode), alpha=0.5)

            ax.set_xlabel('Real measurement position of diode (mm)')
            ax.set_ylabel('Measured Signal (a.u.)')
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
