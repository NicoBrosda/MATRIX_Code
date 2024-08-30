import os
import pathlib

import numpy as np
from pathlib import Path
from helper_modules import array_txt_file_search


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

    def __init__(self, diodes_dimension, diode_size, diode_spacing, readout='AMS_evaluation'):
        """
        This is an analyzer class for the readout and data analysis from measurements with diode arrays in the MATRIX
        project. This class aims to be as general as possible to adapt easily to changing diode geometries,
        readout circuits, data treatment methods and data usage. The idea is to pack the critical specific elements into
        several modules and allowing a quick adaptation by choosing the correct module combination for the present
        experimental situation. Thus, other software parts like the image generation can be kept the same.
        This class will cover a rectangular diode pattern with equally sized diodes, which should be sufficient for the
        MATRIX project - three parameters of the diodes are initialized, because they define how e.g. the readout and
        image construction work
        :param diodes_dimension: The dimension of the diode array as (n, m) array-like input. n defines the columns (x)
        of the array, m (y) the rows. Note that the choice of n, m defines the axis for following parts. If only one
        float-like input value is given, the array is set as (1, value).
        :param diode_size: Size of one diode as (len_x, len_y) array-like input. If only one float-like input value
        is given, the dimensions are set as quadratic.
        :param diode_spacing: Spacing between diodes as (spacing_x, spacing_y) array-like input. If only one float-like
        input value is given, both spacings will be set equally.
        :param readout: Option to specify the readout circuit used in measurements. Standard is the first read_out used,
        the evaluation kit of the AMS evaluation kit.
        """
        self.diodes_dimension = diodes_dimension
        self.diode_size = diode_size
        self.diode_spacing = diode_spacing

        self.readout = readout
        self.measurement_files = []
        self.excluded = np.full(self.diodes_dimension, False)
    
    def set_measurement(self, path_to_folder, filter_criterion, file_format='.csv', blacklist=None):
        if blacklist is None:
            blacklist = []
        if not isinstance(path_to_folder, pathlib.PurePath):
            path_to_folder = Path(path_to_folder)
        files = os.listdir(path_to_folder)
        measurement_files = array_txt_file_search(files, searchlist=[filter_criterion], blacklist=blacklist,
                                                  file_suffix=file_format, txt_file=False)
        print(len(measurement_files), ' files found in the folder: ', path_to_folder, ' under the search criterion: ',
              filter_criterion)
        self.measurement_files = measurement_files

    def overview(self):
        pass

    def plot_parameter(self, parameter):
        pass



A = Analyzer(1, True, 4)
A.diode_size = 10
print(A.diode_size)
