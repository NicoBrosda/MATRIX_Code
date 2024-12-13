from EvaluationSoftware.main import *

# ----------------------------------------------------------------------------------------------------------------------
# Clean example for obtaining measurement data as diode array shaped np.NdArray
# Here example for Matrix array
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# Quick choice of the correct read in functions / mapping
# ----------------------------------------------------------------------------------------------------------------------
# - Matrix array, Line array or shifted 2Line array?
array_type = 'matrix'  # = 'line' / = '2line'

# - Size of the diodes? size_x, size_y | spacing_x, spacing_y -> size_x + spacing_x = period_x
diode_size, diode_spacing = (0.4, 0.4), (0.1, 0.1)

# - Dimensions of the diode array?
if array_type == 'matrix':
    array_dimensions = (11, 11)
elif array_type == '2line':
    array_dimensions = (2, 64)
else:
    array_dimensions = (1, 128)

# - Offset?
if array_type == '2line':
    offset = [[0, - 0.25], np.zeros(array_dimensions[1])]

# ----------------------------------------------------------------------------------------------------------------------
# 1. Define the mappings required

# This section is required to translate in between both PCB - readout channel mappings (we always had "direction" 2, but
# the matrix mappings are written for mapping "direction1"
mapping = Path('../Files/mapping.xlsx')
direction1 = pd.read_excel(mapping, header=1)
direction1 = np.array([int(k[-3:]) for k in direction1['direction_1']])
direction2 = pd.read_excel(mapping, header=1)
direction2 = np.array([int(k[-3:]) for k in direction2['direction_2']])

# Import the mapping of the Matrix array - for linear arrays we only need the mapping direction2
if array_type == 'matrix':
    mapping = Path('../Files/Mapping_BigMatrix_2.xlsx')
    data2 = pd.read_excel(mapping, header=None)
    mapping_map = data2.to_numpy().flatten()
    translated_mapping = np.array([direction2[np.argwhere(direction1 == i)[0][0]] - 1 for i in mapping_map])
else:
    translated_mapping = [k - 1 for k in direction2]

# Insert the mapping into the readout function, here the correct readout needs to be chosen (for now there are 3 modules
# necessary - one for linear arrays, one for the shifted 2Line arrays and one for 2D arrays.)
if array_type == 'matrix':
    readout = lambda x, y: ams_2D_assignment_readout(x, y, channel_assignment=translated_mapping)
elif array_type == '2line':
    readout = lambda x, y: ams_2line_readout(x, y, channel_assignment=translated_mapping)
else:
    readout = lambda x, y: ams_channel_assignment_readout(x, y, channel_assignment=translated_mapping)

position_parser = standard_position
voltage_parser = standard_voltage
current_parser = standard_current

# Define the diode geometry - the first parameter give the shape of the diode array (e.g. 11 x 11 diodes for a matrix,
# 1 x 128 diodes for a line array or 2 x 64 diodes with diode_offset=[[0, - 0.25], np.zeros(64)] for the 2Line array.).
A = Analyzer(array_dimensions, diode_size, diode_spacing, readout=readout, position_parser=position_parser,
             voltage_parser=voltage_parser, current_parser=current_parser)


def wrap_func1(path_to_csv, analyzer_instance=A, path_to_dark_file=None):
    path_to_csv = Path(path_to_csv)
    if path_to_dark_file is not None:
        path_to_dark_file = Path(path_to_dark_file)
        analyzer_instance.set_dark_measurement(path_to_dark_file.parent, [path_to_dark_file.name])
    analyzer_instance.set_measurement(path_to_csv.parent, path_to_csv.name)
    analyzer_instance.load_measurement(progress_bar=False)
    return analyzer_instance.measurement_data[0]['signal']


def wrap_func2(path_to_csv, analyzer_instance=A, path_to_dark_file=None):
    path_to_csv = Path(path_to_csv)
    if path_to_dark_file is not None:
        path_to_dark_file = Path(path_to_dark_file)
        analyzer_instance.set_dark_measurement(path_to_dark_file.parent, [path_to_dark_file.name])
    analyzer_instance.set_measurement(path_to_csv.parent, path_to_csv.name)
    analyzer_instance.load_measurement(progress_bar=False)
    analyzer_instance.create_map(inverse=[True, False])
    return analyzer_instance.maps[0]
