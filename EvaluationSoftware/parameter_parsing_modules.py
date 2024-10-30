from EvaluationSoftware.helper_modules import *


def standard_voltage(path_to_data_file):
    if not isinstance(path_to_data_file, str):
        path_to_data_file = str(path_to_data_file)
    # The parsing of the position out of the name and save it
    try:
        index2 = path_to_data_file.rindex('_x_')
        index1 = path_to_data_file.rindex('_nA_')
        voltage = float(path_to_data_file[index1+4:index2])
    except ValueError:
        voltage = None
    return voltage


def first_measurement_voltage(path_to_data_file):
    if not isinstance(path_to_data_file, str):
        path_to_data_file = str(path_to_data_file)
    # The parsing of the position out of the name and save it
    try:
        index2 = path_to_data_file.rindex('.csv')
        index1 = path_to_data_file.rindex('_nA_')
        voltage = float(path_to_data_file[index1+4:index2])
    except ValueError:
        voltage = None
    return voltage


def standard_current(path_to_data_file):
    if not isinstance(path_to_data_file, str):
        path_to_data_file = str(path_to_data_file)

    # The parsing of the position out of the name and save it
    try:
        index1 = path_to_data_file.rindex('um_')
        index2 = path_to_data_file.rindex('_nA_nA_')
        current = float(comma_replace(path_to_data_file[index1+3:index2]))
    except ValueError:
        current = None
    return current
