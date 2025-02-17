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


def standard_current(path_to_data_file, conversion_factor=1):
    if not isinstance(path_to_data_file, str):
        path_to_data_file = str(path_to_data_file)

    # The parsing of the position out of the name and save it
    try:
        index1 = path_to_data_file.rindex('um_')
        index2 = path_to_data_file.rindex('_nA_nA_')
        current = float(comma_replace(path_to_data_file[index1+3:index2])) * conversion_factor
    except ValueError:
        current = None
    return current


def current2(path_to_data_file, conversion_factor=1):
    # Current as format: _5_nA_nA_
    if not isinstance(path_to_data_file, str):
        path_to_data_file = str(path_to_data_file)

    # The parsing of the position out of the name and save it
    try:
        index2 = path_to_data_file.rindex('_nA_nA_')
        index1 = path_to_data_file[:index2].rindex('_')
        current = float(comma_replace(path_to_data_file[index1+1:index2])) * conversion_factor
    except ValueError:
        current = None
    return current


def current3(path_to_data_file, conversion_factor=1):
    # Current as format: _5nA_nA_
    if not isinstance(path_to_data_file, str):
        path_to_data_file = str(path_to_data_file)

    # The parsing of the position out of the name and save it
    try:
        index2 = path_to_data_file.rindex('nA_nA_')
        index1 = path_to_data_file[:index2].rindex('_')
        current = float(comma_replace(path_to_data_file[index1+1:index2])) * conversion_factor
    except ValueError:
        current = None
    return current


def current4(path_to_data_file, conversion_factor=0.688):
    # Current as format: expN_nA with manual distribution to experiment number
    conversion_factor = 0.688
    if not isinstance(path_to_data_file, str):
        path_to_data_file = str(path_to_data_file)

    # The parsing of the position out of the name and save it
    try:
        index2 = path_to_data_file.rindex('_nA_')
        index1 = path_to_data_file.rindex('exp')
        current = comma_replace(path_to_data_file[index1:index2])
        current_transcript = {'exp8': 1, 'exp7': 2, 'exp6': 4, 'exp5': 8, 'exp4': 16, 'exp3': 25, 'exp2': 0, 'exp1': 0}
        current = current_transcript[current] * conversion_factor
    except ValueError:
        current = None
    return current