def standard_position(path_to_data_file):
    if not isinstance(path_to_data_file, str):
        path_to_data_file = str(path_to_data_file)
    # The parsing of the position out of the name and save it
    try:
        index3 = path_to_data_file.index('.csv')
        index2 = path_to_data_file.index('_y_')
        index1 = path_to_data_file.index('_x_')
        pos_x = float(path_to_data_file[index1 + 3:index2])
        pos_y = float(path_to_data_file[index2 + 3:index3])
    except ValueError:
        pos_x, pos_y = None, None
    return pos_x, pos_y


def first_measurements_position(path_to_data_file, y=0):
    if not isinstance(path_to_data_file, str):
        path_to_data_file = str(path_to_data_file)
    # The parsing of the position out of the name and save it
    i = 1
    pos = None
    while True:
        try:
            index = path_to_data_file.index('.csv')
            pos = float(path_to_data_file[(index - i):index])
        except ValueError:
            break
        i += 1
    return pos, y
