import pathlib


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

