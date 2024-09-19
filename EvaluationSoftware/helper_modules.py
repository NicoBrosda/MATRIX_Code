import pathlib
from pathlib import Path
import os
import codecs


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
    if isinstance(path, pathlib.PosixPath):
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
