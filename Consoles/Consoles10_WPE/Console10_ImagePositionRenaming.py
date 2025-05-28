import os

from EvaluationSoftware.main import *
from EvaluationSoftware.helper_modules import rename_files
import contextlib
import io
import shutil

from Plot_Methods.helper_functions import array_txt_file_search

folder_path = Path('/Users/nico_brosda/Desktop/matrix_210525/Nouveau dossier/')
folder_image1 = Path('/Users/nico_brosda/Desktop/matrix_210525/exp52_eye_image/')
folder_image2 = Path('/Users/nico_brosda/Desktop/matrix_210525/exp53_eye_image/')
folder_image3 = Path('/Users/nico_brosda/Desktop/matrix_210525/exp56_phantom_9_steps/')
image_path = Path('/Users/nico_brosda/Desktop/matrix_210525/Images/')


image1 = [f'_{i}.csv' for i in range(0, 21)]
image2 = [f'_{i}.csv' for i in range(0, 21)]
image3 = [f'_{i}.csv' for i in range(0, 9)]

images = [image1, image2, image3]

# Time structure of images
# '''
for j, folder_path in enumerate([folder_image1, folder_image2, folder_image3]):
    files = array_txt_file_search(os.listdir(folder_path), searchlist=[''], file_suffix='.csv', txt_file=False)
    print(files)
    if j == 0:
        phrase = 'exp52_eye_100MeV_1MU_volt_1,9'
    elif j == 1:
        phrase = 'exp53_eye_150MeV_1MU_volt_1,9'
    else:
        phrase = 'exp56_StepPhantom_100MeV_0,1MU_volt_1,9'

    for i, file in enumerate(files):
        # Image frame 0 of image 2 was not recorded
        if i == 0 and j == 1:
            continue
        if j <= 1:
            rename = f'{phrase}_x_{(i % 7) * 5 + 50:.0f}_y_{-(i // 7) * 5 + 50:.0f}.csv'
        else:
            rename = f'{phrase}_x_{(i % 3) * 5 + 50:.0f}_y_{-(i // 3) * 5 + 50:.0f}.csv'
        print(rename)
        # os.rename(folder_path / file, folder_path / rename)
        shutil.copy(folder_path / file, image_path / rename)


