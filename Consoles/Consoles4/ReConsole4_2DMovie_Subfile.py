import matplotlib.pyplot as plt
import numpy as np

from EvaluationSoftware.main import *

# To get the mapping of the 2d array correct I have to translate the mapping of the contacts
# The given info by The-Duc are for the 1 direction of mapping, but we have by standard the other mapping
# Therefore, I need the correct channel assignment between these two to replace the channels in the 2D pixel placement

mapping = Path('../../Files/mapping.xlsx')
direction1 = pd.read_excel(mapping, header=1)
direction1 = np.array([int(k[-3:]) for k in direction1['direction_1']])
direction2 = pd.read_excel(mapping, header=1)
direction2 = np.array([int(k[-3:]) for k in direction2['direction_2']])

mapping = Path('../../Files/Mapping_MatrixArray.xlsx')
data2 = pd.read_excel(mapping, header=None)
mapping_map = data2.to_numpy().flatten()
# print(mapping_map)
translated_mapping = np.array([direction2[np.argwhere(direction1 == i)[0][0]]-1 for i in mapping_map])
# print(translated_mapping)

readout, position_parser = lambda x, y: ams_2D_assignment_readout(x, y, channel_assignment=translated_mapping), standard_position

A = Analyzer((11, 11), 0.8, 0.2, readout=readout)

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_111024/')

dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_111024/')
matrix_dark = ['2DLarge_dark_200_um_0_nA__nA_1.9_x_21.0_y_70.35.csv']

zero_frame = 1305
frame_start = 1305
frame_space = 1
fps = 0.7
frames = 20
frames = [frame_start + i*frame_space for i in range(frames)]
# '''
results_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/Results_111024/MatrixArray/movie_subres{frame_start}_{frame_space}_{frames}/')


crit = '2DLarge_movieScan_'
files = os.listdir(folder_path)
map_storage = []
names = []
for i in range(len(array_txt_file_search(files, searchlist=[crit], txt_file=False, file_suffix='.csv'))):
    if i+1 in frames:
        for j in tqdm(range(99)):
            readout = lambda x, y: ams_2D_assignment_fast_avg(x, y, channel_assignment=translated_mapping, sample_size=[j*10, (j+1)*10], keep_first_row=True)
            A.readout = readout
            A.set_measurement(folder_path, '_'+str(i+1)+'_'+crit)
            A.set_dark_measurement(dark_path, matrix_dark)
            norm_func = lambda list_of_files, instance, method='least_squares': normalization_from_translated_array_v3(
                list_of_files, instance, method, align_lines=True)
            A.normalization(folder_path, ['2DLarge_YScan_'], normalization_module=norm_func)
            A.load_measurement()
            A.create_map(inverse=[True, False])
            map_storage.append(A.maps)
            names.append(f'_{i+1}{j}_{crit}')
            break
        break

plt.error

intensity_limits = [0, np.max([np.max(i[0]['z']) for i in map_storage])*0.8]
for i, image_map in tqdm(enumerate(map_storage)):
    A.name = names[i]
    A.maps = map_storage[i]
    A.maps[0]['z'] = zero_pixel_replace(A.maps[0]['z'])
    print(np.shape(A.maps[0]['z']))
    A.plot_map(results_path / 'pixel/', pixel='fill', save_format='png', imshow=True,
               intensity_limits=intensity_limits)

# '''
import cv2
import os

# Do 2 videos for contour and pixel: 1 with increased fps (x5) and with realistic fps (42 fpm = 0.7 fps)
image_folder_pixel = results_path / 'pixel/'

video_name_pixel_original = results_path / f'movie_subres{frame_start}_{frame_space}_{frames}.mp4'

# videos of pixel images
images = np.array([img for img in os.listdir(image_folder_pixel) if (img.endswith(".png"))])
images = images[np.argsort([float(i[1:i.index(f'_{crit}')]) for i in images])]
frame = cv2.imread(os.path.join(image_folder_pixel, images[0]))
height, width, layers = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case

video = cv2.VideoWriter(video_name_pixel_original, fourcc, 100, (width, height))
for image in images:
    video.write(cv2.imread(os.path.join(image_folder_pixel, image)))
cv2.destroyAllWindows()
video.release()

