import numpy as np

from EvaluationSoftware.main import *
from EvaluationSoftware.helper_modules import rename_files

# To get the mapping of the 2d array correct I have to translate the mapping of the contacts
# The given info by The-Duc are for the 1 direction of mapping, but we have by standard the other mapping
# Therefore, I need the correct channel assignment between these two to replace the channels in the 2D pixel placement

mapping = Path('../Files/mapping.xlsx')
direction1 = pd.read_excel(mapping, header=1)
direction1 = np.array([int(k[-3:]) for k in direction1['direction_1']])
direction2 = pd.read_excel(mapping, header=1)
direction2 = np.array([int(k[-3:]) for k in direction2['direction_2']])

mapping = Path('../Files/Mapping_BigMatrix_2.xlsx')
data2 = pd.read_excel(mapping, header=None)
mapping_map = data2.to_numpy().flatten()
translated_mapping = np.array([direction2[np.argwhere(direction1 == i)[0][0]]-1 for i in mapping_map])

readout, position_parser = lambda x, y: ams_2D_assignment_readout(x, y, channel_assignment=translated_mapping), standard_position

A = Analyzer((11, 11), 0.8, 0.2, readout=readout)

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_161224/')
results_root = Path('/Users/nico_brosda/Cyrce_Messungen/Results_161224/movies_normed/')

movie_crits = ['exp17_', 'exp20_', 'exp21_', 'exp22_', 'exp23_']

dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_161224/')
dark_crit = ['exp16_dark_big_matrix_2_nA_1.8_x_28.25_y_0.0', 'exp18_dark_big_matrix_2_nA_1.8_x_28.25_y_0.0',
             'exp18_dark_big_matrix_2_nA_1.1_x_28.25_y_0.0', 'exp18_dark_big_matrix_2_nA_1.1_x_28.25_y_0.0',
             'exp18_dark_big_matrix_2_nA_1.8_x_28.25_y_0.0']

for j, crit in enumerate(movie_crits):
    print(f'Movie from crit {crit}')
    print('-'*50)
    results_path = results_root / crit
    matrix_dark = [dark_crit[j]]
    A.set_dark_measurement(dark_path, matrix_dark)
    # '''
    norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
    norm_func = lambda list_of_files, instance, method='least_squares': normalization_from_translated_array_v3(
            list_of_files, instance, method, align_lines=True)
    A.normalization(norm_path, ['2DLarge_YTranslation_'], normalization_module=norm_func)
    A.norm_factor[8, 3] = 1
    A.norm_factor[7, 3] = 1
    # '''
    # print(os.listdir(folder_path))
    # print(array_txt_file_search(os.listdir(folder_path), searchlist=['exp'], file_suffix='.csv', txt_file=False))
    # print(array_txt_file_search(os.listdir(folder_path), searchlist=[crit], file_suffix='.csv', txt_file=False))
    if array_txt_file_search(os.listdir(folder_path), searchlist=[crit], file_suffix='.csv', txt_file=False)[0][0] != '_':
        rename_files(folder_path, crit, file_suffix='.csv')  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # '''
    files = os.listdir(folder_path)
    map_storage = []
    names = []
    for i in tqdm(range(len(array_txt_file_search(files, searchlist=[crit], txt_file=False, file_suffix='.csv'))),
                  colour='green'):
        A.set_measurement(folder_path, '_'+str(i+1)+'_'+crit)
        A.load_measurement()
        A.create_map(inverse=[True, False])
        map_storage.append(A.maps)
        names.append(A.name)

    intensity_limits = [0, np.max([np.max(i[0]['z']) for i in map_storage])]
    for i, image_map in tqdm(enumerate(map_storage), colour='blue'):
        A.name = names[i]
        A.maps = map_storage[i]
        A.maps[0]['z'] = zero_pixel_replace(A.maps[0]['z'])
        if i == 0:
            A.plot_map(None, pixel='fill',
                       intensity_limits=intensity_limits)
            fig = plt.gcf()
            bbox1 = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.15)

            A.plot_map(None, pixel='contour',
                       intensity_limits=intensity_limits)
            fig = plt.gcf()
            bbox2 = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.15)

        A.plot_map(results_path / 'pixel/', pixel='fill', intensity_limits=intensity_limits, bbox=bbox1)
        A.plot_map(results_path / 'contour/', pixel=False, intensity_limits=intensity_limits, bbox=bbox2)

    # '''
    import cv2
    import os

    # Do 2 videos for contour and pixel: 1 with increased fps (x5) and with realistic fps (42 fpm = 0.7 fps)
    image_folder_pixel = results_path / 'pixel/'
    image_folder_contour = results_path / 'contour/'

    video_name_pixel_original = results_path / 'original_pixel.mp4'
    video_name_pixel_increased = results_path / 'x30increased_pixel.mp4'
    video_name_contour_original = results_path / 'original_contour.mp4'
    video_name_contour_increased = results_path / 'x30increased_contour.mp4'


    # videos of pixel images
    images = np.array([img for img in os.listdir(image_folder_pixel) if (img.endswith(".png"))])
    print(len(images))
    images = images[np.argsort([float(i[1:i.index('_'+crit)]) for i in images])]
    frame = cv2.imread(os.path.join(image_folder_pixel, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case

    video = cv2.VideoWriter(video_name_pixel_original, fourcc, 1., (width, height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder_pixel, image)))
    cv2.destroyAllWindows()
    video.release()

    video = cv2.VideoWriter(video_name_pixel_increased, fourcc, 30, (width, height))
    for i, image in enumerate(images):
        video.write(cv2.imread(os.path.join(image_folder_pixel, image)))
    cv2.destroyAllWindows()
    video.release()

    # videos of contour images
    images = np.array([img for img in os.listdir(image_folder_contour) if (img.endswith(".png"))])
    images = images[np.argsort([float(i[1:i.index('_'+crit)]) for i in images])]
    frame = cv2.imread(os.path.join(image_folder_contour, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case

    video = cv2.VideoWriter(video_name_contour_original, fourcc, 1., (width, height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder_contour, image)))
    cv2.destroyAllWindows()
    video.release()


    video = cv2.VideoWriter(video_name_contour_increased, fourcc, 30, (width, height))
    for i, image in enumerate(images):
        video.write(cv2.imread(os.path.join(image_folder_contour, image)))
    cv2.destroyAllWindows()
    video.release()
    # '''
