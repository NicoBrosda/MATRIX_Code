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

mapping = Path('../Files/Mapping_SmallMatrix2.xlsx')
data2 = pd.read_excel(mapping, header=None)
mapping_map = data2.to_numpy().flatten()
translated_mapping_small2 = np.array([direction2[np.argwhere(direction1 == i)[0][0]]-1 for i in mapping_map])

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_161224/')
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Results_161224/energy_imaging/')

dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_161224/')
norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_211124/')
norm = '8_2DSmall_yscan_'
norm_func = lambda list_of_files, instance, method='least_squares': normalization_from_translated_array_v3(
    list_of_files, instance, method, align_lines=True)

readout, position_parser, voltage_parser, current_parser = lambda x, y: ams_2D_assignment_readout(x, y, channel_assignment=translated_mapping_small2), standard_position, standard_voltage, current3

measurements = (['exp3_bragg_p01_']+[f'exp4_bragg_p0{i}_' for i in range(2, 10)]+
                [f'exp4_bragg_p{i}_' for i in range(10, 21)]+['exp6_star_p01_', 'exp7_star_p06_', 'exp8_star_p06_',
                                                              'exp10_star_p01_', 'exp13_star_p15_', 'exp14_star_p09_'])

'''
# Line Scan image by image
results_path2 = results_path / '8_2DSmall_yscan_'

A.set_measurement(folder_path, norm)
measurement_files = np.array([str(i)[len(str(folder_path))+1:] for i in A.measurement_files], dtype=str)
measurement_files = measurement_files[np.argsort([standard_position(i)[1] for i in measurement_files])]

for file in tqdm(measurement_files[0:], colour='blue'):
    A.set_measurement(folder_path, file)
    A.load_measurement()
    A.create_map(inverse=[True, False])
    intensity_limits = [0, np.max(A.maps[0]['z'])]
    A.plot_map(results_path2, pixel='fill', intensity_limits=intensity_limits)

# '''

for k, crit in enumerate(measurements[0:]):
    print('-'*50)
    print(crit)
    print('-'*50)
    continue
    if 'bragg' in crit:
        continue

    A = Analyzer((11, 11), 0.4, 0.1, readout=readout, voltage_parser=voltage_parser, current_parser=current_parser)
    A.set_measurement(folder_path, crit)
    if 'bragg' in crit:
        dark = ['exp1_dark_voltage_scan_nA_1.9_x_22.0_y_67.75']
    elif 'star' in crit:
        dark = ['exp1_dark_voltage_scan_nA_1.1_x_22.0_y_67.75']
    A.set_dark_measurement(dark_path, dark)
    A.normalization(norm_path, norm, normalization_module=norm_func)
    A.norm_factor[3][5] = 1
    A.norm_factor[4][5] = 1
    A.load_measurement()
    A.create_map(inverse=[True, False])

    if 'exp6_' in crit:
        intensity_limits = [0, np.max(A.maps[0]['z'])]

    if 'exp6_' in crit:
        label = r'23.69$\,$MeV'  # P01
        A.name = '_1_P01'
    elif 'exp7_' in crit:
        label = r'20.19$\,$MeV'  # P06
        A.name = '_2_P06'
    elif 'exp8_' in crit:
        label = r'15.16$\,$MeV'  # P12
        A.name = '_4_P12'
    elif 'exp10_' in crit:
        label = r'8.00$\,$MeV'  # P18
        A.name = '_6_P18'
    elif 'exp13_' in crit:
        label = r'12.01$\,$MeV'  # P15
        A.name = '_5_P15'
    elif 'exp14_' in crit:
        label = r'17.81$\,$MeV'  # P09
        A.name = '_3_P09'

    txt_posi = [0.03, 0.93]
    A.plot_map(results_path / 'pixel/', pixel='fill', intensity_limits=intensity_limits, insert_txt=[txt_posi, label, 15])
    A.plot_map(results_path / 'contour/', pixel=False, intensity_limits=intensity_limits, insert_txt=[txt_posi, label, 15])


import cv2
import os

# Do 2 videos for contour and pixel: 1 with increased fps (x5) and with realistic fps (42 fpm = 0.7 fps)
image_folder_pixel = results_path / 'pixel/'
image_folder_contour = results_path / 'contour/'

video_name_pixel_original = results_path / 'pixel.mp4'
video_name_contour_original = results_path / 'contour.mp4'


# videos of pixel images
images = np.array([img for img in os.listdir(image_folder_pixel) if (img.endswith(".png"))])
print(len(images))
images = images[np.argsort([float(i[1:i.index('_P')]) for i in images])]
frame = cv2.imread(os.path.join(image_folder_pixel, images[0]))
height, width, layers = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case

video = cv2.VideoWriter(video_name_pixel_original, fourcc, 0.5, (width, height))
for image in images:
    video.write(cv2.imread(os.path.join(image_folder_pixel, image)))
cv2.destroyAllWindows()
video.release()

# videos of contour images
images = np.array([img for img in os.listdir(image_folder_contour) if (img.endswith(".png"))])
images = images[np.argsort([float(i[1:i.index('_P')]) for i in images])]
frame = cv2.imread(os.path.join(image_folder_contour, images[0]))
height, width, layers = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case

video = cv2.VideoWriter(video_name_contour_original, fourcc, 0.5, (width, height))
for image in images:
    video.write(cv2.imread(os.path.join(image_folder_contour, image)))
cv2.destroyAllWindows()
video.release()
