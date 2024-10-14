from EvaluationSoftware.main import *
from EvaluationSoftware.readout_modules import ams_channel_assignment_readout
from EvaluationSoftware.normalization_modules import normalization_from_translated_array

mapping = Path('../Files/mapping.xlsx')
data = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:])-1 for k in data['direction_2']]

readout, position_parser = lambda x, y: ams_channel_assignment_readout(x, y, channel_assignment=channel_assignment), standard_position

folder_path = Path('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/matrix_111024/')
results_path = Path('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_111024/video/')

live_scan_array1 = [str(round(i+1, 0))+'_live1_' for i in range(9)]
# new_measurements_array_matrix = ['']

dark_path = Path('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/matrix_230924/')
dark_path = Path('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/matrix_111024/')

dark_paths_array1 = ['Dark_QuickYScan_0_um_2_nA_.csv']
# dark_paths_array1 = ['voltage_scan_no_beam_nA_1.9000000000000006_x_20.0_y_70.0.csv']

dark_paths_array_matrix = ['Array3_VoltageScan_dark_nA_1.8_x_0.0_y_40.0.csv']

norm_path = Path('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/matrix_230924/')
norm_array1 = ['Normalization2']
norm_array1 = ['uniformity_scan_']

new_measurements = live_scan_array1
map_storage = []
names = []
for k, crit in enumerate(new_measurements):
    print('-'*50)
    print(crit)
    print('-'*50)

    A = Analyzer((1, 128), 0.5, 0.0, readout=readout)

    # Correct sizing of the arrays
    A.diode_size = (0.5, 0.5)
    A.diode_size = (0.4, 0.4)
    A.diode_spacing = (0.1, 0.1)

    dark = dark_paths_array1

    A.set_measurement(folder_path, crit)
    A.set_dark_measurement(dark_path, dark)

    norm = norm_array1

    A.normalization(norm_path, norm, normalization_module=normalization_from_translated_array_v2)
    A.load_measurement()
    A.create_map(inverse=[True, False])
    map_storage.append(A.maps)
    names.append(A.name)

intensity_limits = [0, np.max([np.max(i[0]['z']) for i in map_storage])]
for i, image_map in enumerate(map_storage):
    A.name = names[i]
    A.maps = map_storage[i]
    A.plot_map(results_path / 'array_pixel/', pixel='fill',
               intensity_limits=intensity_limits)
    A.plot_map(results_path / 'array_contour/', pixel=False,
               intensity_limits=intensity_limits)

import cv2
import os

image_folder = results_path / 'array_contour/'
video_name = image_folder / 'array1_contour.mp4'

images = np.array([img for img in os.listdir(image_folder) if (img.endswith(".png"))])
images = images[np.argsort([float(i[0:1]) for i in images])]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
video = cv2.VideoWriter(video_name, fourcc, 1, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()