from EvaluationSoftware.main import *
import cv2
import os

mapping = Path('../../Files/mapping.xlsx')
data = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:])-1 for k in data['direction_2']]
readout, position_parser = lambda x, y: ams_2line_readout(x, y, channel_assignment=channel_assignment), standard_position

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Results_221024/SampleSize/Videos/')

new_measurements = ['2Line_Beam_', '_GafCompMisc_', '_GafComp200_', '_GafComp400_', '_GafComp40_', '_GafCompLogo_',
                    '_GafCompPEEK_', '_MouseFoot_', '_MouseFoot2_']
live_scan_array1 = [str(round(i+1, 0))+'_live1_' for i in range(9)]

dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')

dark_paths_array1 = ['2Line_DarkVoltageScan_200_ um_0_nA_nA_1.9_x_22.0_y_66.625.csv']

norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
norm_array1 = ['2Line_YScan_']

for k, crit in enumerate(new_measurements):
    print('-' * 50)
    print(crit)
    print('-' * 50)
    readout = lambda x, y: ams_2line_readout(x, y, channel_assignment=channel_assignment)
    A = Analyzer((2, 64), (0.4, 0.4), (0.1, 0.1), readout=readout, diode_offset=[[0, - 0.25], np.zeros(64)])
    dark = dark_paths_array1
    A.set_measurement(folder_path, crit)
    A.set_dark_measurement(dark_path, dark)
    norm = norm_array1
    norm_func = lambda list_of_files, instance, method='least_squares': normalization_from_translated_array_v3(
        list_of_files, instance, method, align_lines=True)
    A.normalization(norm_path, norm, normalization_module=norm_func)
    A.load_measurement()
    A.create_map(inverse=[True, False])
    intensity_limits = [0, np.max(A.maps[0]['z'])]

    txt_posi = [0.03, 0.93]
    # '''
    for sample_size in [300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000]:  # range(1, 201):
        readout = lambda x, y: ams_2line_readout(x, y, channel_assignment=channel_assignment, sample_size=sample_size)
        A.readout = readout
        A.load_measurement()
        A.create_map(inverse=[True, False])
        A.name = f"_{sample_size}__"
        A.plot_map(results_path / crit, pixel='fill', intensity_limits=intensity_limits, insert_txt=[txt_posi, f"Samples: {sample_size}", 15])
    # '''

    image_folder_pixel = results_path / crit
    video_name = results_path / f'Sample_{crit}.mp4'

    # videos of pixel images
    images = np.array([img for img in os.listdir(image_folder_pixel) if (img.endswith(".png"))])
    print(len(images))
    images = images[np.argsort([float(i[1:i.index('__')]) for i in images])]
    frame = cv2.imread(os.path.join(image_folder_pixel, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    video = cv2.VideoWriter(video_name, fourcc, 10, (width, height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder_pixel, image)))
    cv2.destroyAllWindows()
    video.release()