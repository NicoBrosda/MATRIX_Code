from AMS_Evaluation.read_MATRIX import *
from tqdm import tqdm

# Add the path to the save directory here
save_path = Path('/Users/nico_brosda/Desktop/iphc_python_misc/Results/maps_e1/')
paths = ['/Users/nico_brosda/Desktop/iphc_python_misc/matrix_27052024/e2_500p_bottom_nA_2.csv',
         '/Users/nico_brosda/Desktop/iphc_python_misc/matrix_27052024/e2_500p_nA_2.csv',
         '/Users/nico_brosda/Desktop/iphc_python_misc/matrix_27052024/e2_500p_top_nA_2.csv']


# ---------------------------------------------------------------------------------------------------------------------
# Automatic Plots of all maps
# for crit in tqdm(['screwsmaller_horizontal', 'noscrew', '_screw_', 'screw8_vertical', 'screw8_horizontal_', 'screw8_horizontal2_', 'beamshape_', 'beamshape2_']):
# for crit in tqdm(['noscrew']):
for crit in tqdm(['scan_2_nA', 'scan_3_nA']):
    ex = [25, 26]
    if 'beamshape2_' in crit:
        x_step = 0.5
    else:
        x_step = 0.25

    out = interpret_map('/Users/nico_brosda/Desktop/iphc_python_misc/matrix_27052024/d2/', crit, plot=True,
                        paths_of_norm_files=paths, excluded_channel=ex, do_normalization=False, save_path=save_path,
                        convert_param=True, super_resolution=False, contour=True, x_stepwidth=x_step)