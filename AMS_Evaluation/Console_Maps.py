from read_MATRIX import *

# Add the path to the save directory here
save_path = Path('/Users/nico_brosda/Desktop/iphc_python_misc/Results/maps/')
paths = ['/Users/nico_brosda/Desktop/iphc_python_misc/matrix_27052024/e2_500p_bottom_nA_2.csv',
         '/Users/nico_brosda/Desktop/iphc_python_misc/matrix_27052024/e2_500p_nA_2.csv',
         '/Users/nico_brosda/Desktop/iphc_python_misc/matrix_27052024/e2_500p_top_nA_2.csv']


# ---------------------------------------------------------------------------------------------------------------------
# Automatic Plots of all maps
for crit in ['screwsmaller_horizontal', 'noscrew', '_screw_', 'screw8_vertical', 'screw8_horizontal_', 'screw8_horizontal2_', 'beamshape_', 'beamshape2_']:
# for crit in ['noscrew', 'beamshape_']:
    if 'beamshape' in crit:
        ex = []
    else:
        ex = [38]
    out = interpret_map('/Users/nico_brosda/Desktop/iphc_python_misc/matrix_27052024/', crit, plot=True,
                        paths_of_norm_files=paths, excluded_channel=ex, do_normalization=True, save_path=save_path,
                        convert_param=False)
    out = interpret_map('/Users/nico_brosda/Desktop/iphc_python_misc/matrix_27052024/', crit, plot=True,
                        paths_of_norm_files=paths, excluded_channel=ex, do_normalization=False, save_path=save_path,
                        convert_param=False)
    out = interpret_map('/Users/nico_brosda/Desktop/iphc_python_misc/matrix_27052024/', crit, plot=True,
                        paths_of_norm_files=paths, excluded_channel=ex, save_path=save_path, do_normalization=True)
    out = interpret_map('/Users/nico_brosda/Desktop/iphc_python_misc/matrix_27052024/', crit, plot=True,
                        paths_of_norm_files=paths, excluded_channel=ex, save_path=save_path, do_normalization=False)
