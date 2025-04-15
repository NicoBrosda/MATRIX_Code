from copy import deepcopy

from EvaluationSoftware.main import *
from EvaluationSoftware.readout_modules import ams_channel_assignment_readout
from EvaluationSoftware.normalization_modules import normalization_from_translated_array

mapping = Path('../../Files/mapping.xlsx')
data = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:])-1 for k in data['direction_2']]
readout, position_parser, voltage_parser, current_parser = (
    lambda x, y: ams_2line_readout(x, y, channel_assignment=channel_assignment),
    standard_position,
    standard_voltage,
    current5
)

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_260325/')
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Results_260325/Maps/')

# ----------------------- Short summary log of measurements -----------------------
# Exp 1 : Dark voltage scan ["dark_"]
# Exp 2 - 5 : Linearity (100 pA - 1 nA at target with 400 um diffuser) ["linearity1_"]
# -> exp 6 empty (not possible)
# -> Until here no um in filename for diffuser thickness - ================= Fixed =================
# Exp 7 - 9 : Norming at different detector voltages ["norm{}V_"]
# Exp 10 - 13 : Norming at different proton energies ["norm{}V_P{}_"]
# Exp 14 - 32 : Mapping of wheel aperture for all wheel positions, P0 - P18 ["energydiffmap_P{}_"]
# Exp 33 - 35 : Norming at different voltages for low proton energy ["norm{}V_P{}_"]
# -> Exp 33 named with 1.9 V instead of 1,9 V - ================= Fixed =================
# Exp 36 - 43 : Mapping of PEEK wedge (wrong position) P0 - P6, P18 ["PEEKwedge_P{}_"]
# Exp 44 - 62 : Mapping of PEEK wedge (correct position) P0, P18, P1 - P17 ["PEEKwedge_P{}_"]
# -> exp 47 with P3 instead of P2 in filename!!! - ================= Fixed =================
# Exp 63 : Increased distance 10 mm, P0 ["LargerGap10mm_P{}_"]
# Exp 64 : Dark Voltage End of Day ["darkEnd_"]
# Exp 65 : Increased distance 10 mm, P7 ["LargerGap10mm_P{}_"]
# ------------------------ End day1 ----------------------------------------------
# Exp 66 : Dark Voltage Scan Day2 1-2V ["Dark_"]
# Exp 67 : Dark Voltage Scan Day2 0-2V ["Dark2_"]
# Exp 68 : Voltage Scan 0.8-2V with beam (PEEK wedge, P7, 10 mm distance) ["BeamCurrent1_"]
# Exp 69 - 71 : Increased distance 10 mm, P7, P12, P16 ["Distance10mm_P{}_"]
# Exp 72 - 75 : Increased distance 20 mm, P0, P7, P12, P16 ["Distance20mm_P{}_"]
# Gafchromic I - VII
# -> Switch to 200 um diffuser
# Exp 76 : Norming day2 P0 and 1.9 V ["normday2_"]
# Exp 77 - 95 : Mapping of wheel aperture for all wheel positions, P0 - P18 ["energyDep_"]
# -> Exp 80 contains two runs - only the run with _bis_ is good (no beam in other run) - ====== Fixed =======
# Exp 96 - 97 : Mapping of PEEK wedge (wrong position) P0, P18 ["PEEKWedge_P{}_"]
# Exp 98 - 117 : Mapping of PEEK wedge (correct position) P18, P0 - P17, P19 ["PEEKWedge_P{}_"]
# Gafchromic VIII - XI
# Exp 118 - 125 : Wedge border in middle of aperture P19 - P12 ["PEEKWedgeMiddle_P{}_"]
# -> Exp 120 named labeled falsely with 118 - needs to be identified with P19 / P17 for real Exp 120 - ==== Fixed =====
# Gafchromic XII
# Exp 126 - 128 : Straggling test distance 5 mm - P0, P12, Misc ["Round8mm_5mm_P{}_", "Misc_5mm_P0_"]
# Exp 129 - 131 : Straggling test distance 10 mm - Misc, P0, P12 ["Round8mm_10mm_P{}_", "Misc_10mm_P0_"]
# Exp 132 - 134 : Straggling test distance 20 mm - P12, P0, Misc ["Round8mm_20mm_P{}_", "Misc_20mm_P0_"]
# Exp 135 - 137 : Straggling test distance 40 mm - Misc, P0, P12 ["Round8mm_40mm_P{}_", "Misc_40mm_P0_"]
# Exp 138 : Dark voltage scan end 0-2V ["DarkEnd_"]

new_measurements = []
'''
new_measurements += [f'exp{i+14}_energydiffmap_P{i}_' for i in range(19)]
new_measurements += [f'exp{i+36}_PEEKwedge_P{i}_' if i < 7 else f'exp{i+36}_PEEKwedge_P18_' for i in range(8)]
new_measurements += [f'exp{44}_PEEKwedge_P{0}_', f'exp{45}_PEEKwedge_P{18}_']
new_measurements += [f'exp{i+46}_PEEKwedge_P{i+1}_' for i in range(17)]
new_measurements += [f'exp{63}_LargerGap10mm_P{0}_']
new_measurements += [f'exp{65}_LargerGap10mm_P{7}_']
new_measurements += [f'exp{69}_Distance10mm_P{7}_', f'exp{70}_Distance10mm_P{12}_', f'exp{71}_Distance10mm_P{16}_']
new_measurements += [f'exp{72}_Distance20mm_P{0}_', f'exp{73}_Distance20mm_P{7}_', f'exp{74}_Distance20mm_P{12}_', f'exp{75}_Distance20mm_P{16}_']
new_measurements += [f'exp{i+77}_energyDep_P{i}_' for i in range(19)]
new_measurements += [f'exp{96}_PEEKWedge_P{0}_', f'exp{97}_PEEKWedge_P{18}_', f'exp{98}_PEEKWedge_P{18}_']
new_measurements += [f'exp{i+99}_PEEKWedge_P{i}_' for i in range(18)]
new_measurements += [f'exp{117}_PEEKWedge_P{19}_']
new_measurements += [f'exp{i+118}_PEEKWedgeMiddle_P{19-i}_' for i in range(8)]
# '''
new_measurements += [[f'exp{i+14}_energydiffmap_P{i}_' for i in range(19)]]
new_measurements += [[f'exp{44}_PEEKwedge_P{0}_'] + [f'exp{i+46}_PEEKwedge_P{i+1}_' for i in range(17)] + [f'exp{45}_PEEKwedge_P{18}_']]
new_measurements += [[f'exp{i+77}_energyDep_P{i}_' for i in range(19)]]
new_measurements += [[f'exp{i+99}_PEEKWedge_P{i}_' for i in range(18)] + [f'exp{98}_PEEKWedge_P{18}_'] + [f'exp{117}_PEEKWedge_P{19}_']]
new_measurements += [[f'exp{i+118}_PEEKWedgeMiddle_P{19-i}_' for i in range(8)][::-1]]

set_names = ['Aperture400um', 'PEEKwedge400um', 'Aperture200um', 'PEEKwedge200um', 'PEEKwedgeMiddle200um']

dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_260325/')

dark_paths_array1 = ['exp1_dark_0nA_400um_nA_1.9_x_20.0_y_68.0', 'exp64_darkEnd_0.5nA_400um_nA_1.9_x_20.0_y_68.0',
                     '2exp66_Dark_0.0nA_0um_nA_1.9_x_20.0_y_68.0', '2exp138_DarkEnd_0nA_200um_nA_1.9_x_20.0_y_68.0']

norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_260325/')
norm_array1 = ['exp7_norm1,9V_']
# norm_array1 = ['2exp76_normday2_P0_']

for j, measurement_set in enumerate(new_measurements):
    print(':' * 50)
    print(set_names[j])
    print(':' * 50)
    # Load in the wheel energies and assign the colormap
    if j < 2:
        diff = 400
    else:
        diff = 200
    dat = pd.read_csv(f'../../Files/energies_after_wheel_diffusor{diff}.txt', sep='\t',
                                 header=4, names=['position', 'thickness', 'energies'])
    energy_cmap = sns.color_palette("crest_r", as_cmap=True)
    energy_colormapper = lambda energy: color_mapper(energy, np.min(dat['energies']), np.max(dat['energies']))
    energy_color = lambda energy: energy_cmap(energy_colormapper(energy))

    cache = []
    for k, crit in enumerate(measurement_set):
        print('-'*50)
        print(crit)
        print('-'*50)

        A = Analyzer((2, 64), (0.4, 0.4), (0.1, 0.1), readout=readout,
                     diode_offset=[[0, - 0.25], np.zeros(64)], position_parser=position_parser,
                     voltage_parser=voltage_parser, current_parser=current_parser)
        dark = dark_paths_array1
        A.set_measurement(folder_path, crit)
        A.load_measurement()
        A.set_dark_measurement(dark_path, dark)
        norm = norm_array1
        norm_func = lambda list_of_files, instance, method='least_squares': normalization_from_translated_array_v3(
            list_of_files, instance, method, align_lines=True)
        A.normalization(norm_path, norm, normalization_module=norm_func)
        A.update_measurement()
        A.create_map(inverse=[True, False])
        cache.append(A.maps[0])

    intensity_limits = [0, np.max([np.max(i['z']) for i in cache])]
    A.maps = cache
    A.rescale_maps()
    cache = A.maps

    # ----------------------------------------------------------------------------------------------------------------
    # Plot maps with uniform intensity scale and coordinate axis limits
    # ----------------------------------------------------------------------------------------------------------------
    '''
    for k, obj in enumerate(cache):
        A.maps = [obj]
        A.name = measurement_set[k]
        crit = A.name
        txt_posi = [0.03, 0.87]
        wpos = crit[crit.rindex('_P')+2:]
        try:
            wpos = int(wpos[:wpos.index('_')])
        except ValueError:
            continue

        insert_txt = [txt_posi, f"Wheel position P{wpos} \n Proton energy {dat['energies'][wpos]: .2f}$\,$MeV", 15,
                      energy_color(dat['energies'][wpos])]
        A.plot_map(results_path / f'series/{set_names[j]}/', pixel='fill', intensity_limits=intensity_limits,
                   insert_txt=insert_txt, save_format='.png')

    # Merge single images into a video
    import cv2

    image_folder = Path(results_path / f'series/{set_names[j]}/')
    images = measurement_set
    video_name = results_path / f'series/{set_names[j]}.mp4'

    frame = cv2.imread(image_folder / f'{images[0]}_map_fill.png')
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case

    video = cv2.VideoWriter(video_name, fourcc, 1., (width, height))
    for image in images:
        video.write(cv2.imread(image_folder / f'{image}_map_fill.png'))
    cv2.destroyAllWindows()
    video.release()
    # '''

    # ----------------------------------------------------------------------------------------------------------------
    # Plot maps with uniform intensity scale and coordinate axis limits - better colour scale - rocket_r
    # ----------------------------------------------------------------------------------------------------------------
    '''
    for k, obj in enumerate(cache):
        A.maps = [obj]
        A.name = measurement_set[k]
        crit = A.name
        txt_posi = [0.03, 0.87]
        wpos = crit[crit.rindex('_P')+2:]
        try:
            wpos = int(wpos[:wpos.index('_')])
        except ValueError:
            continue

        insert_txt = [txt_posi, f"Wheel position P{wpos} \n Proton energy {dat['energies'][wpos]: .2f}$\,$MeV", 15,
                      energy_color(dat['energies'][wpos])]
        A.plot_map(results_path / f'series2/{set_names[j]}/', pixel='fill', intensity_limits=intensity_limits,
                   insert_txt=insert_txt, save_format='.png', cmap=sns.color_palette("rocket_r", as_cmap=True))

    # Merge single images into a video
    import cv2

    image_folder = Path(results_path / f'series2/{set_names[j]}/')
    images = measurement_set
    video_name = results_path / f'series2/{set_names[j]}.mp4'

    frame = cv2.imread(image_folder / f'{images[0]}_map_fill.png')
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case

    video = cv2.VideoWriter(video_name, fourcc, 1., (width, height))
    for image in images:
        video.write(cv2.imread(image_folder / f'{image}_map_fill.png'))
    cv2.destroyAllWindows()
    video.release()
    # '''

    # ----------------------------------------------------------------------------------------------------------------
    # Special case: Plotting PEEK wedge signal with aperture in background
    # ----------------------------------------------------------------------------------------------------------------
    # '''
    if 'Aperture' in set_names[j]:
        cache_ap = cache
        continue

    for k, obj in enumerate(cache):
        A.maps = [obj]
        A.name = measurement_set[k]
        crit = A.name
        txt_posi = [0.03, 0.87]
        wpos = crit[crit.rindex('_P')+2:]
        try:
            wpos = int(wpos[:wpos.index('_')])
        except ValueError:
            continue

        try:
            ap_map = [cache_ap[wpos]]
        except IndexError:
            continue

        insert_txt = [txt_posi, f"Wheel position P{wpos} \n Proton energy {dat['energies'][wpos]: .2f}$\,$MeV", 15,
                      energy_color(dat['energies'][wpos])]
        fig, ax = plt.subplots()
        save_path = results_path / f'series_overlay/{set_names[j]}/'
        A.plot_map(None, pixel='fill', intensity_limits=intensity_limits,
                   insert_txt=insert_txt, save_format='.png', cmap=sns.color_palette("rocket_r", as_cmap=True),
                   ax_in=ax, fig_in=fig, colorbar=True)
        A.maps = ap_map
        A.plot_map(save_path, pixel='fill', intensity_limits=None,
                   insert_txt=insert_txt, save_format='.png', cmap=sns.color_palette("Greys", as_cmap=True),
                   ax_in=ax, fig_in=fig, alpha=0.5, colorbar=False)

    # Merge single images into a video
    import cv2

    image_folder = Path(results_path / f'series_overlay/{set_names[j]}/')
    images = measurement_set
    video_name = results_path / f'series_overlay/{set_names[j]}.mp4'

    frame = cv2.imread(image_folder / f'{images[0]}_map_fill.png')
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case

    video = cv2.VideoWriter(video_name, fourcc, 1., (width, height))
    for image in images:
        video.write(cv2.imread(image_folder / f'{image}_map_fill.png'))
    cv2.destroyAllWindows()
    video.release()
    # '''


