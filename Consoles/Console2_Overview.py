from AMS_Evaluation.read_MATRIX import *

folder_path = Path('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/matrix_19062024/')
dark_path = folder_path / 'd2_1n_3s_beam_all_without_diffuser_dark.csv'
# Normalization factor - 5s_flat_calib is from the measurements later that day, where the beam was seemingly more stable
for crit in ['5s_flat_calib_', '500p_center_'][0:1]:
    excluded = []
    print('-' * 100)
    print(crit)
    files_norm = os.listdir(folder_path)
    files_norm = array_txt_file_search(files_norm, blacklist=['.png'], searchlist=[crit],
                                  file_suffix='.csv', txt_file=False)
    start = time.time()
    factor = normalization_new(folder_path, files_norm, excluded, 'y', 'least_squares', dark_path, cache_save=True)
    end = time.time()
    print(end - start)

for crit in np.array(['10s_iphcmatrixcrhea_', '3s_beam_all_without_diffuser_mesures_', '5s_biseau_blanc_topolino_decal7_',
             '5s_biseau_blanc_topolino_nA_', '5s_biseau_blanc_vide_', '5s_biseau2D_vide_nA_', '5s_flat_calib_',
             '5s_flat_mesures', '5s_misc_shapes_', '5s_topolino_thin_', 'trapeze_bragg_0_10s_',
             '10s_neonoff_iphcmatrixcrhea_nA_', '10s_neonoff_iphcmatrixcrhea_suite_', '500p_center_',
             '500p_fullscan_nA_', '500p_fullscan_2_'], dtype=str)[0:1]:  # [list(range(6))+list(range(8, 13))]:
    varied_beam = False
    print('-'*50)
    print(crit)
    paths = ['/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/iphc_python_misc/matrix_27052024/e2_500p_bottom_nA_2.csv',
             '/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/iphc_python_misc/matrix_27052024/e2_500p_nA_2.csv',
             '/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/iphc_python_misc/matrix_27052024/e2_500p_top_nA_2.csv']
    interpret_2Dmap(folder_path, crit, save_path='/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_19062024/',
                    plot=True, varied_beam=varied_beam, contour=True, realistic=True, do_normalization=True,
                    paths_of_norm_files=folder_path, normalization_files=files_norm, path_dark=dark_path)
    continue
    interpret_2Dmap(folder_path, crit, save_path='/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_19062024/',
                    plot=True, varied_beam=varied_beam, contour=False, realistic=True, do_normalization=False,
                    paths_of_norm_files=folder_path, normalization_files=files_norm, path_dark=dark_path)
    interpret_2Dmap(folder_path, crit, save_path='/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_19062024/',
                    plot=True, varied_beam=varied_beam, contour=True, realistic=True, do_normalization=True,
                    paths_of_norm_files=folder_path, normalization_files=files_norm, path_dark=dark_path)