from read_MATRIX import *

folder_path = Path('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/matrix_19062024/')
for crit in ['dark', 'd2_1.45nA__10s_iphcmatrixcrhea_nA_2.0_x_9.5_y_85.0.csv']:
    excluded = []
    print('-'*100)
    print(crit)
    files = os.listdir(folder_path)
    files = array_txt_file_search(files, blacklist=['.png'], searchlist=[crit],
                                  file_suffix='.csv', txt_file=False)
    print(files)
    for file in files:
        data = read(folder_path / file)
        signal = read_channels(data, excluded_channel=excluded, varied_beam=False, path_dark=[folder_path / 'd2_1n_3s_beam_all_without_diffuser_dark.csv'])
        signal[60] /= 2000

        fig, ax = plt.subplots()
        ax.plot(signal)
        plt.show()
