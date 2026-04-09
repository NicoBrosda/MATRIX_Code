from EvaluationSoftware.main import *
from EvaluationSoftware.readout_modules import ams_channel_assignment_readout

mapping = Path('../../Files/mapping.xlsx')
data = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:])-1 for k in data['direction_2']]

readout, position_parser = lambda x, y: ams_fast_avg(x, y, channel_assignment=channel_assignment), standard_position

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_3D_AIFIRA_2026w12/Nouveau dossier/Linearity?')
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Results_03_26_AIFIRA/Linearity/')

A = Analyzer((1, 128), 0.5, 0.0, readout=readout)
A.scale = 'pico'

'''
dark_path = folder_path
dark = ['dark_current.csv']
A.set_dark_measurement(dark_path, dark)
norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_260325/')
norm_array1 = ['exp7_norm1,9V_']
norm = norm_array1
norm_func = lambda list_of_files, instance, method='least_squares': normalization_from_translated_array_v3(
    list_of_files, instance, method, align_lines=True)
A.normalization(norm_path, norm, normalization_module=norm_func)
A.scale = 'nano'
'''
measurements = array_txt_file_search(os.listdir(folder_path), searchlist=['.csv'], file_suffix='.csv', txt_file=False, blacklist=['.png'])

print(measurements)

measurements = sorted(
    measurements,
    key=lambda s: int(s.replace(".csv", ""))
)

cache = []
for measurement in tqdm(measurements):
    break
    cache.append(A.readout(folder_path / measurement, A)['signal'])

    fig, ax = plt.subplots()
    ax.plot(cache[-1].flatten(), ls='-', c='k')
    ax.set_xlabel('$\\#$ Diode Array')
    ax.set_ylabel("Diode signal (pA)")
    format_save(results_path, f'ArraySignal_{measurement}', save_format='.png',)
cache = np.array(cache)

'''
A.readout = lambda x, y: ams_fast_time_profiles(x, y, channel_assignment=channel_assignment)
for measurement in tqdm(measurements):

    signal = A.readout(folder_path / measurement, A)['signal'][0]
    print(np.shape(signal))

    max_sd = np.argmax(np.mean(signal, axis=1))

    fig, ax = plt.subplots()
    ax.plot([i*0.5 for i in range(len(signal[max_sd]))], signal[max_sd], ls='-', c='k')
    if max_sd != 127:
        ax.plot([i*0.5 for i in range(len(signal[max_sd]))], signal[max_sd+1], ls='-', c='r')
    if max_sd != 0:
        ax.plot([i*0.5 for i in range(len(signal[max_sd]))], signal[max_sd-1], ls='-', c='b')

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Diode signal (pA)')
    ax.set_title(f'Max signal = {np.max(signal[max_sd])}')
    format_save(results_path, f'TimeSignal_{measurement}', save_format='.png',)
'''



