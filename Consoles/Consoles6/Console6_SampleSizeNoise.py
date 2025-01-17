import numpy as np

from EvaluationSoftware.main import *
import cv2
import os

mapping = Path('../../Files/mapping.xlsx')
data = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:])-1 for k in data['direction_2']]
readout, position_parser = lambda x, y: ams_2line_readout(x, y, channel_assignment=channel_assignment), standard_position

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Results_221024/SampleSize/Analyze2Line/')

new_measurements = ['2Line_Beam_', '_GafCompMisc_', '_GafComp200_', '_GafComp400_', '_GafComp40_', '_GafCompLogo_',
                    '_GafCompPEEK_', '_MouseFoot_', '_MouseFoot2_']
live_scan_array1 = [str(round(i+1, 0))+'_live1_' for i in range(9)]

dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')

dark_paths_array1 = ['2Line_DarkVoltageScan_200_ um_0_nA_nA_1.9_x_22.0_y_66.625.csv']

norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
norm_array1 = ['2Line_YScan_']

for k, crit in enumerate(new_measurements[0:]):
    print('-' * 50)
    print(crit)
    print('-' * 50)
    readout = lambda x, y: ams_sample_noise_readout(x, y, channel_assignment=channel_assignment)
    A = Analyzer((1, 128), (0.4, 0.4), (0.1, 0.1), readout=readout)
    dark = dark_paths_array1
    A.set_measurement(folder_path, crit)
    for file in tqdm(A.measurement_files):
        pos = position_parser(file)
        cache = A.readout(file, A)
        cache.update({'position': pos})
        if A.voltage_parser is not None:
            cache.update({'voltage': A.voltage_parser(file)})
        if A.current_parser is not None:
            cache.update({'current': A.current_parser(file)})
        A.measurement_data.append(cache)

    all_data = np.array([i['signal'] for i in A.measurement_data])
    print(np.shape(all_data))

    threshold = 1500  # ski_threshold_otsu(all_data)

    samples = np.shape(all_data)[2]
    no_sig_deviation_from_mean = []
    no_sig_std_data = []
    no_sig = []
    sig_deviation_from_mean = []
    sig_std_data = []
    sig = []

    sample_range = range(2, samples, 1)
    for data in tqdm(all_data):
        for line in data:
            cache_dev = []
            cache_std = []
            mean = np.mean(line)
            for i in sample_range:
                cache_std.append(np.std(line[0:i]))
                cache_dev.append(np.abs(np.mean(line[0:i]) - mean))
            if mean != 0:
                cache_dev, cache_std = np.array(cache_dev)/mean, np.array(cache_std)/mean
            else:
                cache_dev, cache_std = np.array(cache_dev), np.array(cache_std)
            if mean < 1500:
                no_sig_std_data.append(cache_std)
                no_sig_deviation_from_mean.append(cache_dev)
                no_sig.append(np.array(line))
            else:
                sig_std_data.append(cache_std)
                sig_deviation_from_mean.append(cache_dev)
                sig.append(np.array(line))

    # Signal vs. time
    fig, ax = plt.subplots()
    ax.plot(range(1, samples+1, 1), np.mean(sig, axis=0), ls='-', c='r', label='Mean of signal pixel')
    # ax.axhline(np.mean(sig), c='r', ls='--')
    ax.plot(range(1, samples+1, 1), np.mean(no_sig, axis=0), ls='-', c='b', label='Mean of no signal pixel')
    # ax.axhline(np.mean(no_sig), c='b', ls='--')

    ax.set_xlabel(r'Sample $\#$')
    ax.set_ylabel(r'Response at each sample (a.u.)')
    ax.legend()
    format_save(results_path, 'SignalComp_'+crit, legend=False, fig=fig, axes=[ax])

    fig, ax = plt.subplots()
    ax.plot(sample_range[5:], np.mean(sig_std_data, axis=0)[5:], ls='-', c='r', label='Mean of signal pixel')
    ax2 = ax.twinx()
    ax2.plot(sample_range[5:], np.mean(no_sig_std_data, axis=0)[5:], ls='-', c='b', label='Mean of no signal pixel')
    ax.set_xlabel(r'Measurement time (ms)')
    ax.set_ylabel(r'Std of signal pixels / pixel mean ($\%$)', color='red')
    ax2.set_ylabel(r'Std of no signal pixels / pixel mean ($\%$)', color='blue')
    format_save(results_path, 'Std_'+crit, legend=False, fig=fig, axes=[ax])

    fig, ax = plt.subplots()
    ax.plot(sample_range, np.mean(sig_deviation_from_mean, axis=0), ls='-', c='r', label='Mean of signal pixel')
    ax.plot(sample_range, np.mean(no_sig_deviation_from_mean, axis=0), ls='-', c='b', label='Mean of no signal pixel')
    ax.set_xlabel(r'Measurement time (ms)')
    ax.set_ylabel(r'Deviation from pixel mean normed to pixel mean ($\%$)')
    ax.set_ylim(0, 0.03)
    format_save(results_path, 'DevFromMean_'+crit, legend=False, fig=fig, axes=[ax])


