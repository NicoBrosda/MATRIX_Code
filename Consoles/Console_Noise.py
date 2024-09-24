from AMS_Evaluation.read_MATRIX import *

save_path = Path('/Users/nico_brosda/Desktop/iphc_python_misc/Results/noise/')
# ---------------------------------------------------------------------------------------------------------------------
# Linearity measurements sample e2

# Find the corresponding files:
folder_path_e2 = Path('/Users/nico_brosda/Desktop/iphc_python_misc/matrix_27052024/')
files_e2 = os.listdir(folder_path_e2)
files_e2 = array_txt_file_search(files_e2, blacklist=['scan', '.png', 'top_', 'bottom_'], searchlist=['p_nA'],
                              file_suffix='.csv', txt_file=False)
print(files_e2)
print(len(files_e2))

# Find the corresponding files:
folder_path_e1 = Path('/Users/nico_brosda/Desktop/iphc_python_misc/matrix_27052024/d2/')
files_e1 = os.listdir(folder_path_e1)
files_e1 = array_txt_file_search(files_e1, blacklist=['scan', '.png', 'top_', 'bottom_'], searchlist=['long_nA_2'],
                              file_suffix='.csv', txt_file=False)
print(files_e1)
print(len(files_e1))

# Find the corresponding files:
folder_path_d2 = Path('/Users/nico_brosda/Desktop/iphc_python_misc/matrix_27052024/e1/')
files_d2 = os.listdir(folder_path_d2)
files_d2 = array_txt_file_search(files_d2, blacklist=['scan', '.png', 'top_', 'bottom_'], searchlist=['lin_nA_2'],
                              file_suffix='.csv', txt_file=False)
print(files_d2)
print(len(files_d2))

mask_e1 = list(range(6, 25, 1)) + list(range(27, 48, 1))
mask_e2 = list(range(10, 36, 1)) + list(range(37, 39, 1)) + list(range(41, 59, 1))
mask_d2 = list(range(7, 28, 1)) + list(range(29, 51, 1)) + list(range(52, 53, 1))

excluded = [[25, 26], [38], [28, 51]]
mask = [mask_e1, mask_e2, mask_d2]
folder_path = [folder_path_e1, folder_path_e2, folder_path_d2]
labels = [r'Sample e1', r'Sample e2', r'Sample d2']
fig, ax = plt.subplots()

for j, files in enumerate([files_e1, files_e2, files_d2]):
    print(labels[j])
    # Parse the current into a float
    current = []
    signals = []
    noise = []
    noise_dark = []
    for file in files:
        i = 1
        cur = None
        while True:
            try:
                index = file.index('p_')
                cur = float(file[(index - i):index])
            except ValueError:
                break
            i += 1
        current.append(cur)
        if cur is None:
            continue

        data = read(folder_path[j] / file)
        signal = read_channels(data, excluded_channel=excluded[j])
        signal2, signal_std, dark, dark_std, thresh = read_channels(data, excluded_channel=excluded[j], advanced_output=True)
        signals.append(np.array(signal))
        noise.append(np.array(signal_std))
        noise_dark.append(np.array(dark_std))


    print(current)
    print(np.shape(signals))
    # Data plotting
    current = np.array(current)
    signals = np.array(signals)
    noise = np.array(noise)
    noise_dark = np.array(noise_dark)
    c = sns.color_palette("bright")[j]
    c2 = sns.color_palette("dark")[j]
    markers = ['x', '+', 'o', '^', 'v', '*']
    for k, i in enumerate([30, 32, 34, 36, 38, 40]):
        ax.plot(signals[:, i], noise[:, i], label=labels[j]+'-Channel'+str(i), ls='', marker=markers[k], c=c)
        ax.plot(signals[:, i], noise_dark[:, i], label=labels[j] + '-Channel' + str(i), ls='', marker=markers[k], c=c2)

ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
plt.show()
