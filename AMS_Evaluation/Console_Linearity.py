import os
import numpy as np
from scipy.optimize import curve_fit

from FitFuncs import linear_function
from read_MATRIX import *

save_path = Path('/Users/nico_brosda/Desktop/iphc_python_misc/Results/linearity/')
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
        signals.append(np.array(signal))

    print(current)
    print(np.shape(signals))
    # Data plotting
    current = np.array(current)
    signals = np.array(signals)
    y_data = np.mean(signals[:, mask[j]], axis=1)
    c = sns.color_palette("tab10")[j]
    # ax.plot(current, signals[:, 33], ls='', marker='x', c='r')
    popt, pcov = curve_fit(linear_function, current, y_data)

    residuals = y_data - linear_function(current, *popt)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    ax.plot(current, y_data, ls='', marker='x', c=c, label=labels[j]+r' - Linear Fit R$^{2}$ of: '+str(round(r_squared, 4)))


    # popt2, pcov2 = curve_fit(linear_function, current, signals[:, 33])
    x = np.linspace(min(current), max(current), 1000)
    ax.plot(x, linear_function(x, *popt), c=c)
    # ax.plot(x, linear_function(x, *popt2), c='r')
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_xlabel('Proton Beam Current [pA]')
    ax.set_ylabel('Measured amplitude')

save_name = 'Linearity_Comparison'
# just_save(save_path=save_path, save_name=save_name, minor_xticks=False)
format_save(save_path=save_path, save_name=save_name)
