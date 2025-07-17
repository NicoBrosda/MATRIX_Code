import SimpleITK as sitk
import h5py
import matplotlib.pyplot as plt

from EvaluationSoftware.main import *
from EvaluationSoftware.readout_modules import ams_channel_assignment_readout
from EvaluationSoftware.normalization_modules import normalization_from_translated_array
from scipy.constants import e
from scipy.optimize import curve_fit

fast_mode = True
def simulation_response(run_name, comp_list=[], param_info='', gradual_parameter=True, param_unit='MeV', diff=200):
    dat = pd.read_csv(Path(f'../../Files/energies_after_wheel_diffusor{diff}.txt'), header=4, delimiter='\t', decimal='.',
                      names=['pos', 'thickness', 'energy'])
    comp_list = dat['thickness']

    # ----------------------------------------------------------------------------------------------------------------
    # Obtaining the data
    # ----------------------------------------------------------------------------------------------------------------
    data_cache = []
    response_cache = []
    std_cache = []
    energy_cache = []
    energy_std_cache = []
    for i, param in enumerate(comp_list):
        _run_name = f"{run_name}{param}"
        current_path = Path('/Users/nico_brosda/GateSimulation/GATE10/Simulation/output/')
        output_path = current_path / f'{run_name[0:run_name.index("_")]}/'
        output_file = output_path / f"_{_run_name}_dose.mhd"

        # ----------------- Load the Dose / Edep image ----------------------------------
        # img = sitk.ReadImage(str(output_file).replace(".mhd", "_dose.mhd"))
        img = sitk.ReadImage(str(output_file).replace(".mhd", "_edep.mhd"))
        img_std = sitk.ReadImage(str(output_file).replace(".mhd", "_edep_uncertainty.mhd"))
        data = np.array(sitk.GetArrayFromImage(img))[0][::-1, :]
        data_std = np.array(sitk.GetArrayFromImage(img_std))[0][::-1, :]
        response_ind = np.argsort(data.flatten())[-500:]
        response = data.flatten()[response_ind]
        response_std = data_std.flatten()[response_ind].mean()
        data_cache.append(data)
        response_cache.append(response.mean())
        std_cache.append(np.sqrt((response.std()/np.sqrt(len(response)))**2 + response_std**2))

        # ----------------- Load the energy information ----------------------------------
        hdf5_filename = f"{output_path}/{_run_name}.h5"
        try:
            with h5py.File(hdf5_filename, "r") as f:
                group_path = f"{_run_name}/phsp2"
                group = f[group_path]

                ''' Not needed right now
                E_sample = group["E_positive_sample"][...]
                hist_counts = group["E_hist_counts"][...]
                hist_bins = group["E_hist_bins"][...]
                hist_counts_comp = group["E_hist_counts_comp"][...]
                hist_bins_comp = group["E_hist_bins_comp"][...]
                '''
                if "E_stat_median" in group.attrs:
                    E_stat_median = group.attrs["E_stat_median"]
                    E_stat_std = group.attrs["E_stat_std"]
                else:
                    print('Distribution parameters are only calculated out of sampled set!')
                    E_sample = group["E_positive_sample"][...]
                    E_stat_median = np.median(E_sample)
                    E_stat_std = np.std(E_sample)
            energy_cache.append(E_stat_median)
            energy_std_cache.append(E_stat_std)
        except Exception as e:
            print(f"Error loading data from {hdf5_filename}: {e}")
            energy_cache.append(dat['energy'][i])
            energy_std_cache.append(0)
            continue

    return response_cache, std_cache, energy_cache, energy_std_cache

pixel_size = 50/200

def simulation_response2(run_name):
    folder_path = Path('/Users/nico_brosda/GateSimulation/GATE10/Simulation/output/')
    files = os.listdir(folder_path / f'{run_name[0:run_name.index("_")]}/')
    files = array_txt_file_search(files, blacklist=['.mhd', '.raw'], searchlist=[run_name], txt_file=False)
    # ----------------------------------------------------------------------------------------------------------------
    # Obtaining the data
    # ----------------------------------------------------------------------------------------------------------------
    param_list = []
    data_cache = []
    response_cache = []
    std_cache = []
    for i, file in enumerate(files):
        try:
            param = float(file[file.index('_param')+6:])
        except ValueError:
            continue

        # ----------------- Load the Dose / Edep image ----------------------------------
        # img = sitk.ReadImage(str(output_file).replace(".mhd", "_dose.mhd"))
        try:
            _run_name = f"{run_name}{param}"
            output_path = folder_path / f'{run_name[0:run_name.index("_")]}/'
            output_file = output_path / f"_{_run_name}_dose.mhd"

            img = sitk.ReadImage(str(output_file).replace(".mhd", "_edep.mhd"))
            img_std = sitk.ReadImage(str(output_file).replace(".mhd", "_edep_uncertainty.mhd"))
        except RuntimeError:
            param = int(param)
            _run_name = f"{run_name}{param}"
            output_path = folder_path / f'{run_name[0:run_name.index("_")]}/'
            output_file = output_path / f"_{_run_name}_dose.mhd"

            img = sitk.ReadImage(str(output_file).replace(".mhd", "_edep.mhd"))
            img_std = sitk.ReadImage(str(output_file).replace(".mhd", "_edep_uncertainty.mhd"))

        param_list.append(param)

        data = np.array(sitk.GetArrayFromImage(img))[0][::-1, :]
        data_std = np.array(sitk.GetArrayFromImage(img_std))[0][::-1, :]
        '''
        if np.max(data) < 0.1:
            response = 0
            response_std = 0
        else:
            # threshold = threshold_otsu(data)
            # response = np.mean(data[data > threshold])
            # response_std = np.sqrt(np.mean(data_std[data > threshold]) ** 2 + np.std(data[data > threshold]) ** 2)
        '''
        radius_mm = 8
        height, width = data.shape
        center_y, center_x = height // 2, width // 2
        radius_px = int(radius_mm / pixel_size)
        y, x = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        mask = dist_from_center <= radius_px
        response = np.mean(data[mask])
        print(np.mean(data_std[mask]), np.std(data[mask]))
        response_std = np.sqrt(np.mean(data_std[mask]) ** 2 + (np.std(data[mask])/np.sqrt(len(data[mask]))) ** 2)

        data_cache.append(data)
        response_cache.append(response)
        std_cache.append(response_std)

    indices = np.argsort(param_list)
    return np.array(param_list)[indices], np.array(response_cache)[indices], np.array(std_cache)[indices]


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
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Results_260325/EnergyResponse/')
results_stem = Path('/Users/nico_brosda/Cyrce_Messungen/Results_260325/')

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

diff_400um = [f'exp{i+14}_energydiffmap_P{i}_' for i in range(19)]
diff_200um= [f'exp{i+77}_energyDep_P{i}_' for i in range(19)] + ['exp118_PEEKWedgeMiddle_P19_']

new_measurements += diff_400um
new_measurements += diff_200um

data_wheel_200 = pd.read_csv('../../Files/energies_after_wheel_diffusor200.txt', sep='\t', header=4,
                         names=['position', 'thickness', 'energies'])
data_wheel_400 = pd.read_csv('../../Files/energies_after_wheel_diffusor400.txt', sep='\t', header=4,
                         names=['position', 'thickness', 'energies'])

dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_260325/')

dark_paths_array1 = ['exp1_dark_0nA_400um_nA_1.9_x_20.0_y_68.0', 'exp64_darkEnd_0.5nA_400um_nA_1.9_x_20.0_y_68.0',
                     '2exp66_Dark_0.0nA_0um_nA_1.9_x_20.0_y_68.0', '2exp138_DarkEnd_0nA_200um_nA_1.9_x_20.0_y_68.0']

norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_260325/')
norm_array1 = ['exp7_norm1,9V_']
# norm_array1 = ['2exp76_normday2_P0_']

try:
    cache_400 = np.load(results_stem / 'Fast_Mode/cache_400.npy')
    cache_200 = np.load(results_stem / 'Fast_Mode/cache_200.npy')
    A = Analyzer((2, 64), (0.4, 0.4), (0.1, 0.1), readout=readout,
                 diode_offset=[[0, - 0.25], np.zeros(64)], position_parser=position_parser,
                 voltage_parser=voltage_parser, current_parser=current_parser)
except FileNotFoundError as _error:
    print(_error)
    cache_400 = []
    cache_200 = []

    for k, crit in enumerate(new_measurements[0:]):
        print('-'*50)
        print(crit)
        print('-'*50)

        if k < len(diff_400um):
            diffuser = 400
        else:
            diffuser = 200

        wheel_position = int(crit[crit.rindex('_P')+2:crit.rindex('_')])

        A = Analyzer((2, 64), (0.4, 0.4), (0.1, 0.1), readout=readout,
                     diode_offset=[[0, - 0.25], np.zeros(64)], position_parser=position_parser,
                     voltage_parser=voltage_parser, current_parser=current_parser)
        dark = dark_paths_array1
        A.set_measurement(folder_path, crit)
        A.set_dark_measurement(dark_path, dark)
        norm = norm_array1
        norm_func = lambda list_of_files, instance, method='least_squares': normalization_from_translated_array_v3(
            list_of_files, instance, method, align_lines=True)
        A.normalization(norm_path, norm, normalization_module=norm_func)
        A.load_measurement()
        A.create_map(inverse=[True, False])

        # -------------- Save signal level to a cache list --------------
        if 'PEEK' in crit:
            # I will consider the whole image even with the Bragg wedge - because the signal seems to be max below the wedge
            # Otherwise likely range 64-28:24
            signals = A.signal_conversion(np.array([i['signal'][:, 0:] for i in A.measurement_data]).flatten())
            stds = A.signal_conversion(np.array([i['std'][:, 0:] for i in A.measurement_data]).flatten())
        else:
            signals = A.signal_conversion(np.array([i['signal'] for i in A.measurement_data]).flatten())
            stds = A.signal_conversion(np.array([i['std'] for i in A.measurement_data]).flatten())

        signal_indices = np.argsort(signals)[-200:]
        signal_levels = signals[signal_indices]
        std = np.sqrt((np.mean(stds[signal_indices]))**2+(np.std(signal_levels))**2) / np.sqrt(len(signal_levels))

        print('-----------')
        print(std)
        print(np.sqrt(np.mean(stds[signal_indices])**2+(np.std(signal_levels))**2))
        print('-----------')

        signal_level = np.mean(signal_levels)
        if diffuser == 400:
            cache_400.append([wheel_position, signal_level, std])
        elif diffuser == 200:
            cache_200.append([wheel_position, signal_level, std])

        # -------------- Plot 1: Hist of Signal map --------------
        fig, ax = plt.subplots()
        bins = 150
        ax.hist(signals, bins=bins, color='k')
        ax.axvline(signal_level, color='b', label='estimated signal level')
        ax.axvline(signal_levels[0], color='r', label='lower border signal level')

        ax.set_xlabel(f'Signal Current ({scale_dict[A.scale][1]}A)')
        ax.set_ylabel(f'counts per bin ({bins} bins)')
        format_save(results_path / f'Histograms{diffuser}um/', A.name, legend=True)
    if fast_mode:
        np.save(results_stem / 'Fast_Mode/cache_400.npy', cache_400)
        np.save(results_stem / 'Fast_Mode/cache_200.npy', cache_200)


# -------------- Plot 2: Curve signal vs energy for 400 um diffuser--------------
energy_cmap = sns.color_palette("crest_r", as_cmap=True)
energy_colormapper = lambda energy: color_mapper(energy, np.min(data_wheel_400['energies'][0:len(cache_400)]), np.max(data_wheel_400['energies'][0:len(cache_400)]))
energy_color = lambda energy: energy_cmap(energy_colormapper(energy))

sim_res_400, sim_std_400, sim_energy_400, sim_energy_std_400 = simulation_response('EnergyResponse400um2,73Al_param', diff=400)

fig, ax = plt.subplots()
ax.plot(data_wheel_400['energies'][0:len(cache_400)], [i[1] for i in cache_400], c='b', ls='-', marker='', label='Experiment')
for i in cache_400:
    ax.errorbar(data_wheel_400['energies'][i[0]], i[1], i[2], c=energy_color(data_wheel_400['energies'][i[0]]), marker='', capsize=4, markersize=7)

ax2 = ax.twinx()
ax2.plot(sim_energy_400[0:len(cache_400)], sim_res_400[0:len(cache_400)], c='r', ls='--', marker='', label='Simulation', zorder=0)
for i in range(len(sim_res_400[0:len(cache_400)])):
    ax2.errorbar(sim_energy_400[i], sim_res_400[i], sim_std_400[i], c=energy_color(data_wheel_400['energies'][i]),
                marker='', capsize=4, markersize=7)

ax.set_xlabel(f'Proton energy (MeV)')
ax.set_ylabel(f'Signal Current ({scale_dict[A.scale][1]}A)')
ax2.set_ylabel(f'Deposited Energy (MeV)')

ax.set_ylim(0, 1.2*np.max([i[1] for i in cache_400]))
ax2.set_ylim(0, 1.2*np.max(sim_res_400[0:len(cache_400)]))
ax.set_xlim(ax.get_xlim())

gradient_arrow(ax, transform_axis_to_data_coordinates(ax, [0.9, 0.775]),
                       transform_axis_to_data_coordinates(ax, [0.9, 0.645]), cmap=energy_cmap, lw=10, zorder=5)
ax.text(*transform_axis_to_data_coordinates(ax, [0.82, 0.79]),
        f"{np.min(data_wheel_400['energies'][0:len(cache_400)]): .2f}$\\,${'MeV'}", fontsize=13,
        c=energy_color(np.min(data_wheel_400['energies'])), zorder=3)  # , bbox={'facecolor': freq_colour(1033), 'alpha': 0.2, 'pad': 2})
ax.text(*transform_axis_to_data_coordinates(ax, [0.80, 0.56]),
        f"{np.max(data_wheel_400['energies'][0:len(cache_400)]): .2f}$\\,${'MeV'}", fontsize=13,
        c=energy_color(np.max(data_wheel_400['energies'])), zorder=3)  # , bbox={'facecolor': freq_colour(32033), 'alpha': 0.2, 'pad': 2})
format_save(results_path, f'400umResponse', legend=True)

# -------------- Plot 2,5: Curve signal vs energy for 400 um diffuser--------------
energy_cmap = sns.color_palette("crest_r", as_cmap=True)
energy_colormapper = lambda energy: color_mapper(energy, np.min(data_wheel_400['energies'][0:len(cache_400)]), np.max(data_wheel_400['energies'][0:len(cache_400)]))
energy_color = lambda energy: energy_cmap(energy_colormapper(energy))

fig, ax = plt.subplots()
ax.plot(data_wheel_400['energies'][0:len(cache_400)], [i[1] for i in cache_400], c='b', ls='-', marker='', label='Experiment')
for i in cache_400:
    ax.errorbar(data_wheel_400['energies'][i[0]], i[1], i[2], c=energy_color(data_wheel_400['energies'][i[0]]), marker='', capsize=4, markersize=7)

ax.set_xlabel(f'Proton energy (MeV)')
ax.set_ylabel(f'Signal Current ({scale_dict[A.scale][1]}A)')

ax.set_ylim(0, 1.2*np.max([i[1] for i in cache_400]))
ax.set_xlim(ax.get_xlim())

gradient_arrow(ax, transform_axis_to_data_coordinates(ax, [0.9, 0.775]),
                       transform_axis_to_data_coordinates(ax, [0.9, 0.645]), cmap=energy_cmap, lw=10, zorder=5)
ax.text(*transform_axis_to_data_coordinates(ax, [0.82, 0.79]),
        f"{np.min(data_wheel_400['energies'][0:len(cache_400)]): .2f}$\\,${'MeV'}", fontsize=13,
        c=energy_color(np.min(data_wheel_400['energies'])), zorder=3)  # , bbox={'facecolor': freq_colour(1033), 'alpha': 0.2, 'pad': 2})
ax.text(*transform_axis_to_data_coordinates(ax, [0.80, 0.56]),
        f"{np.max(data_wheel_400['energies'][0:len(cache_400)]): .2f}$\\,${'MeV'}", fontsize=13,
        c=energy_color(np.max(data_wheel_400['energies'])), zorder=3)  # , bbox={'facecolor': freq_colour(32033), 'alpha': 0.2, 'pad': 2})
format_save(results_path, f'400umResponse_Exp', legend=True)

# -------------- Plot 3: Curve signal vs energy for 200 um diffuser--------------
energy_cmap = sns.color_palette("crest_r", as_cmap=True)
energy_colormapper = lambda energy: color_mapper(energy, np.min(data_wheel_200['energies'][0:len(cache_200)]), np.max(data_wheel_200['energies'][0:len(cache_200)]))
energy_color = lambda energy: energy_cmap(energy_colormapper(energy))

sim_res_200, sim_std_200, sim_energy_200, sim_energy_std_200 = simulation_response('EnergyResponse200um2,73Al_param', diff=200)
fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.plot(data_wheel_200['energies'][0:len(cache_200)], [i[1] for i in cache_200], c='b', ls='-', marker='', label='Experiment', zorder=0)
ax2.plot(sim_energy_200[0:len(cache_200)], sim_res_200[0:len(cache_200)], c='r', ls='--', marker='', label='Simulation', zorder=0)
for i in cache_200:
    ax.errorbar(data_wheel_200['energies'][i[0]], i[1], i[2], c=energy_color(data_wheel_200['energies'][i[0]]), marker='', capsize=4, markersize=7)
for i in range(len(sim_res_200[0:len(cache_200)])):
    ax2.errorbar(sim_energy_200[i], sim_res_200[i], sim_std_200[i], c=energy_color(data_wheel_200['energies'][i]),
                marker='', capsize=4, markersize=7)

ax.set_xlabel(f'Proton energy (MeV)')
ax.set_ylabel(f'Signal Current ({scale_dict[A.scale][1]}A)')
ax2.set_ylabel(f'Deposited Energy (MeV)')

ax.set_xlim(ax.get_xlim())
ax.set_ylim(0, 1.2*np.max([i[1] for i in cache_200]))
ax2.set_ylim(0, 1.2*np.max(sim_res_200[0:len(cache_200)]))

gradient_arrow(ax, transform_axis_to_data_coordinates(ax, [0.9, 0.775]),
                       transform_axis_to_data_coordinates(ax, [0.9, 0.645]), cmap=energy_cmap, lw=10, zorder=5)
ax.text(*transform_axis_to_data_coordinates(ax, [0.82, 0.79]),
        f"{np.min(data_wheel_200['energies'][0:len(cache_200)]): .2f}$\\,${'MeV'}", fontsize=13,
        c=energy_color(np.min(data_wheel_200['energies'])), zorder=3)  # , bbox={'facecolor': freq_colour(1033), 'alpha': 0.2, 'pad': 2})
ax.text(*transform_axis_to_data_coordinates(ax, [0.80, 0.56]),
        f"{np.max(data_wheel_200['energies'][0:len(cache_200)]): .2f}$\\,${'MeV'}", fontsize=13,
        c=energy_color(np.max(data_wheel_200['energies'])), zorder=3)  # , bbox={'facecolor': freq_colour(32033), 'alpha': 0.2, 'pad': 2})
format_save(results_path , f'200umResponse', legend=True)

# -------------- Plot 3,5: Curve signal vs energy for 200 um diffuser--------------
energy_cmap = sns.color_palette("crest_r", as_cmap=True)
energy_colormapper = lambda energy: color_mapper(energy, np.min(data_wheel_200['energies'][0:len(cache_200)]), np.max(data_wheel_200['energies'][0:len(cache_200)]))
energy_color = lambda energy: energy_cmap(energy_colormapper(energy))

fig, ax = plt.subplots()
ax.plot(data_wheel_200['energies'][0:len(cache_200)], [i[1] for i in cache_200], c='b', ls='-', marker='', label='Experiment', zorder=0)
for i in cache_200:
    ax.errorbar(data_wheel_200['energies'][i[0]], i[1], i[2], c=energy_color(data_wheel_200['energies'][i[0]]), marker='', capsize=4, markersize=7)

ax.set_xlabel(f'Proton energy (MeV)')
ax.set_ylabel(f'Signal Current ({scale_dict[A.scale][1]}A)')

ax.set_xlim(ax.get_xlim())
ax.set_ylim(0, 1.2*np.max([i[1] for i in cache_200]))

gradient_arrow(ax, transform_axis_to_data_coordinates(ax, [0.9, 0.775]),
                       transform_axis_to_data_coordinates(ax, [0.9, 0.645]), cmap=energy_cmap, lw=10, zorder=5)
ax.text(*transform_axis_to_data_coordinates(ax, [0.82, 0.79]),
        f"{np.min(data_wheel_200['energies'][0:len(cache_200)]): .2f}$\\,${'MeV'}", fontsize=13,
        c=energy_color(np.min(data_wheel_200['energies'])), zorder=3)  # , bbox={'facecolor': freq_colour(1033), 'alpha': 0.2, 'pad': 2})
ax.text(*transform_axis_to_data_coordinates(ax, [0.80, 0.56]),
        f"{np.max(data_wheel_200['energies'][0:len(cache_200)]): .2f}$\\,${'MeV'}", fontsize=13,
        c=energy_color(np.max(data_wheel_200['energies'])), zorder=3)  # , bbox={'facecolor': freq_colour(32033), 'alpha': 0.2, 'pad': 2})
format_save(results_path , f'200umResponse_Exp', legend=True)

# ----------------------------------------------------------------------------------------------------------------
# Calculate normed responses (normed to incoming protons)
#----------------------------------------------------------------------------------------------------------------
rescale_sim = 1e3
scale_sim = 'k'
simn = 1e7 / rescale_sim / 5026.548245743669 / 4.965
sim_res_200, sim_res_400, sim_std_200, sim_std_400 = (np.array(sim_res_200[:len(cache_200)]) / simn,
                                                      np.array(sim_res_400[:len(cache_400)]) / simn,
                                                      np.array(sim_std_200[:len(cache_200)]) / simn,
                                                      np.array(sim_std_400[:len(cache_400)]) / simn)

additional_scale = 1/e * 1e-18 * 4417.864669110646
rescale_current = 1e6 * additional_scale
scale_current = f'$\\#$electrons'
currents_400 = np.array([887, 888, 885, 880, 876, 872, 884, 880, 876, 871, 888, 887, 884, 881, 881, 877, 882, 880, 879]) * 1e-12 / e / rescale_current
currents_200 = np.array([1.73, 1.72, 1.72, 1.70, 1.71, 1.70, 1.72, 1.71, 1.70, 1.72, 1.72, 1.71, 1.70, 1.72, 1.72, 1.71, 1.70, 1.69, 1.69, 1.76]) *1e-9 / e /rescale_current

normed_200, std_200 = np.array([i[1] for i in cache_200]) / currents_200, np.array([i[2] for i in cache_200]) / currents_200
normed_400, std_400 = np.array([i[1] for i in cache_400]) / currents_400, np.array([i[2] for i in cache_400]) / currents_400

#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
sim_energy, sim_res, sim_std = simulation_response2('EnergyVariation1e6_param')

rescale_sim = 1e3
scale_sim = 'k'
simn = 1e6 / rescale_sim / 5026.548245743669
sim_res, sim_std = sim_res / simn, sim_std / simn

sim_res_400 = np.interp(data_wheel_400['energies'][0:len(cache_400)], sim_energy, sim_res)
sim_std_400 = np.interp(data_wheel_400['energies'][0:len(cache_400)], sim_energy, sim_std)
sim_res_200 = np.interp(data_wheel_200['energies'][0:len(cache_200)], sim_energy, sim_res)
sim_std_200 = np.interp(data_wheel_200['energies'][0:len(cache_200)], sim_energy, sim_std)

#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------

# -------------- Plot 4: Normed signal vs energy for 400 um diffuser --------------
energy_cmap = sns.color_palette("crest_r", as_cmap=True)
energy_colormapper = lambda energy: color_mapper(energy, np.min(data_wheel_400['energies'][0:len(cache_400)]), np.max(data_wheel_400['energies'][0:len(cache_400)]))
energy_color = lambda energy: energy_cmap(energy_colormapper(energy))

fig, ax = plt.subplots()
ax.plot(data_wheel_400['energies'][0:len(cache_400)], normed_400, c='b', ls='-', marker='', label='Experiment')
for i, object in enumerate(cache_400):
    ax.errorbar(data_wheel_400['energies'][i], normed_400[i], std_400[i], c=energy_color(data_wheel_400['energies'][i]), marker='', capsize=4, markersize=7)

ax2 = ax.twinx()
ax2.plot(sim_energy_400[0:len(cache_400)], sim_res_400[0:len(cache_400)], c='r', ls='--', marker='', label='Simulation', zorder=0)
for i in range(len(sim_res_400[0:len(cache_400)])):
    ax2.errorbar(sim_energy_400[i], sim_res_400[i], sim_std_400[i], c=energy_color(data_wheel_400['energies'][i]),
                marker='', capsize=4, markersize=7)

ax.set_xlabel(f'Proton energy (MeV)')
ax.set_ylabel(f'Signal Current per primary ({scale_current})')
ax2.set_ylabel(f'Deposited Energy per primary ({scale_sim}eV)')


ax.set_xlim(ax.get_xlim())
ax.set_ylim(0, 1.2*np.max(normed_400))
ax2.set_ylim(0, 1.2*np.max(sim_res_400[0:len(cache_400)]))

gradient_arrow(ax, transform_axis_to_data_coordinates(ax, [0.9, 0.775]),
                       transform_axis_to_data_coordinates(ax, [0.9, 0.645]), cmap=energy_cmap, lw=10, zorder=5)
ax.text(*transform_axis_to_data_coordinates(ax, [0.82, 0.79]),
        f"{np.min(data_wheel_400['energies'][0:len(cache_400)]): .2f}$\\,${'MeV'}", fontsize=13,
        c=energy_color(np.min(data_wheel_400['energies'])), zorder=3)  # , bbox={'facecolor': freq_colour(1033), 'alpha': 0.2, 'pad': 2})
ax.text(*transform_axis_to_data_coordinates(ax, [0.80, 0.56]),
        f"{np.max(data_wheel_400['energies'][0:len(cache_400)]): .2f}$\\,${'MeV'}", fontsize=13,
        c=energy_color(np.max(data_wheel_400['energies'])), zorder=3)  # , bbox={'facecolor': freq_colour(32033), 'alpha': 0.2, 'pad': 2})
format_save(results_path, f'Normed400umResponse', legend=True)


# -------------- Plot 5 : Normed signal vs energy for 200 um diffuser --------------
energy_cmap = sns.color_palette("crest_r", as_cmap=True)
energy_colormapper = lambda energy: color_mapper(energy, np.min(data_wheel_200['energies'][0:len(cache_200)]), np.max(data_wheel_200['energies'][0:len(cache_200)]))
energy_color = lambda energy: energy_cmap(energy_colormapper(energy))

fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.plot(data_wheel_200['energies'][0:len(cache_200)], normed_200, c='b', ls='-', marker='', label='Experiment', zorder=0)
ax2.plot(sim_energy_200[0:len(cache_200)], sim_res_200[0:len(cache_200)], c='r', ls='--', marker='', label='Simulation', zorder=0)
for i, object in enumerate(cache_200):
    ax.errorbar(data_wheel_200['energies'][i], normed_200[i], std_200[i], c=energy_color(data_wheel_200['energies'][i]), marker='', capsize=4, markersize=7)
for i in range(len(sim_res_200[0:len(cache_200)])):
    ax2.errorbar(sim_energy_200[i], sim_res_200[i], sim_std_200[i], c=energy_color(data_wheel_200['energies'][i]),
                marker='', capsize=4, markersize=7)

ax.set_xlabel(f'Proton energy (MeV)')
ax.set_ylabel(f'Signal Current per primary ({scale_current})')
ax2.set_ylabel(f'Deposited Energy per primary ({scale_sim}eV)')

ax.set_xlim(ax.get_xlim())
ax.set_ylim(0, 1.2*np.max(normed_200))
ax2.set_ylim(0, 1.2*np.max(sim_res_200[0:len(cache_200)]))

gradient_arrow(ax, transform_axis_to_data_coordinates(ax, [0.9, 0.775]),
                       transform_axis_to_data_coordinates(ax, [0.9, 0.645]), cmap=energy_cmap, lw=10, zorder=5)
ax.text(*transform_axis_to_data_coordinates(ax, [0.82, 0.79]),
        f"{np.min(data_wheel_200['energies'][0:len(cache_200)]): .2f}$\\,${'MeV'}", fontsize=13,
        c=energy_color(np.min(data_wheel_200['energies'])), zorder=3)  # , bbox={'facecolor': freq_colour(1033), 'alpha': 0.2, 'pad': 2})
ax.text(*transform_axis_to_data_coordinates(ax, [0.80, 0.56]),
        f"{np.max(data_wheel_200['energies'][0:len(cache_200)]): .2f}$\\,${'MeV'}", fontsize=13,
        c=energy_color(np.max(data_wheel_200['energies'])), zorder=3)  # , bbox={'facecolor': freq_colour(32033), 'alpha': 0.2, 'pad': 2})
format_save(results_path , f'Normed200umResponse', legend=True)

# ----------------------------------------------------------------------------------------------------------------

# ---------------- Plot 6: Signal current vs deposited energy for 400 um diffuser --------------------
energy_cmap = sns.color_palette("crest_r", as_cmap=True)
energy_colormapper = lambda energy: color_mapper(energy, np.min(data_wheel_400['energies'][0:len(cache_400)]), np.max(data_wheel_400['energies'][0:len(cache_400)]))
energy_color = lambda energy: energy_cmap(energy_colormapper(energy))

fig, ax = plt.subplots()

ax.plot(sim_res_400[0:len(cache_400)], normed_400, c='k', ls='-', marker='')
for i, object in enumerate(cache_400):
    ax.errorbar(sim_res_400[i], normed_400[i], std_400[i], sim_std_400[i], c=energy_color(data_wheel_400['energies'][i]), marker='', capsize=4, markersize=7, alpha=0.7)

ax.set_xlabel(f'Deposited Energy per primary ({scale_sim}eV)')
ax.set_ylabel(f'Signal Current per primary ({scale_current})')

ax.set_xlim(ax.get_xlim())
ax.set_ylim(ax.get_ylim())

gradient_arrow(ax, transform_axis_to_data_coordinates(ax, [0.9, 0.265]),
                       transform_axis_to_data_coordinates(ax, [0.9, 0.135]), cmap=energy_cmap, lw=10, zorder=5)
ax.text(*transform_axis_to_data_coordinates(ax, [0.82, 0.28]),
        f"{np.min(data_wheel_400['energies'][0:len(cache_400)]): .2f}$\\,${'MeV'}", fontsize=13,
        c=energy_color(np.min(data_wheel_400['energies'])), zorder=3)  # , bbox={'facecolor': freq_colour(1033), 'alpha': 0.2, 'pad': 2})
ax.text(*transform_axis_to_data_coordinates(ax, [0.80, 0.05]),
        f"{np.max(data_wheel_400['energies'][0:len(cache_400)]): .2f}$\\,${'MeV'}", fontsize=13,
        c=energy_color(np.max(data_wheel_400['energies'])), zorder=3)  # , bbox={'facecolor': freq_colour(32033), 'alpha': 0.2, 'pad': 2})
format_save(results_path, f'EnergyCalib400', legend=False)

# ---------------- Plot 7: Signal current vs deposited energy for 200 um diffuser --------------------
energy_cmap = sns.color_palette("crest_r", as_cmap=True)
energy_colormapper = lambda energy: color_mapper(energy, np.min(data_wheel_200['energies'][0:len(cache_200)]), np.max(data_wheel_200['energies'][0:len(cache_200)]))
energy_color = lambda energy: energy_cmap(energy_colormapper(energy))

fig, ax = plt.subplots()

ax.plot(sim_res_200[0:len(cache_200)], normed_200, c='k', ls='-', marker='')
for i, object in enumerate(cache_200):
    ax.errorbar(sim_res_200[i], normed_200[i], std_200[i], sim_std_200[i], c=energy_color(data_wheel_200['energies'][i]), marker='', capsize=4, markersize=7, alpha=0.7)

ax.set_xlabel(f'Deposited Energy per primary ({scale_sim}eV)')
ax.set_ylabel(f'Signal Current per primary ({scale_current})')

ax.set_xlim(ax.get_xlim())
ax.set_ylim(ax.get_ylim())

gradient_arrow(ax, transform_axis_to_data_coordinates(ax, [0.9, 0.265]),
                       transform_axis_to_data_coordinates(ax, [0.9, 0.135]), cmap=energy_cmap, lw=10, zorder=5)
ax.text(*transform_axis_to_data_coordinates(ax, [0.82, 0.28]),
        f"{np.min(data_wheel_200['energies'][0:len(cache_200)]): .2f}$\\,${'MeV'}", fontsize=13,
        c=energy_color(np.min(data_wheel_200['energies'])), zorder=3)  # , bbox={'facecolor': freq_colour(1033), 'alpha': 0.2, 'pad': 2})
ax.text(*transform_axis_to_data_coordinates(ax, [0.80, 0.05]),
        f"{np.max(data_wheel_200['energies'][0:len(cache_200)]): .2f}$\\,${'MeV'}", fontsize=13,
        c=energy_color(np.max(data_wheel_200['energies'])), zorder=3)  # , bbox={'facecolor': freq_colour(32033), 'alpha': 0.2, 'pad': 2})

format_save(results_path , f'EnergyCalib200', legend=False)

# ----------------------------------------------------------------------------------------------------------------
# How good is this resembled by a proportional relation?
def proportional(x, a):
    return a*x

popt_400, pcov_400 = curve_fit(proportional, sim_res_400, normed_400, sigma=std_400)
residuals = normed_400 - proportional(sim_res_400, *popt_400)
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((normed_400 - np.mean(normed_400)) ** 2)
if ss_tot == 0:
    r_squared_400 = 0
else:
    r_squared_400 = 1 - (ss_res / ss_tot)
print(r_squared_400)
print(popt_400, np.sqrt(np.diag(pcov_400)))

popt_200, pcov_200 = curve_fit(proportional, sim_res_200, normed_200, sigma=std_200)
residuals = normed_200 - proportional(sim_res_200, *popt_200)
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((normed_200 - np.mean(normed_200)) ** 2)
if ss_tot == 0:
    r_squared_200 = 0
else:
    r_squared_200 = 1 - (ss_res / ss_tot)
print(r_squared_200)
print(popt_200, np.sqrt(np.diag(pcov_200)))

# ---------------- Plot 8: Signal current vs deposited energy for 400 um diffuser --------------------
energy_cmap = sns.color_palette("crest_r", as_cmap=True)
energy_colormapper = lambda energy: color_mapper(energy, np.min(data_wheel_400['energies'][0:len(cache_400)]), np.max(data_wheel_400['energies'][0:len(cache_400)]))
energy_color = lambda energy: energy_cmap(energy_colormapper(energy))

fig, ax = plt.subplots()

ax.plot(sim_res_400[0:len(cache_400)], normed_400, c='k', ls='-', marker='')
ax.plot(sim_res_400[0:len(cache_400)], proportional(sim_res_400[0:len(cache_400)], *popt_400), c='r', ls='--')
for i, object in enumerate(cache_400):
    ax.errorbar(sim_res_400[i], normed_400[i], std_400[i], sim_std_400[i], c=energy_color(data_wheel_400['energies'][i]), marker='', capsize=4, markersize=7, zorder=-1, alpha=0.7)

ax.set_xlabel(f'Deposited Energy per primary ({scale_sim}eV)')
ax.set_ylabel(f'Signal Current per primary ({scale_current})')

ax.set_xlim(ax.get_xlim())
ax.set_ylim(ax.get_ylim())

ax.text(*transform_axis_to_data_coordinates(ax, [0.04, 0.93]), f'Proportional law $\\mathrm{"{R}"}^2$ = {r_squared_400:.4f}', fontsize=12)
ax.text(*transform_axis_to_data_coordinates(ax, [0.04, 0.88]), f'Effectivity C = {popt_400[0]:.3f}$\\,${scale_current}/{scale_sim}eV', fontsize=12)

gradient_arrow(ax, transform_axis_to_data_coordinates(ax, [0.9, 0.265]),
                       transform_axis_to_data_coordinates(ax, [0.9, 0.135]), cmap=energy_cmap, lw=10, zorder=5)
ax.text(*transform_axis_to_data_coordinates(ax, [0.82, 0.28]),
        f"{np.min(data_wheel_400['energies'][0:len(cache_400)]): .2f}$\\,${'MeV'}", fontsize=13,
        c=energy_color(np.min(data_wheel_400['energies'])), zorder=3)  # , bbox={'facecolor': freq_colour(1033), 'alpha': 0.2, 'pad': 2})
ax.text(*transform_axis_to_data_coordinates(ax, [0.80, 0.05]),
        f"{np.max(data_wheel_400['energies'][0:len(cache_400)]): .2f}$\\,${'MeV'}", fontsize=13,
        c=energy_color(np.max(data_wheel_400['energies'])), zorder=3)  # , bbox={'facecolor': freq_colour(32033), 'alpha': 0.2, 'pad': 2})
format_save(results_path, f'EnergyCalib400_Fit', legend=False)

# ---------------- Plot 9: Signal current vs deposited energy for 200 um diffuser --------------------
energy_cmap = sns.color_palette("crest_r", as_cmap=True)
energy_colormapper = lambda energy: color_mapper(energy, np.min(data_wheel_200['energies'][0:len(cache_200)]), np.max(data_wheel_200['energies'][0:len(cache_200)]))
energy_color = lambda energy: energy_cmap(energy_colormapper(energy))

fig, ax = plt.subplots()

ax.plot(sim_res_200[0:len(cache_200)], normed_200, c='k', ls='-', marker='')
ax.plot(sim_res_200[0:len(cache_200)], proportional(sim_res_200[0:len(cache_200)], *popt_200), c='r', ls='--')
for i, object in enumerate(cache_200):
    ax.errorbar(sim_res_200[i], normed_200[i], std_200[i], sim_std_200[i], c=energy_color(data_wheel_200['energies'][i]), marker='', capsize=4, markersize=7, zorder=-1, alpha=0.7)

ax.set_xlabel(f'Deposited Energy per primary ({scale_sim}eV)')
ax.set_ylabel(f'Signal Current per primary ({scale_current})')

ax.set_xlim(ax.get_xlim())
ax.set_ylim(ax.get_ylim())

ax.text(*transform_axis_to_data_coordinates(ax, [0.04, 0.93]), f'Proportional law $\\mathrm{"{R}"}^2$ = {r_squared_200:.4f}', fontsize=12)
ax.text(*transform_axis_to_data_coordinates(ax, [0.04, 0.88]), f'Effectivity C = {popt_200[0]:.3f}$\\,${scale_current}/{scale_sim}eV', fontsize=12)

gradient_arrow(ax, transform_axis_to_data_coordinates(ax, [0.9, 0.265]),
                       transform_axis_to_data_coordinates(ax, [0.9, 0.135]), cmap=energy_cmap, lw=10, zorder=5)
ax.text(*transform_axis_to_data_coordinates(ax, [0.82, 0.28]),
        f"{np.min(data_wheel_200['energies'][0:len(cache_200)]): .2f}$\\,${'MeV'}", fontsize=13,
        c=energy_color(np.min(data_wheel_200['energies'])), zorder=3)  # , bbox={'facecolor': freq_colour(1033), 'alpha': 0.2, 'pad': 2})
ax.text(*transform_axis_to_data_coordinates(ax, [0.80, 0.05]),
        f"{np.max(data_wheel_200['energies'][0:len(cache_200)]): .2f}$\\,${'MeV'}", fontsize=13,
        c=energy_color(np.max(data_wheel_200['energies'])), zorder=3)  # , bbox={'facecolor': freq_colour(32033), 'alpha': 0.2, 'pad': 2})

format_save(results_path , f'EnergyCalib200_Fit', legend=False)