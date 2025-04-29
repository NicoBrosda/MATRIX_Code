import SimpleITK as sitk
import h5py
from EvaluationSoftware.main import *
from EvaluationSoftware.readout_modules import ams_channel_assignment_readout
from EvaluationSoftware.normalization_modules import normalization_from_translated_array
from scipy.constants import e
from scipy.optimize import curve_fit

fast_mode = True
def simulation_Bragg(run_name, diff=200, pixel_size=50/200, mean_range=(20, 30)):
    dat = pd.read_csv(Path(f'../../Files/energies_after_wheel_diffusor{diff}.txt'), header=4, delimiter='\t', decimal='.',
                      names=['pos', 'thickness', 'energy'])
    comp_list = dat['thickness']

    # ----------------------------------------------------------------------------------------------------------------
    # Obtaining the data
    # ----------------------------------------------------------------------------------------------------------------
    data_cache = []
    line_cache = []
    line_std_cache = []
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
        line = np.mean(data[:, int(mean_range[0] / pixel_size):int(mean_range[1] / pixel_size)], axis=1)[::-1]
        line_std = np.mean(data_std[:, int(mean_range[0] / pixel_size):int(mean_range[1] / pixel_size)], axis=1)[::-1]
        data_cache.append(data)
        line_cache.append(line)
        line_std_cache.append(line_std)
    data_max = np.max(line_cache)
    return data_cache, line_cache, line_std_cache, data_max


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
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Results_260325/Bragg/')
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
aperture_measurements = []

# Used Bragg wedge measurements
diff_400um =  ([f'exp{44}_PEEKwedge_P{0}_'] + [f'exp{i+46}_PEEKwedge_P{i+1}_' for i in range(17)] +
               [f'exp{45}_PEEKwedge_P{18}_'])
diff_200um= ([f'exp{i+99}_PEEKWedge_P{i}_' for i in range(18)] + [f'exp{98}_PEEKWedge_P{18}_'] +
             [f'exp{117}_PEEKWedge_P{19}_'])
diff_200um_middle = [f'exp{i+118}_PEEKWedgeMiddle_P{19-i}_' for i in range(8)][::-1]
new_measurements += diff_400um
new_measurements += diff_200um
new_measurements += diff_200um_middle

# Also load round aperture scans for correction purposes
diff_400um_aperture = [f'exp{i+14}_energydiffmap_P{i}_' for i in range(19)]
diff_200um_aperture= [f'exp{i+77}_energyDep_P{i}_' for i in range(19)] + ['exp118_PEEKWedgeMiddle_P19_']
new_measurements += diff_400um_aperture
new_measurements += diff_200um_aperture

data_wheel_200 = pd.read_csv('../../Files/energies_after_wheel_diffusor200.txt', sep='\t', header=4,
                         names=['position', 'thickness', 'energies'])
data_wheel_400 = pd.read_csv('../../Files/energies_after_wheel_diffusor400.txt', sep='\t', header=4,
                         names=['position', 'thickness', 'energies'])

# Cmaps for two diffuser
cmap=sns.color_palette("rocket_r", as_cmap=True)

param_cmap = sns.color_palette("crest_r", as_cmap=True)
param_colormapper_200 = lambda param: color_mapper(param, np.min(data_wheel_200['energies']), np.max(data_wheel_200['energies']))
param_color_200 = lambda param: param_cmap(param_colormapper_200(param))
param_colormapper_400 = lambda param: color_mapper(param, np.min(data_wheel_400['energies']), np.max(data_wheel_400['energies']))
param_color_400 = lambda param: param_cmap(param_colormapper_400(param))

dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_260325/')

dark_paths_array1 = ['exp1_dark_0nA_400um_nA_1.9_x_20.0_y_68.0', 'exp64_darkEnd_0.5nA_400um_nA_1.9_x_20.0_y_68.0',
                     '2exp66_Dark_0.0nA_0um_nA_1.9_x_20.0_y_68.0', '2exp138_DarkEnd_0nA_200um_nA_1.9_x_20.0_y_68.0']

norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_260325/')
norm_array1 = ['exp7_norm1,9V_']
# norm_array1 = ['2exp76_normday2_P0_']

try:
    aperture_400 = np.load(results_stem / 'Fast_Mode/aperture_400.npz', allow_pickle=True)['arr_0']
    aperture_200 = np.load(results_stem / 'Fast_Mode/aperture_200.npz', allow_pickle=True)['arr_0']
    wedge_400 = np.load(results_stem / 'Fast_Mode/wedge_400.npz', allow_pickle=True)['arr_0']
    wedge_200 = np.load(results_stem / 'Fast_Mode/wedge_200.npz', allow_pickle=True)['arr_0']
    wedge_200_middle = np.load(results_stem / 'Fast_Mode/wedge_200_middle.npz', allow_pickle=True)['arr_0']

    A = Analyzer((2, 64), (0.4, 0.4), (0.1, 0.1), readout=readout,
                 diode_offset=[[0, - 0.25], np.zeros(64)], position_parser=position_parser,
                 voltage_parser=voltage_parser, current_parser=current_parser)
except FileNotFoundError as _error:
    print(_error)
    aperture_400 = []
    aperture_200 = []
    wedge_400 = []
    wedge_200 = []
    wedge_200_middle = []

    for k, crit in enumerate(new_measurements[0:]):
        print('-'*50)
        print(crit)
        print('-'*50)

        if k < len(diff_400um):
            diffuser = 400
        elif k < len(diff_400um) + len(diff_200um):
            diffuser = 200
        elif k < len(diff_400um) + len(diff_200um) + len(diff_200um_middle):
            diffuser = 200
        elif k < len(diff_400um) + len(diff_200um) + len(diff_200um_middle) + len(diff_400um_aperture):
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

        # -------------- Save signal map to a cache list --------------
        save_obj = A.maps[0]
        save_obj['wheel_position'] = wheel_position
        if k < len(diff_400um):
            wedge_400.append(save_obj)
        elif k < len(diff_400um) + len(diff_200um):
            wedge_200.append(save_obj)
        elif k < len(diff_400um) + len(diff_200um) + len(diff_200um_middle):
            wedge_200_middle.append(save_obj)
        elif k < len(diff_400um) + len(diff_200um) + len(diff_200um_middle) + len(diff_400um_aperture):
            aperture_400.append(save_obj)
        else:
            aperture_200.append(save_obj)

    if fast_mode:
        np.savez(results_stem / 'Fast_Mode/aperture_400.npz', aperture_400)
        np.savez(results_stem / 'Fast_Mode/aperture_200.npz', aperture_200)
        np.savez(results_stem / 'Fast_Mode/wedge_400.npz', wedge_400)
        np.savez(results_stem / 'Fast_Mode/wedge_200.npz', wedge_200)
        np.savez(results_stem / 'Fast_Mode/wedge_200_middle.npz', wedge_200_middle)

# ---------------------------------------------------------------------------------------------------------------------
# Positioning of PEEK wedge
# ---------------------------------------------------------------------------------------------------------------------
# For 400 um measurement: Bottom line at ≈ 57.25 - validation bottom aperture at 55.54
# Difference = 1.71 mm but in Gaf ≈ 2.4 mm !x!x!
bragg_pos_wedge400 = 57.25
# For 200 um measurement:
# Holes at 70.3, 72.1, ...
# -> Lower edge is 14 or 15.8 mm lower = 56.3
bragg_pos_wedge200 = 56.3
# For 200 um measurement in middle: ≈ 65.4 as end of wedge object -> lower edge of wedge 2.1 mm higher
# This would mean: Lower edge of wedge is at 67.5
# But will be validated with synchronizing max signal to 200 um measurement position
bragg_pos_wedge200_middle = 67.5

# ---------------------------------------------------------------------------------------------------------------------
# Without correction
# ---------------------------------------------------------------------------------------------------------------------
# Extract Bragg curves out of signal maps:
x_range = (16, 18)
signal_cache_200 = []
for data in wedge_200:
    x_data = data['x']
    indices = [np.argmin(np.abs(x_data - x_range[0])), np.argmin(np.abs(x_data - x_range[1]))]
    signal_cache_200.append((data['wheel_position'], data['y'], np.mean(data['z'][:, indices[0]:indices[1]], axis=1)))

x_range = (16, 18)
signal_cache_200_middle = []
for data in wedge_200_middle:
    x_data = data['x']
    indices = [np.argmin(np.abs(x_data - x_range[0])), np.argmin(np.abs(x_data - x_range[1]))]
    signal_cache_200_middle.append(
        (data['wheel_position'], data['y'], np.mean(data['z'][:, indices[0]:indices[1]], axis=1)))

x_range = (16, 18)
signal_cache_400 = []
for data in wedge_400:
    x_data = data['x']
    indices = [np.argmin(np.abs(x_data - x_range[0])), np.argmin(np.abs(x_data - x_range[1]))]
    signal_cache_400.append(
        (data['wheel_position'], data['y'], np.mean(data['z'][:, indices[0]:indices[1]], axis=1)))

# ---------------------------------------------------------------------------------------------------------------------
# Correcting by the aperture characteristic
# ---------------------------------------------------------------------------------------------------------------------
# Dividing the aperture images into signal / no-signal regions (no-signal = mask assigned with 0)

# Assigning the signal regions with a homogenization factor (normed to 1)

# Correcting the wedge measurement by this factor

# ----------------------------------------------------------------------------------------------------------------
# Plot functions
# ----------------------------------------------------------------------------------------------------------------

def plot_maps_and_wedge(save_path, comp_list, map_cache, signal_cache, param_color, shape_position, param_unit='MeV',
                        add_wedge=True, keep_scale=True):
    material_depth = []
    signal_height = []
    signal_pos = []
    for i, obj in enumerate(signal_cache):
        wheel_position = obj[0]
        line = obj[2]
        y_pos = obj[1]
        param = comp_list[i]
        color = param_color(param)
        data_max = np.max(line)
        line_max = y_pos[np.argmax(line)]
        signal_pos.append(line_max)
        signal_height.append(np.max(line))

        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        A.maps[0]['x'], A.maps[0]['y'], A.maps[0]['z'] = map_cache[i]['y'], map_cache[i]['x'], map_cache[i]['z'].T
        A.plot_map(None, pixel='fill', ax_in=ax, fig_in=fig, cmap=cmap)
        ax2.plot(y_pos, line, c=color)
        ax2.set_ylim(0, data_max * 1.2)
        ax2.set_yticklabels([])
        ax.set_xlabel('Position y (mm)')
        ax.set_ylabel('Position x (mm)')
        ax.axvline(line_max, c='m', ls='--')
        ax.axhline(x_range[0], c='k', ls='--')
        ax.axhline(x_range[1], c='k', ls='--')
        axlim_before = ax.get_xlim()

        if add_wedge:
            shape = LineShape([[0, 1e-9], [40, 10]], distance_mode=True)
            shape.print_shape()
            shape.position(shape_position, 0)
            shape.add_to_plot(0.0, 0.5, color='grey', alpha=0.6, zorder=5, edgecolor='k')
            ax.axvline(line_max, c='m', ls='--')
            material_depth.append(shape.calculate_value(line_max))
            if keep_scale:
                ax.set_xlim(axlim_before)
            text = f"{param: .2f}$\\,${param_unit} Max signal at depth {shape.calculate_value(line_max): .2n}$\\,$mm"
        else:
            text = f"{param: .2f}$\\,${param_unit}"
        ax.text(*transform_axis_to_data_coordinates(ax, [0.05, 0.93]), text, fontsize=13,
                c=color, zorder=7, bbox={'facecolor': 'white', 'edgecolor': 'white', 'alpha': 0.9, 'pad': 2})
        format_save(save_path, save_name=f'Map{wheel_position}', save=True, legend=False, fig=fig)
    if add_wedge:
        return material_depth, signal_height, signal_pos
    else:
        return signal_height, signal_pos


def plots_vs_wedge_position(save_path, comp_list, signal_cache, material_depth, signal_height, signal_pos,
                            param_color, shape_position, param_unit='MeV'):
    # ----------------------------------------------------------------------------------------------------------------
    # Plot signal curves comp vs wedge
    # ----------------------------------------------------------------------------------------------------------------

    fig, ax = plt.subplots()
    ax.set_xlabel('Position y (mm)')
    ax.set_ylabel(f'Signal Current ({scale_dict[A.scale][1]}A)')
    for i, obj in enumerate(signal_cache):
        line = obj[2]
        y_pos = obj[1]
        param = comp_list[i]
        color = param_color(param)

        ax.plot(y_pos, line, c=color, zorder=1)
        ax.axvline(signal_pos[i], c=color, ls='--', alpha=0.6, zorder=0)

    shape = LineShape([[0, 1e-9], [40, 10]], distance_mode=True)
    shape.print_shape()
    shape.position(shape_position, 0)
    shape.add_to_plot(0.0, 0.5, color='grey', alpha=0.6, zorder=5, edgecolor='k')

    gradient_arrow(ax, transform_axis_to_data_coordinates(ax, [0.1, 0.925]),
                       transform_axis_to_data_coordinates(ax, [0.1, 0.795]),
                   cmap=param_cmap, lw=10, zorder=5)
    ax.text(*transform_axis_to_data_coordinates(ax, [0.035, 0.94]),
            f'{np.min(comp_list): .2f}$\\,${param_unit}', fontsize=13, c=param_color(np.min(comp_list)),
            zorder=3, bbox={'facecolor': 'w', 'alpha': 0.8, 'pad': 2, 'edgecolor': 'w'})
    ax.text(*transform_axis_to_data_coordinates(ax, [0.025, 0.71]),
            f'{np.max(comp_list): .2f}$\\,${param_unit}', fontsize=13, c=param_color(np.max(comp_list)),
            zorder=3, bbox={'facecolor': 'w', 'alpha': 0.8, 'pad': 2, 'edgecolor': 'w'})

    format_save(save_path, save_name=f'SignalVsWedgeScaled', save=True, legend=False, fig=fig)

    # ----------------------------------------------------------------------------------------------------------------
    # Plot signal curves comp vs wedge
    # ----------------------------------------------------------------------------------------------------------------

    fig, ax = plt.subplots()
    ax.set_xlabel('Position y (mm)')
    ax.set_ylabel(f'Signal Current ({scale_dict[A.scale][1]}A)')
    for i, obj in enumerate(signal_cache):
        line = obj[2]
        y_pos = obj[1]
        param = comp_list[i]
        color = param_color(param)

        ax.plot(y_pos, line, c=color, zorder=1)
        ax.axvline(signal_pos[i], c=color, ls='--', alpha=0.6, zorder=0)
    xlim_before = ax.get_xlim()

    shape = LineShape([[0, 1e-9], [40, 10]], distance_mode=True)
    shape.print_shape()
    shape.position(shape_position, 0)
    shape.add_to_plot(0.0, 0.5, color='grey', alpha=0.6, zorder=5, edgecolor='k')
    ax.set_xlim(xlim_before)

    gradient_arrow(ax, transform_axis_to_data_coordinates(ax, [0.1, 0.925]),
                   transform_axis_to_data_coordinates(ax, [0.1, 0.795]),
                   cmap=param_cmap, lw=10, zorder=5)
    ax.text(*transform_axis_to_data_coordinates(ax, [0.035, 0.94]),
            f'{np.min(comp_list): .2f}$\\,${param_unit}', fontsize=13, c=param_color(np.min(comp_list)),
            zorder=3, bbox={'facecolor': 'w', 'alpha': 0.8, 'pad': 2, 'edgecolor': 'w'})
    ax.text(*transform_axis_to_data_coordinates(ax, [0.025, 0.71]),
            f'{np.max(comp_list): .2f}$\\,${param_unit}', fontsize=13, c=param_color(np.max(comp_list)),
            zorder=3, bbox={'facecolor': 'w', 'alpha': 0.8, 'pad': 2, 'edgecolor': 'w'})

    format_save(save_path, save_name=f'SignalVsWedge', save=True, legend=False, fig=fig)

    # ----------------------------------------------------------------------------------------------------------------
    # Plot Material depth vs energy
    # ----------------------------------------------------------------------------------------------------------------

    fig, ax = plt.subplots()

    ax.set_xlabel('Energy (MeV)')
    ax.set_ylabel(f'Wedge Material Depth (mm)')
    for i, obj in enumerate(signal_cache):
        param = comp_list[i]
        color = param_color(param)
        ax.plot(param, material_depth[i], c=color, marker='x')

    format_save(save_path, save_name=f'MaterialDepthVsWedge', save=True, legend=False, fig=fig)

    # ----------------------------------------------------------------------------------------------------------------
    # Plot signal height + material depth vs energy
    # ----------------------------------------------------------------------------------------------------------------

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.set_xlabel('Energy (MeV)')
    ax.set_ylabel(f'Wedge Material Depth (mm)')
    ax2.set_ylabel(f'Signal Current ({scale_dict[A.scale][1]}A)')

    for i, obj in enumerate(signal_cache):
        param = comp_list[i]
        color = param_color(param)
        if i == 0:
            ax.plot(param, material_depth[i], c='k', marker='x', label='Wedge Material Depth', ls='')
            ax2.plot(param, signal_height[i], c='k', marker='o', label='Signal Current', ls='')
        ax.plot(param, material_depth[i], c=color, marker='x')
        ax2.plot(param, signal_height[i], c=color, marker='o')
    format_save(save_path, save_name=f'MaterialDepthAndSignalHeightVsWedge', save=True, legend=True, fig=fig)

# ----------------------------------------------------------------------------------------------------------------
# Calls for 200 wedge
# ----------------------------------------------------------------------------------------------------------------
# '''
comp_list = data_wheel_200['energies']
param_unit = 'MeV'
param_color = param_color_200
signal_cache = signal_cache_200
map_cache = wedge_200
shape_position = bragg_pos_wedge200
save_path = results_path / 'Wedge200/'

plot_maps_and_wedge(save_path / 'NoWedge/', comp_list, map_cache, signal_cache, param_color,
                    shape_position, param_unit, False, False)
plot_maps_and_wedge(save_path / 'Scaled/', comp_list, map_cache, signal_cache, param_color,
                    shape_position, param_unit, True, False)
material_depth, signal_height, signal_pos = plot_maps_and_wedge(save_path / 'Original/', comp_list,
                                                                map_cache, signal_cache, param_color, shape_position,
                                                                param_unit, True, True)
plots_vs_wedge_position(save_path / 'Results/', comp_list, signal_cache, material_depth, signal_height,
                          signal_pos, param_color, shape_position, param_unit)
# '''

# ----------------------------------------------------------------------------------------------------------------
# Calls for 200 wedge middle
# ----------------------------------------------------------------------------------------------------------------
comp_list = data_wheel_200['energies'].to_numpy()[-len(signal_cache_200_middle):]
print(comp_list)
param_unit = 'MeV'
param_color = param_color_200
signal_cache = signal_cache_200_middle
map_cache = wedge_200_middle
shape_position = bragg_pos_wedge200_middle
save_path = results_path / 'Wedge200Middle/'

plot_maps_and_wedge(save_path / 'NoWedge/', comp_list, map_cache, signal_cache, param_color,
                    shape_position, param_unit, False, False)
plot_maps_and_wedge(save_path / 'Scaled/', comp_list, map_cache, signal_cache, param_color,
                    shape_position, param_unit, True, False)
material_depth, signal_height, signal_pos = plot_maps_and_wedge(save_path / 'Original/', comp_list,
                                                                map_cache, signal_cache, param_color, shape_position,
                                                                param_unit, True, True)
plots_vs_wedge_position(save_path / 'Results/', comp_list, signal_cache, material_depth, signal_height,
                          signal_pos, param_color, shape_position, param_unit)

# ----------------------------------------------------------------------------------------------------------------
# Calls for 400 wedge
# ----------------------------------------------------------------------------------------------------------------
comp_list = data_wheel_400['energies']
param_unit = 'MeV'
param_color = param_color_400
signal_cache = signal_cache_400
map_cache = wedge_400
shape_position = bragg_pos_wedge400
save_path = results_path / 'Wedge400/'

plot_maps_and_wedge(save_path / 'NoWedge/', comp_list, map_cache, signal_cache, param_color,
                    shape_position, param_unit, False, False)
plot_maps_and_wedge(save_path / 'Scaled/', comp_list, map_cache, signal_cache, param_color,
                    shape_position, param_unit, True, False)
material_depth, signal_height, signal_pos = plot_maps_and_wedge(save_path / 'Original/', comp_list,
                                                                map_cache, signal_cache, param_color, shape_position,
                                                                param_unit, True, True)
plots_vs_wedge_position(save_path / 'Results/', comp_list, signal_cache, material_depth, signal_height,
                          signal_pos, param_color, shape_position, param_unit)



