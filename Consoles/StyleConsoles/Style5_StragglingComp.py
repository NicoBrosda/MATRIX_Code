from copy import deepcopy

from EvaluationSoftware.main import *
from EvaluationSoftware.readout_modules import ams_channel_assignment_readout
from EvaluationSoftware.normalization_modules import normalization_from_translated_array
import SimpleITK as sitk


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
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Style/StragglingComp/')

new_measurements = []


new_measurements += [f'exp{126}_Round8mm_5mm_P{0}_', f'exp{127}_Round8mm_5mm_P{12}_']
new_measurements += [f'exp{130}_Round8mm_10mm_P{0}_', f'exp{131}_Round8mm_10mm_P{12}_']
new_measurements += [f'exp{133}_Round8mm_20mm_P{0}_', f'exp{132}_Round8mm_20mm_P{12}_']
new_measurements += [f'exp{136}_Round8mm_40mm_P{0}_', f'exp{137}_Round8mm_40mm_P{12}_']

dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_260325/')

dark_paths_array1 = ['exp1_dark_0nA_400um_nA_1.9_x_20.0_y_68.0', 'exp64_darkEnd_0.5nA_400um_nA_1.9_x_20.0_y_68.0',
                     '2exp66_Dark_0.0nA_0um_nA_1.9_x_20.0_y_68.0', '2exp138_DarkEnd_0nA_200um_nA_1.9_x_20.0_y_68.0']

norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_260325/')
norm_array1 = ['exp7_norm1,9V_']
# norm_array1 = ['2exp76_normday2_P0_']

cache_round0 = []
cache_round12 = []
cache_misc = []
cache_PEEK = []
for k, crit in enumerate(new_measurements[0:]):
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
    intensity_limits = [0, np.max(A.maps[0]['z'])]
    txt_posi = [0.03, 0.93]
    if 'Misc' in crit:
        cache_misc.append([crit, A.maps])
    elif 'PEEK' in crit or 'LargerGap' in crit or 'Distance' in crit:
        cache_PEEK.append([crit, A.maps])
    elif 'P0' in crit:
        cache_round0.append([crit, A.maps])
    else :
        cache_round12.append([crit, A.maps])

'''
intensity_limits = [0, np.max([np.max(i[1][0]['z']) for i in cache_PEEK])]
A.maps = [i[1][0] for i in cache_PEEK]
A.rescale_maps()
for k, map in enumerate(A.maps):
    cache_PEEK[k][1] = [map]

for k, obj in enumerate(cache_PEEK):
    A.maps = obj[1]
    A.name = obj[0]
    crit = A.name
    if not 'Distance' in crit:
        if 'PEEK' in crit:
            crit = f'{"_Distance5mm"}{crit[crit.rindex("_P"):]}'
        else:
            crit = f'{"_Distance10mm"}{crit[crit.rindex("_P"):]}'

    insert_txt = [txt_posi, crit[crit.index('_')+1:-1], 15]
    A.plot_map(results_path / 'straggling/', pixel='fill', intensity_limits=intensity_limits, insert_txt=insert_txt)

intensity_limits = [0, np.max([np.max(i[1][0]['z']) for i in cache_round0])]
A.maps = [i[1][0] for i in cache_round0]
A.rescale_maps()
for k, map in enumerate(A.maps):
    cache_round0[k][1] = [map]

for k, obj in enumerate(cache_round0):
    A.maps = obj[1]
    A.name = obj[0]
    crit = obj[0]
    insert_txt = [txt_posi, crit[crit.index('_')+1:-1], 15]
    A.plot_map(results_path / 'straggling/', pixel='fill', intensity_limits=intensity_limits, insert_txt=insert_txt)

intensity_limits = [0, np.max([np.max(i[1][0]['z']) for i in cache_round12])]
A.maps = [i[1][0] for i in cache_round12]
A.rescale_maps()
for k, map in enumerate(A.maps):
    cache_round12[k][1] = [map]

for k, obj in enumerate(cache_round12):
    A.maps = obj[1]
    A.name = obj[0]
    crit = obj[0]
    insert_txt = [txt_posi, crit[crit.index('_') + 1:-1], 15]
    A.plot_map(results_path / 'straggling/', pixel='fill', intensity_limits=intensity_limits, insert_txt=insert_txt)

intensity_limits = [0, np.max([np.max(i[1][0]['z']) for i in cache_misc])]
A.maps = [i[1][0] for i in cache_misc]
A.rescale_maps()
for k, map in enumerate(A.maps):
    cache_misc[k][1] = [map]

for obj in cache_misc:
    A.maps = obj[1]
    A.name = obj[0]
    crit = obj[0]
    insert_txt = [txt_posi, crit[crit.index('_')+1:-1], 15]
    A.plot_map(results_path / 'straggling/', pixel='fill', intensity_limits=intensity_limits, insert_txt=insert_txt)
'''

def dose_plot(run_name, pixel_size=50/200):
    current_path = pathlib.Path(__file__).parent.resolve()
    output_path = current_path / f"output/{run_name[0:run_name.index('_')]}/"
    output_file = output_path / f"_{run_name}_dose.mhd"

    # img = sitk.ReadImage(str(output_file).replace(".mhd", "_dose.mhd"))
    img = sitk.ReadImage(str(output_file).replace(".mhd", "_edep.mhd"))
    data = np.array(sitk.GetArrayFromImage(img))[0][::-1, :]

    fig, ax = plt.subplots()
    extent = (-np.shape(data)[1] * pixel_size / 2, np.shape(data)[1] * pixel_size / 2, - np.shape(data)[0] * pixel_size / 2, np.shape(data)[0] * pixel_size / 2)
    color_map = ax.imshow(data, vmin=0, vmax=np.max(data), extent=extent, cmap=cmap)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=np.max(data))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map.cmap)
    sm.set_array([])
    bar = fig.colorbar(sm, ax=ax, extend='max')
    ax.set_xlabel('Position x (mm)')
    ax.set_ylabel('Position y (mm)')
    bar.set_label('Deposited energy (MeV)')
    name = f'Ideal_DoseMap_'
    format_save(output_path / run_name, save_name=name, save=True, legend=False, fig=fig)

# Simulation maps:
pixel_size=50/200
run_phrase_P0 = "5e6StragglingSmallerBeam8mmApertureVarAirGap"
run_phrase_P12 = "5e7StragglingSmallerBeam8mmApertureVarAirGapP12"
intensity_limits0 = [0, np.max([np.max(i[1][0]['z']) for i in cache_round0])]
intensity_limits12 = [0, np.max([np.max(i[1][0]['z']) for i in cache_round12])]

cache_sim12 = []
cache_sim0 = []
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black", "red", "yellow"])

for k, param in enumerate([5, 10, 20, 40]):
    run_name = run_phrase_P0 + f"_param{param}"
    output_path = Path("/Users/nico_brosda/GateSimulation/GATE10/Simulation/output/") / run_phrase_P0
    output_file = output_path / f"_{run_name}_dose.mhd"
    img = sitk.ReadImage(str(output_file).replace(".mhd", "_edep.mhd"))
    data = np.array(sitk.GetArrayFromImage(img))[0][::-1, :]
    cache_sim0.append(data)

    run_name = run_phrase_P12 + f"_param{param}"
    output_path = Path("/Users/nico_brosda/GateSimulation/GATE10/Simulation/output/") / run_phrase_P12
    output_file = output_path / f"_{run_name}_dose.mhd"
    img = sitk.ReadImage(str(output_file).replace(".mhd", "_edep.mhd"))
    data = np.array(sitk.GetArrayFromImage(img))[0][::-1, :]
    cache_sim12.append(data)

intensity_limits_sim0 = [0, np.max([np.max(cache_sim0) for i in cache_sim0])*0.9]
intensity_limits_sim12 = [0, np.max([np.max(cache_sim12) for i in cache_sim12])*0.9]

lines_through0 = []
lines_through12 = []

for k, param in enumerate([5, 10, 20, 40]):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    label = f"14.15$\\,$MeV with AirGap of {param}$\\,$mm"
    obj = cache_round12[k]
    A.maps = obj[1]
    A.maps[0]['x'] = A.maps[0]['x'] - 19
    A.maps[0]['y'] = A.maps[0]['y'] - 67.5

    y_mid_index = np.shape(data)[0] // 2
    # Linie extrahieren
    x_line = data[y_mid_index, :]
    lines_through12.append([A.maps[0]['x'], A.get_signal_xline(y_position=0), x_line])

    A.name = obj[0]
    crit = obj[0]
    insert_txt = [txt_posi, label, 15]
    A.plot_map(None, pixel='fill', imshow=True, intensity_limits=intensity_limits12, fig_in=fig, ax_in=ax1)

    data = cache_sim12[k]
    extent = (-np.shape(data)[1] * pixel_size / 2, np.shape(data)[1] * pixel_size / 2,
              - np.shape(data)[0] * pixel_size / 2, np.shape(data)[0] * pixel_size / 2)
    color_map = ax2.imshow(data, vmin=intensity_limits_sim12[0], vmax=intensity_limits_sim12[1], extent=extent, cmap=cmap)
    norm = matplotlib.colors.Normalize(vmin=intensity_limits_sim12[0], vmax=intensity_limits_sim12[1])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map.cmap)
    sm.set_array([])
    bar = fig.colorbar(sm, ax=ax2, extend='max')
    ax2.set_xlabel('Position x (mm)')
    ax2.set_ylabel('Position y (mm)')
    bar.set_label('Deposited energy (MeV)')
    name = f'Ideal_DoseMap_'

    ax1.set_xlim(-10, 10), ax1.set_ylim(-10, 10)
    ax2.set_xlim(-10, 10), ax2.set_ylim(-10, 10)

    ax1.text(*transform_axis_to_data_coordinates(ax1, [0.05, 0.93]), label, fontsize=13,
            c='k', zorder=7)  # , bbox={'facecolor': 'white', 'edgecolor': 'white', 'alpha': 0.9, 'pad': 2})
    ax2.text(*transform_axis_to_data_coordinates(ax2, [0.05, 0.93]), "GATE10 Simulation", fontsize=13,
             c='k', zorder=7)  # , bbox={'facecolor': 'white', 'edgecolor': 'white', 'alpha': 0.9, 'pad': 2})
    plot_size = (latex_textwidth * 2, latex_textwidth / 1.2419)
    format_save(results_path, save_name=f'StragglingP12{param}mm', save=True, legend=False, fig=fig, plot_size=plot_size)

    # ---------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    label = f"23.69$\\,$MeV with AirGap of {param}$\\,$mm"
    obj = cache_round0[k]
    A.maps = obj[1]
    A.maps[0]['x'] = A.maps[0]['x'] - 19
    A.maps[0]['y'] = A.maps[0]['y'] - 67.5

    y_mid_index = np.shape(data)[0] // 2
    # Linie extrahieren
    x_line = data[y_mid_index, :]
    lines_through0.append([A.maps[0]['x'], A.get_signal_xline(y_position=0), x_line])

    A.name = obj[0]
    crit = obj[0]
    insert_txt = [txt_posi, label, 15]
    A.plot_map(None, pixel='fill', imshow=True, intensity_limits=intensity_limits0, fig_in=fig, ax_in=ax1)

    data = cache_sim0[k]
    extent = (-np.shape(data)[1] * pixel_size / 2, np.shape(data)[1] * pixel_size / 2,
              - np.shape(data)[0] * pixel_size / 2, np.shape(data)[0] * pixel_size / 2)
    color_map = ax2.imshow(data, vmin=intensity_limits_sim0[0], vmax=intensity_limits_sim0[1], extent=extent,
                           cmap=cmap)
    norm = matplotlib.colors.Normalize(vmin=intensity_limits_sim0[0], vmax=intensity_limits_sim0[1])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map.cmap)
    sm.set_array([])
    bar = fig.colorbar(sm, ax=ax2, extend='max')
    ax2.set_xlabel('Position x (mm)')
    ax2.set_ylabel('Position y (mm)')
    bar.set_label('Deposited energy (MeV)')
    name = f'Ideal_DoseMap_'

    ax1.set_xlim(-10, 10), ax1.set_ylim(-10, 10)
    ax2.set_xlim(-10, 10), ax2.set_ylim(-10, 10)

    ax1.text(*transform_axis_to_data_coordinates(ax1, [0.05, 0.93]), label, fontsize=13,
             c='k', zorder=7)  # , bbox={'facecolor': 'white', 'edgecolor': 'white', 'alpha': 0.9, 'pad': 2})
    ax2.text(*transform_axis_to_data_coordinates(ax2, [0.05, 0.93]), "GATE10 Simulation", fontsize=13,
             c='k', zorder=7)  # , bbox={'facecolor': 'white', 'edgecolor': 'white', 'alpha': 0.9, 'pad': 2})
    plot_size = (latex_textwidth * 2, latex_textwidth / 1.2419)
    format_save(results_path, save_name=f'StragglingP0{param}mm', save=True, legend=False, fig=fig,
                plot_size=plot_size)


fig, ax = plt.subplots()
for obj in lines_through12:
    ax.plot(obj[0], obj[1]/intensity_limits12[1])
    ax.plot(np.linspace(-len(obj[2]) * pixel_size / 2, len(obj[2]) * pixel_size / 2, len(obj[2])), obj[2]/intensity_limits_sim12[1]/0.9, ls='--')

plt.show()