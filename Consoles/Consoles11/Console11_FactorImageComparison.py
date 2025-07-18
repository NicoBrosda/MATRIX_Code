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

results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Results_260325/Homogeneity/ImageComp')

new_measurements = []
new_measurements += ['exp14_']

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_260325/')
dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_260325/')
dark_paths = ['exp1_dark_0nA_400um_nA_1.9_x_20.0_y_68.0', 'exp64_darkEnd_0.5nA_400um_nA_1.9_x_20.0_y_68.0',
                     '2exp66_Dark_0.0nA_0um_nA_1.9_x_20.0_y_68.0', '2exp138_DarkEnd_0nA_200um_nA_1.9_x_20.0_y_68.0']
def dark_voltage(voltage):
    return [f'exp1_dark_0nA_400um_nA_{voltage:.1f}_x_20.0_y_68.0',
            f'exp64_darkEnd_0.5nA_400um_nA_{voltage:.1f}_x_20.0_y_68.0',
            f'2exp66_Dark_0.0nA_0um_nA_{voltage:.1f}_x_20.0_y_68.0',
            f'2exp138_DarkEnd_0nA_200um_nA_{voltage:.1f}_x_20.0_y_68.0']

norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_260325/')
# norm_array1 = ['exp7_norm1,9V_']
# norm_array1 = ['2exp76_normday2_P0_']

shortlabels = ['1,9V_22,7MeV_400um', '1,5V_22,7MeV_400um', '1,1V_22,7MeV_400um', '1,9V_17,5MeV_400um',
          '1,9V_12,7MeV_400um', '1,9V_7,4MeV_400um', '1,9V_2,9MeV_400um', '1,9V_2,9MeV_400um',
          '1,5V_2,9MeV_400um', '1,1V_2,9MeV_400um', '1,9V_23.7MeV_200um', '1,9V_23.7MeV_200um_5monthBefore']

folder_path2 = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
dark_path2 = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
dark_paths2 = ['2Line_DarkVoltageScan_200_ um_0_nA_nA_1.9_x_22.0_y_66.625.csv']

for k, crit in enumerate(new_measurements[0:]):
    print('-'*50)
    print(crit)
    print('-'*50)

    instance = Analyzer((2, 64), (0.4, 0.4), (0.1, 0.1), readout=readout,
                 diode_offset=[[0, - 0.25], np.zeros(64)], position_parser=position_parser,
                 voltage_parser=voltage_parser, current_parser=current_parser)
    dark = dark_paths
    instance.set_dark_measurement(dark_path, dark)
    instance.set_measurement(folder_path, crit)
    instance.load_measurement()
    cache_save = deepcopy(instance.measurement_data)
    instance.create_map(inverse=[True, False])
    intensity_limits = [0, np.max(instance.maps[0]['z'])]
    instance.name = "_NoNormalization"

    # --------------------------------------------------------------------------------------------------------------

    fig, ax = plt.subplots()
    instance.plot_map(None, pixel='fill', intensity_limits=intensity_limits, fig_in=fig, ax_in=ax)
    ax.set_xlim(ax.get_xlim()), ax.set_ylim(ax.get_ylim())
    text = f"No Normalization"
    ax.text(*transform_axis_to_data_coordinates(ax, [0.01, 0.99]), text, fontsize=12, ha='left',
            va='top')  # , bbox={'facecolor': 'white', 'edgecolor': 'white', 'alpha': 0.7, 'pad': 2})
    format_save(results_path, instance.name, legend=False, fig=fig)

    # --------------------------------------------------------------------------------------------------------------
    print(np.shape(instance.maps[0]['x']), np.shape(instance.maps[0]['y']), np.shape(instance.maps[0]['z']))
    x_range = (instance.maps[0]['x'] > 15) & (instance.maps[0]['x'] < 25)
    y_range = (instance.maps[0]['y'] > 60) & (instance.maps[0]['y'] < 75)
    print(np.shape(x_range), np.shape(y_range))

    fig, ax = plt.subplots()
    data = instance.maps[0]['z'][y_range][:, x_range].flatten()
    ax.hist(data, bins=50, edgecolor='black', color='k')
    ax.set_xlim(ax.get_xlim()), ax.set_ylim(ax.get_ylim())
    text = f"No Normalization \n Std of signal {np.std(data)*100:.2f}$\\,$\\%"
    ax.text(*transform_axis_to_data_coordinates(ax, [0.01, 0.99]), text, fontsize=12, ha='left',
            va='top', color='k')  # , bbox={'facecolor': 'white', 'edgecolor': 'white', 'alpha': 0.7, 'pad': 2})
    format_save(results_path, f"HistAlign_{instance.name}", legend=False, fig=fig)

    # --------------------------------------------------------------------------------------------------------------
    color = sns.color_palette("Spectral", as_cmap=True)

    for i, norm in enumerate([f'exp7_', f'exp8_', f'exp9_', f'exp10_', f'exp11_', f'exp12_', f'exp13_', f'exp33_',
                              f'exp34', f'exp35', f'exp76', '2Line_YScan_']):

        c = color(i / (len(shortlabels) - 1))

        instance2 = deepcopy(instance)
        if i > len(shortlabels) - 2:
            instance2.set_measurement(folder_path2, [norm])
            instance2.set_dark_measurement(dark_path2, dark_paths2)
        else:
            instance2.set_measurement(folder_path, [norm])
            voltage = voltage_parser(instance2.measurement_files[0])
            print(voltage)
            instance2.set_dark_measurement(dark_path, dark_voltage(voltage))

        factor, diff = normalization_from_translated_array_v5(instance2.measurement_files, instance2, align_lines=True,
                                                              remove_background=True)
        factor2, diff2 = normalization_from_translated_array_v5(instance2.measurement_files, instance2, align_lines=False,
                                                                remove_background=True)

        if i > len(shortlabels) - 2:
            factor[factor < 0.7] = cache[factor < 0.7]
            factor2[factor2 < 0.7] = cache2[factor2 < 0.7]
        else:
            cache, cache2 = factor, factor2

        # --------------------------------------------------------------------------------------------------------------

        instance.measurement_data = deepcopy(cache_save)
        instance.norm_factor = factor
        instance.update_measurement(dark=False)
        instance.create_map(inverse=[True, False])
        intensity_limits = [0, np.max(instance.maps[0]['z'])]
        instance.name = f"{norm}_Align"
        data = instance.maps[0]['z'][y_range][:, x_range].flatten()

        fig, ax = plt.subplots()
        instance.plot_map(None, pixel='fill', intensity_limits=intensity_limits, fig_in=fig, ax_in=ax)
        ax.set_xlim(ax.get_xlim()), ax.set_ylim(ax.get_ylim())
        text = f"{shortlabels[i]} Lines Aligned \n Std of signal {np.std(data)*100:.2f}$\\,$\\%"
        ax.text(*transform_axis_to_data_coordinates(ax, [0.01, 0.99]), text, fontsize=12, ha='left',
                va='top', color=c)  # , bbox={'facecolor': 'white', 'edgecolor': 'white', 'alpha': 0.7, 'pad': 2})
        format_save(results_path, instance.name, legend=False, fig=fig)

        # --------------------------------------------------------------------------------------------------------------

        fig, ax = plt.subplots()
        ax.hist(data, bins=50, edgecolor='black', color=c)
        ax.set_xlim(ax.get_xlim()), ax.set_ylim(ax.get_ylim())
        text = f"{shortlabels[i]} Lines Aligned \n Std of signal {np.std(data)*100:.2f}$\\,$\\%"
        ax.text(*transform_axis_to_data_coordinates(ax, [0.01, 0.99]), text, fontsize=12, ha='left',
                va='top', color=c)  # , bbox={'facecolor': 'white', 'edgecolor': 'white', 'alpha': 0.7, 'pad': 2})
        format_save(results_path, f"HistAlign_{instance.name}", legend=False, fig=fig)

        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        instance.measurement_data = deepcopy(cache_save)
        instance.norm_factor = factor2
        instance.update_measurement(dark=False)
        instance.create_map(inverse=[True, False])
        intensity_limits = [0, np.max(instance.maps[0]['z'])]
        instance.name = f"{norm}_noAlign"
        data = instance.maps[0]['z'][y_range][:, x_range].flatten()

        fig, ax = plt.subplots()
        instance.plot_map(None, pixel='fill', intensity_limits=intensity_limits, fig_in=fig, ax_in=ax)
        ax.set_xlim(ax.get_xlim()), ax.set_ylim(ax.get_ylim())
        text = f"{shortlabels[i]} \n Std of signal {np.std(data)*100:.2f}$\\,$\\%"
        ax.text(*transform_axis_to_data_coordinates(ax, [0.01, 0.99]), text, fontsize=12, ha='left',
                va='top', color=c)  # , bbox={'facecolor': 'white', 'edgecolor': 'white', 'alpha': 0.7, 'pad': 2})
        format_save(results_path, instance.name, legend=False, fig=fig)

        # --------------------------------------------------------------------------------------------------------------

        fig, ax = plt.subplots()
        ax.hist(data, bins=50, edgecolor='black', color=c)
        ax.set_xlim(ax.get_xlim()), ax.set_ylim(ax.get_ylim())
        text = f"{shortlabels[i]} \n Std of signal {np.std(data)*100:.2f}$\\,$\\%"
        ax.text(*transform_axis_to_data_coordinates(ax, [0.01, 0.99]), text, fontsize=12, ha='left',
                va='top', color=c)  # , bbox={'facecolor': 'white', 'edgecolor': 'white', 'alpha': 0.7, 'pad': 2})
        format_save(results_path, f"HistNoAlign_{instance.name}", legend=False, fig=fig)

        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

