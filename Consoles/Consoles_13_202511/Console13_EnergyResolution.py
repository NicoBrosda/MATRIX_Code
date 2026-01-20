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

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_161225/')
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/ResultsPEEK_161225/')

measurements = []
# '''
energy_meas = [23, 23, 24, 24, 25, 25]
measurements += ['23MeV_at_Control_1nA.csv', '23MeV_at_Control_bis_1nA.csv', '24MeV_at_Control_1nA.csv', '24MeV_at_Control_bis_1nA.csv', '25MeV_at_Control_1nA.csv', '25MeV_at_Control_bis_1nA.csv']

dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_161225/')

dark_paths_array1 = ['dark_current']

norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_260325/')
norm_array1 = ['exp7_norm1,9V_']
# norm_array1 = ['2exp76_normday2_P0_']

cache = []
data_wheel_200 = pd.read_csv('../../Files/energies_after_wheel_diffusor200.txt', sep='\t', header=4,
                         names=['position', 'thickness', 'energies'])
param_cmap = sns.color_palette("crest_r", as_cmap=True)
comp_list = data_wheel_200['energies'].to_numpy()[:-1]
comp_list = np.array(energy_meas)
param_colormapper_200 = lambda param: color_mapper(param, np.min(comp_list), np.max(comp_list))
param_color = lambda param: param_cmap(param_colormapper_200(param))
param_unit = 'MeV'

A = Analyzer((2, 64), (0.4, 0.4), (0.1, 0.1), readout=readout,
                 diode_offset=[[0, - 0.25], np.zeros(64)], position_parser=position_parser,
                 voltage_parser=voltage_parser, current_parser=current_parser)
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

cache = []
energy = []
for measurement in tqdm(measurements):
    cache.append(A.readout(folder_path / measurement, A)['signal'])
    energy.append(float(measurement[:measurement.index('MeV')]))
cache = np.array(cache)
cache2 = np.empty((cache.shape[0], 128), dtype=cache.dtype)
cache2[:, 0::2] = cache[:, 1, :]
cache2[:, 1::2] = cache[:, 0, :]
cache = cache2
print(energy)
np.savetxt(results_path / 'energy_dep.txt', cache.T, delimiter=',', fmt='%.6f', header='23 MeV, 23 MeV, 24 MeV, 24 MeV, 25 MeV, 25 MeV')

print(np.shape(cache))

fig, ax = plt.subplots()

for i, curve in enumerate(cache[0:]):
    print(curve)
    cache_max = np.max(curve)
    cache_max = np.max(cache[0:])
    ax.plot([0.25*i for i in range(len(curve))], curve / cache_max, color=param_color(energy_meas[i]), lw=1.5, zorder=3)
    ax.axvline([0.25*i for i in range(len(curve))][np.argmax(curve / cache_max)], color=param_color(energy_meas[i]), zorder=1, alpha=0.5, lw=0.8)


def get_sim(run_name, diff=200, pixel_size=50/200, mean_range=(20, 30), simn=1e+6, param=0, mirror=True):

    # ----------------------------------------------------------------------------------------------------------------
    # Obtaining the data
    # ----------------------------------------------------------------------------------------------------------------
    _run_name = f"{run_name}{param}"
    current_path = Path('/Users/nico_brosda/GateSimulation/GATE10/Simulation/output/')
    output_path = current_path / f'{run_name[0:run_name.index("_")]}/'
    output_file = output_path / f"_{_run_name}_dose.mhd"

    # ----------------- Load the Dose / Edep image ----------------------------------
    # img = sitk.ReadImage(str(output_file).replace(".mhd", "_dose.mhd"))
    print('-'*50)
    output_path2 = Path(str(output_file).replace(".mhd", "_edep.mhd"))
    print(f"Checking file: {output_path2}")
    print(f"Exists? {output_path2.exists()}")
    print(output_path)
    print(f"Exists? {output_path.exists()}")

    img = sitk.ReadImage(str(output_file).replace(".mhd", "_edep.mhd"))
    img_std = sitk.ReadImage(str(output_file).replace(".mhd", "_edep_uncertainty.mhd"))
    if mirror:
        data = np.array(sitk.GetArrayFromImage(img))[0][::-1, :] / simn
        data_std = np.array(sitk.GetArrayFromImage(img_std))[0][::-1, :] / simn
    else:
        data = np.array(sitk.GetArrayFromImage(img))[0][:, :] / simn
        data_std = np.array(sitk.GetArrayFromImage(img_std))[0][:, :] / simn
    line = np.mean(data[:, int(mean_range[0] / pixel_size):int(mean_range[1] / pixel_size)], axis=1)[:]
    line_std = np.mean(data_std[:, int(mean_range[0] / pixel_size):int(mean_range[1] / pixel_size)], axis=1)[:]

    return data, line, line_std


x = 0.4
x = 0
start, end = 9, 41
crit = '1e+06Real5degWedgePEEK_param'
crit = '1e+06Real5degWedgeSigmaTestPEEK_param'
crit = '2e+06Real5degWedgeSigmaTestPEEK_param'
crit = '1e+07Real5degWedgePEEK_param'
crit = f'5e+0640umDiffSimPEEK_param'
crit = f'5e+0640umDiffSimSigmaPEEK_param'
crit = f'1e+0740umDiffSimSigmaPEEK_param'


param_list = [24.73, 23.73, 22.73]
param_list = [0, 0.01, 0.05, 0.1, 0.5, 1, 5]
param_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
param_list = [24.73, 23.73, 22.73]
param_list = [23, 24, 25]
param_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
param_list = [0, 0.05, 0.1, 0.15, 0.2]


colors = sns.color_palette("flare", as_cmap=True)

max_line_cache = 0
for j, param in enumerate(param_list):
    data_cache, line_cache, line_std_cache = get_sim(crit, param=param)
    line_cache = line_cache[int(start/0.25):int(end/0.25)]
    max_line_cache = max(np.max(line_cache), max_line_cache)

for j, param in enumerate(param_list):
    data_cache, line_cache, line_std_cache = get_sim(crit, param=param)
    line_cache = line_cache[int(start/0.25):int(end/0.25)]
    # ax.plot([0.25*i+x for i in range(len(line_cache))], line_cache / np.max(line_cache), label=param, color=colors(j/len(param_list)), lw=1.5, alpha=0.7, zorder=2)
    ax.plot([0.25*i+x for i in range(len(line_cache))], line_cache / max_line_cache, label=param, color=colors(j/len(param_list)), lw=1.5, alpha=0.7, zorder=2)
    ax.axvline([0.25*i+x for i in range(len(line_cache))][np.argmax(line_cache / np.max(line_cache))], color=colors(j/len(param_list)), zorder=1, alpha=0.5, lw=0.8)

ax.set_xlabel('Y-Position [mm]')
ax.set_ylabel('Signal normed / Signal max')
ax.legend()

ax.set_xlim(ax.get_xlim()), ax.set_ylim(ax.get_ylim())

shape = LineShape([[-4, 2.14], [36, 5.64]], distance_mode=False)
shape.print_shape()
shape.add_to_plot(0.0, 0.4, color='grey', alpha=1, zorder=-1, edgecolor='k', add_angle=True)
format_save(save_path=results_path/'SimCompGlobalNorm', save_name=f"{crit}", dpi=300, save_format='.png', fig=fig)

'''
fig, ax = plt.subplots()

crit = '2e+06DirectionCheck1PEEK_param'
param = 24.73
data_cache, line_cache, line_std_cache = get_sim(crit, param=param, mirror=True)
ax.plot([0.25 * i + x for i in range(len(line_cache))], line_cache / np.max(line_cache), label=param,
        color='r', lw=1.5, alpha=0.9)

crit = '2e+06DirectionCheck2PEEK_param'
param = 24.73
data_cache, line_cache, line_std_cache = get_sim(crit, param=param, mirror=False)
ax.plot([0.25 * i + x for i in range(len(line_cache))], line_cache / np.max(line_cache), label=param,
        color='b', lw=1.5, alpha=0.9)

plt.show()
'''
