from Consoles.Consoles9.Console9_EnergyExtrapolation import new_measurements
from EvaluationSoftware.main import *
from EvaluationSoftware.simulation_connectors import get_sim

save_format = '.png'

# Energy wheel
diff = 200
dat = pd.read_csv(Path(f'../../Files/energies_after_wheel_diffusor{diff}.txt'), header=4, delimiter='\t', decimal='.',
                      names=['pos', 'thickness', 'energy'])
param_list = dat['thickness']

mapping = Path('../../Files/mapping.xlsx')
data = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:])-1 for k in data['direction_2']]
readout, position_parser, voltage_parser, current_parser = (
    lambda x, y: ams_2line_fast_avg(x, y, channel_assignment=channel_assignment),
    standard_position,
    standard_voltage,
    current5
)
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_171225/')
results_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/ResultsPEEK_161225/')

crit = f'exp{8}_'

dark_path = folder_path
dark_paths_array1 = ['dark_current']
norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_260325/')
norm_array1 = ['exp7_norm1,9V_']
# norm_array1 = ['2exp76_normday2_P0_']

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

cache = []
wheel_pos = []
energy = []
for k, measurement in enumerate(A.measurement_files):
    print('-'*50)
    print(measurement)
    print('-'*50)
    match = re.search(r"nA_p(\d+)_x", str(measurement))
    if match:
        number = int(match.group(1))
        print(number)
        wheel_pos.append(number)
    else:
        wheel_pos.append(None)
    cache.append((A.readout(folder_path / measurement, A)['signal']-A.dark)*A.norm_factor)

cache = np.array(cache)
wheel_pos = np.array(wheel_pos)

cache2 = np.empty((cache.shape[0], 128), dtype=cache.dtype)
cache2[:, 0::2] = cache[:, 1, :]
cache2[:, 1::2] = cache[:, 0, :]
cache = cache2
print(np.shape(cache))
cache = A.signal_conversion(cache)

cache = cache[np.argsort(wheel_pos)][:]
wheel_pos = wheel_pos[np.argsort(wheel_pos)][:]

param_list = param_list[:len(cache)]

param_cmap = sns.color_palette("crest_r", as_cmap=True)
param_colormapper = lambda param: color_mapper(param, np.min(param_list), np.max(param_list))
param_color = lambda param: param_cmap(param_colormapper(param))

shape = LineShape([[45, 5.64], [5, 2.14]], distance_mode=False)

# --- Plot curves with energy wheel behind wedge ---
fig, ax = plt.subplots()
for i, curve in enumerate(cache[0:]):
    ax.plot([i*0.25+25-16 for i in range(len(curve))], curve, color=param_color(wheel_pos[i]), ls='-')
shape.add_to_plot(0, 0.3, ax, add_angle=False, color='grey', alpha=0.5, edgecolor='k')
improved_gradient_scale(param_list, param_cmap, ax, 'mm', [0.1, 0.94], param_colormapper)
format_save(results_path, "5degWheel", save_format=save_format, legend=False)

# --- Plot curves with energy wheel behind wedge + damage correction ---
compensation = np.load(results_path / 'compensation_factor.npy')
correction_factor = np.load(results_path / 'correction_factor.npy')
fig, ax = plt.subplots()
for i, curve in enumerate(cache[0:]):
    ax.plot([i*0.25+25-16 for i in range(len(curve))], curve/compensation, color=param_color(wheel_pos[i]), ls='-')
shape.add_to_plot(0, 0.3, ax, add_angle=False, color='grey', alpha=0.5, edgecolor='k')
improved_gradient_scale(param_list, param_cmap, ax, 'mm', [0.1, 0.94], param_colormapper)
format_save(results_path, "5degWheel_correction", save_format=save_format, legend=False)

# --- Compare with simulation ---
fig, ax = plt.subplots()
max_exp = 0
for i, curve in enumerate(cache[0:]):
    signal = curve / compensation
    max_exp = max(np.max(signal), max_exp)

for i, curve in enumerate(cache[0:]):
    signal = curve / compensation
    ax.plot([i*0.25+25-16 for i in range(len(curve))], signal/max_exp, color=param_color(wheel_pos[i]), ls='-')

x = 9
start, end = 9, 41
crit = f'1e+075degEnergyWheel100diffPEEK_param'
# crit = f'1e+075degEnergyWheel200diffPEEK_param'

max_line_cache = 0
for j, param in enumerate(param_list):
    data_cache, line_cache, line_std_cache = get_sim(crit, param=param)
    line_cache = line_cache[int(start/0.25):int(end/0.25)]
    max_line_cache = max(np.max(line_cache), max_line_cache)

for j, param in enumerate(param_list):
    data_cache, line_cache, line_std_cache = get_sim(crit, param=param)
    line_cache = line_cache[int(start/0.25):int(end/0.25)]
    # ax.plot([0.25*i+x for i in range(len(line_cache))], line_cache / np.max(line_cache), label=param, color=colors(j/len(param_list)), lw=1.5, alpha=0.7, zorder=2)
    if j == 0:
        ax.plot([0.25*i+x for i in range(len(line_cache))], line_cache / max_line_cache, color='orange', ls='--',  lw=1.5, alpha=1, zorder=4, label='GATE')
    else:
        ax.plot([0.25*i+x for i in range(len(line_cache))], line_cache / max_line_cache, color='orange', ls='--',  lw=1.5, alpha=1, zorder=4)
    ax.axvline([0.25*i+x for i in range(len(line_cache))][np.argmax(line_cache / np.max(line_cache))], color='orange', ls='--', zorder=2, alpha=1, lw=1.2)

shape.add_to_plot(0, 0.3, ax, add_angle=False, color='grey', alpha=0.5, edgecolor='k')

improved_gradient_scale(param_list, param_cmap, ax, 'mm', [0.1, 0.94], param_colormapper)
ax.legend()
format_save(results_path, "5degWheel_SimComp_100diff", save_format=save_format, legend=False)

