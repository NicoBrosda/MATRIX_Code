from Consoles.StyleConsoles.Utils_ImageLoad import *
from EvaluationSoftware.simulation_connectors import *

SMALL_SIZE = 9  # 8
MEDIUM_SIZE = 10  # 9
BIGGER_SIZE = 11

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Save path and options for the map
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Style/Paper/')
background_subtraction = True
normalization = True

plot_size = (18 * cm, 2 / 2 * 18 / 1.2419 * cm)
# Setup of the final figure
# Structure: Ax1 Logo Line Super Res - Ax2 Logo Gaf - Ax3 BeamShape 2-Line - Ax4 BeamShape Gaf -
# Ax5 Logo Overlay - Ax6 Beam Overlay
fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=plot_size)
# Adjust spacing: left, right, bottom, top, wspace, hspace
fig.subplots_adjust(wspace=0.25, hspace=0.25)
# Axis limits
y_limits = []
x_limits = []
zero_scale = True
intensity_limits = None
intensity_limitsg = [0, 1]

pixel = 'fill'

dpi = 300
format = '.png'
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black", "red", "yellow"])

data_wheel_200 = pd.read_csv('../../Files/energies_after_wheel_diffusor200.txt', sep='\t', header=4,
                         names=['position', 'thickness', 'energies'])
data_wheel_400 = pd.read_csv('../../Files/energies_after_wheel_diffusor400.txt', sep='\t', header=4,
                         names=['position', 'thickness', 'energies'])

results_stem = Path('/Users/nico_brosda/Cyrce_Messungen/Results_260325/')
mapping = Path('../../Files/mapping.xlsx')
data = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:])-1 for k in data['direction_2']]
readout, position_parser, voltage_parser, current_parser = (
    lambda x, y: ams_2line_readout(x, y, channel_assignment=channel_assignment),
    standard_position,
    standard_voltage,
    current5
)
try:
    cache_400 = np.load(results_stem / 'Fast_Mode/cache_400.npy')
    cache_200 = np.load(results_stem / 'Fast_Mode/cache_200.npy')
    A = Analyzer((2, 64), (0.4, 0.4), (0.1, 0.1), readout=readout,
                 diode_offset=[[0, - 0.25], np.zeros(64)], position_parser=position_parser,
                 voltage_parser=voltage_parser, current_parser=current_parser)
except FileNotFoundError as _error:
    print(_error)

energy_cmap = sns.color_palette("crest_r", as_cmap=True)
energy_colormapper = lambda energy: color_mapper(energy, np.min(data_wheel_400['energies'][0:len(cache_400)]), np.max(data_wheel_400['energies'][0:len(cache_400)]))
energy_color = lambda energy: energy_cmap(energy_colormapper(energy))

sim_res_400, sim_std_400, sim_energy_400, sim_energy_std_400 = simulation_response('EnergyResponse400um2,73Al_param', diff=400)

# ----------------------------------------------------------------------------------------------------------------
# Calculate normed responses (normed to incoming protons)
#----------------------------------------------------------------------------------------------------------------
rescale_sim = 1e3
scale_sim = 'k'
simn = 1e7 / rescale_sim / 5026.548245743669 / 4.965

additional_scale = 1/e * 1e-18 * 4417.864669110646
rescale_current = 1e6 * additional_scale
scale_current = r'e$^{-}$/primary'
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
sim_res, sim_std = sim_res / simn / 2, sim_std / simn / 2

sim_res_400 = np.interp(data_wheel_400['energies'][0:len(cache_400)], sim_energy, sim_res)
sim_std_400 = np.interp(data_wheel_400['energies'][0:len(cache_400)], sim_energy, sim_std)
sim_res_200 = np.interp(data_wheel_200['energies'][0:len(cache_200)], sim_energy, sim_res)
sim_std_200 = np.interp(data_wheel_200['energies'][0:len(cache_200)], sim_energy, sim_std)

#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
# How good is this resembled by a proportional relation?
def proportional(x, a):
    return a*x

# How good is this resembled by a proportional relation?
def linear(x, a, b):
    return a*x + b

sim, res, std = np.append(sim_res_400, sim_res_200), np.append(normed_400, normed_200), np.append(std_400, std_200)
energies = np.append(data_wheel_400['energies'][0:len(cache_400)], data_wheel_200['energies'][0:len(cache_200)])
order = np.argsort(energies)
sim, res, std, energies = sim[order], res[order], std[order], energies[order]

energy_mask = energies > 15
sim2, res2, std2 = sim[energy_mask], res[energy_mask], std[energy_mask]

popt, pcov = curve_fit(proportional, sim, res)  # , sigma=std)
residuals = res - proportional(sim, *popt)
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((res - np.mean(res)) ** 2)
if ss_tot == 0:
    r_squared = 0
else:
    r_squared = 1 - (ss_res / ss_tot)

popt2, pcov2 = curve_fit(linear, sim, res)  # , sigma=std2)
residuals = res - linear(sim, *popt2)
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((res - np.mean(res)) ** 2)
if ss_tot == 0:
    r_squared2 = 0
else:
    r_squared2 = 1 - (ss_res / ss_tot)

energy_cmap = sns.color_palette("crest_r", as_cmap=True)
energy_colormapper = lambda energy: color_mapper(energy, np.min(energies), np.max(energies))
energy_color = lambda energy: energy_cmap(energy_colormapper(energy))
# ------------------------------------------------------------------------------------------------------------------
# Ax1: (Normed) Response vs proton energy
# ------------------------------------------------------------------------------------------------------------------
ax = ax1

# ax.plot(energies, res, c='k', ls='', marker='x')
for i, object in enumerate(energies):
    ax.plot(energies[i], res[i], ls='', marker='x', color=energy_color(energies[i]))
    continue
    ax.errorbar(data_wheel_400['energies'][i], normed_400[i], std_400[i], c=energy_color(data_wheel_400['energies'][i]), marker='', capsize=4, markersize=7)

'''
ax2 = ax.twinx()
ax2.plot(sim_energy_400[0:len(cache_400)], sim_res_400[0:len(cache_400)], c='r', ls='--', marker='', label='Simulation', zorder=0)
for i in range(len(sim_res_400[0:len(cache_400)])):
    ax2.errorbar(sim_energy_400[i], sim_res_400[i], sim_std_400[i], c=energy_color(data_wheel_400['energies'][i]),
                marker='', capsize=4, markersize=7)
ax2.set_ylabel(f'Deposited Energy per primary ({scale_sim}eV)')
ax2.set_ylim(0, 1.2*np.max(sim_res_400[0:len(cache_400)]))
'''

ax.set_xlabel(f'Proton energy (MeV)')
ax.set_ylabel(f'Signal Current ({scale_current})')

# ax.plot(100, 98, marker='x')
# ax.plot(226.7, 56.4, marker='x')
# ax.plot(np.append(data_wheel_200['energies'][0:len(cache_200)][::-1], [100, 226.7]), np.append(normed_200[::-1], [98, 56.4]), marker='x')

ax.set_xlim(ax.get_xlim())
ax.set_ylim(0, 1.2*np.max(normed_400))

start, end = [0.83, 0.8], [0.83, 0.65]
txtend = end[1] - 0.075 if end[1] < start[1] else end[1]+0.075
gradient_arrow(ax, transform_axis_to_data_coordinates(ax, start),
                       transform_axis_to_data_coordinates(ax, end), cmap=energy_cmap, lw=10, zorder=5)
ax.text(*transform_axis_to_data_coordinates(ax, start),
        f"{energies.min(): .2f}$\\,${'MeV'}", fontsize=11,
        c=energy_color(energies.min()), zorder=3, va='bottom', ha='center')  # , bbox={'facecolor': freq_colour(1033), 'alpha': 0.2, 'pad': 2})
ax.text(*transform_axis_to_data_coordinates(ax, [end[0], txtend]),
        f"{energies.max(): .2f}$\\,${'MeV'}", fontsize=11,
        c=energy_color(energies.max()), zorder=3, va='top', ha='center')  # , bbox={'facecolor': freq_colour(32033), 'alpha': 0.2, 'pad': 2})

# ------------------------------------------------------------------------------------------------------------------
# Ax2: Normed response vs deposited energy from simulation
# ------------------------------------------------------------------------------------------------------------------
ax = ax2

ax.plot(sim, res, c='k', ls='', marker='x', label='CYRCé data')
for i, object in enumerate(energies):
    ax.plot(sim[i], res[i], ls='', marker='x', color=energy_color(energies[i]))
    continue
    ax.errorbar(sim_res_400[i], normed_400[i], std_400[i], sim_std_400[i], c=energy_color(data_wheel_400['energies'][i]), marker='', capsize=4, markersize=7, alpha=0.7)

ax.plot(sim, proportional(sim, *popt), c='r', ls='--', zorder=2)
ax.plot(sim, linear(sim, *popt2), c='b', ls='--', zorder=2)

ax.plot(np.interp([100, 226.7], sim_energy, sim_res), [98, 56.4], ls='', marker='o', c='orange', label='WPE data')
ax.set_xlim(ax.get_xlim()), ax.set_ylim(ax.get_ylim())

ax.set_xlim(ax.get_xlim())
ax.set_ylim(0, 1.1*ax.get_ylim()[1])

ax.text(*transform_axis_to_data_coordinates(ax, [0.03, 0.85]), f'Proportional law $\\mathrm{"{R}"}^2$={r_squared:.4f}', fontsize=9, ha='left', va='top', c='r')
ax.text(*transform_axis_to_data_coordinates(ax, [0.03, 0.77]), f'Linear law $\\mathrm{"{R}"}^2$={r_squared2:.4f}', fontsize=9, ha='left', va='top', c='b')
ax.text(*transform_axis_to_data_coordinates(ax, [0.03, 0.975]), f'Effectivity C =', fontsize=9, ha='left', va='top')
ax.text(*transform_axis_to_data_coordinates(ax, [0.03, 0.92]), f'{popt[0]:.3f}$\\,${scale_current}/{scale_sim}eV', fontsize=9, ha='left', va='top')

ax.legend(loc='lower right')
ax.set_xlabel(r'Simulated E$_\mathrm{dep}$ (keV/primary)')
ax.set_ylabel(f'Signal Current ({scale_current})')


# ------------------------------------------------------------------------------------------------------------------
# Ax3: Old wedge - Bragg peak at different energies
# ------------------------------------------------------------------------------------------------------------------
ax = ax3

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_260325/')
results_stem = Path('/Users/nico_brosda/Cyrce_Messungen/Results_260325/')
mapping = Path('../../Files/mapping.xlsx')
data = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:])-1 for k in data['direction_2']]
readout, position_parser, voltage_parser, current_parser = (
    lambda x, y: ams_2line_readout(x, y, channel_assignment=channel_assignment),
    standard_position,
    standard_voltage,
    current5
)
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
additional_scale = 1/e * 1e-18 * 4417.864669110646
rescale_current = 1e6 * additional_scale
scale_current = f'$\\#$electrons'
currents_400_aperture = np.array([887, 888, 885, 880, 876, 872, 884, 880, 876, 871, 888, 887, 884, 881, 881, 877, 882, 880, 879]) * 1e-12 / e / rescale_current
currents_200_aperture = np.array([1.73, 1.72, 1.72, 1.70, 1.71, 1.70, 1.72, 1.71, 1.70, 1.72, 1.72, 1.71, 1.70, 1.72, 1.72, 1.71, 1.70, 1.69, 1.69, 1.76]) *1e-9 / e /rescale_current
factor_200 = []
for i in range(len(currents_200_aperture)):
    aperture_200[i]['z'] = aperture_200[i]['z'] / currents_200_aperture[i]
    factor_200.append(aperture_200[i])
    factor_200[i]['z'] = factor_200[i]['z'] / np.max(factor_200[i]['z'])
# '''
factor_400 = []
for i in range(len(currents_400_aperture)):
    aperture_400[i]['z'] = aperture_400[i]['z'] / currents_400_aperture[i]
    factor_400.append(aperture_400[i])
    factor_400[i]['z'] = factor_400[i]['z'] / np.max(factor_400[i]['z'])
rescale_current = 1e6 * additional_scale
scale_current = f'$\\#$electrons'
# Correctly ordered from P0 (or P12) increasing
currents_400 = np.array([882.5, 879, 877.5, 877, 874.5, 874, 872, 876.5, 876, 875.5, 873.5, 873.5, 872.5, 872, 870.5, 888, 888.5, 888, 881]) * 1e-12 / e / rescale_current
currents_200 = np.array([1.74, 1.736, 1.7295, 1.7245, 1.7195, 1.715, 1.7115, 1.7085, 1.710, 1.7345, 1.726, 1.735, 1.7335, 1.731, 1.7265, 1.726, 1.725, 1.7225, 1.7455, 1.7225]) * 1e-9 / e / rescale_current
currents_200_middle = np.array([1.7815, 1.781, 1.778, 1.778, 1.777, 1.7765, 1.7745, 1.774]) * 1e-9 / e / rescale_current
for i in range(len(currents_200)):
    wedge_200[i]['z'] = wedge_200[i]['z'] / currents_200[i]
for i in range(len(currents_400)):
    wedge_400[i]['z'] = wedge_400[i]['z'] / currents_400[i]
for i in range(len(currents_200_middle)):
    wedge_200_middle[i]['z'] = wedge_200_middle[i]['z'] / currents_200_middle[i]
def correct_wedge_with_aperture(wheel_position, map_set='400', threshold=0.1):
    # Select the appropriate maps based on the set parameter
    if map_set == '400':
        wedge_map = wedge_400[wheel_position]
        aperture_map = factor_400[wheel_position]
    elif map_set == '200':
        wedge_map = wedge_200[wheel_position]
        aperture_map = factor_200[wheel_position]
    elif map_set == '200_middle':
        wedge_map = wedge_200_middle[wheel_position - 12]
        aperture_map = factor_200[wheel_position]  # Note: using aperture_200 for 200_middle
    else:
        raise ValueError(f"Unknown map set: {map_set}. Use '400', '200', or '200_middle'")

    if (map_set == '200_middle' or map_set == '200') and wheel_position == 19:
        print(f"No valid aperture map for {map_set} at position {wheel_position} existent - Returning a zero map")
        return {'z': np.zeros(wedge_map['z'].shape), 'x': wedge_map['x'], 'y': wedge_map['y'],
                'position': wedge_map['position']}
    # Extract data and coordinates from input maps
    wedge_data = wedge_map['z']
    x_wedge = wedge_map['x']
    y_wedge = wedge_map['y']
    aperture_data = aperture_map['z']
    x_aperture = aperture_map['x']
    y_aperture = aperture_map['y']
    # Create result dictionary with same structure as input
    corrected_wedge_map = deepcopy(wedge_map)
    corrected_wedge_data = corrected_wedge_map['z']
    # Check if both datasets have identical coordinate systems
    if np.array_equal(x_wedge, x_aperture) and np.array_equal(y_wedge, y_aperture):
        # Fast path: direct vectorized operation when coordinates match exactly
        print(f"Using fast path for {map_set} at position {wheel_position}")
        # Create mask for valid aperture signals
        valid_aperture = aperture_data > threshold
        # Apply correction only where aperture signal is valid
        # and avoid division by zero
        valid_division = np.logical_and(valid_aperture, aperture_data != 0)
        corrected_wedge_data[valid_division] = wedge_data[valid_division] / aperture_data[valid_division]
        # Set values to zero where aperture signal is below threshold
        corrected_wedge_data[~valid_division] = 0  # or keep original with: wedge_data[~valid_division]
    else:
        # Slower path: process only common coordinates
        print(f"Using slow path for {map_set} at position {wheel_position}")
        # Find common coordinates
        common_x = np.intersect1d(x_wedge, x_aperture)
        common_y = np.intersect1d(y_wedge, y_aperture)
        # Process only points that exist in both datasets
        for x in common_x:
            for y in common_y:
                # Find indices for this point in both arrays
                i_wedge = np.where(x_wedge == x)[0]
                j_wedge = np.where(y_wedge == y)[0]
                i_aperture = np.where(x_aperture == x)[0]
                j_aperture = np.where(y_aperture == y)[0]
                # Check if the point exists in both datasets and indices are valid
                if (i_wedge.size > 0 and j_wedge.size > 0 and
                        i_aperture.size > 0 and j_aperture.size > 0):
                    i_w, j_w = i_wedge[0], j_wedge[0]
                    i_a, j_a = i_aperture[0], j_aperture[0]
                    # Ensure indices are within bounds
                    if (i_w < wedge_data.shape[1] and j_w < wedge_data.shape[0] and
                            i_a < aperture_data.shape[1] and j_a < aperture_data.shape[0]):
                        # Apply correction logic
                        if aperture_data[j_a, i_a] > threshold and aperture_data[j_a, i_a] != 0:
                            corrected_wedge_data[j_w, i_w] = wedge_data[j_w, i_w] / aperture_data[j_a, i_a]
                        else:
                            # No sufficient signal in aperture map or division by zero
                            corrected_wedge_data[j_w, i_w] = 0  # or keep original with: wedge_data[i_w, j_w]
                    else:
                        pass
                        # print(f"Warning: Index out of bounds: wedge[{i_w},{j_w}], aperture[{i_a},{j_a}]")
    return corrected_wedge_map
threshold = 0.3
for wheel_position in range(19):
    wedge_200[wheel_position] = correct_wedge_with_aperture(wheel_position, map_set='200', threshold=threshold)
    wedge_400[wheel_position] = correct_wedge_with_aperture(wheel_position, map_set='400', threshold=threshold)
    if wheel_position >= 12:
        wedge_200_middle[wheel_position-12] = correct_wedge_with_aperture(wheel_position, map_set='200_middle', threshold=threshold)
x_range = (16, 18)
signal_cache_200 = []
for data in wedge_200:
    x_data = data['x']
    indices = [np.argmin(np.abs(x_data - x_range[0])), np.argmin(np.abs(x_data - x_range[1]))]
    signal_cache_200.append([data['wheel_position'], data['y'], np.mean(data['z'][:, indices[0]:indices[1]], axis=1)])
x_range = (16, 18)
signal_cache_200_middle = []
for data in wedge_200_middle:
    x_data = data['x']
    indices = [np.argmin(np.abs(x_data - x_range[0])), np.argmin(np.abs(x_data - x_range[1]))]
    signal_cache_200_middle.append(
        [data['wheel_position'], data['y'], np.mean(data['z'][:, indices[0]:indices[1]], axis=1)])
x_range = (16, 18)
signal_cache_400 = []
for data in wedge_400:
    x_data = data['x']
    indices = [np.argmin(np.abs(x_data - x_range[0])), np.argmin(np.abs(x_data - x_range[1]))]
    signal_cache_400.append(
        [data['wheel_position'], data['y'], np.mean(data['z'][:, indices[0]:indices[1]], axis=1)])
A.scale = 'atto'

signal_max = np.max(signal_cache_200[1][-1])
for i in range(len(signal_cache_200)):
    x = signal_cache_200[i][1]
    ind = (28.25-16 <= x) & (x <= 28.25+16)
    x, sig = x[ind]-28.25+16, signal_cache_200[i][-1][ind]
    # ax.plot(x, sig/signal_max, color=energy_color(signal_cache_200[i][0]))
    # ax.plot([i*0.25 for i in range(len(signal_cache_200[i][-1]))], signal_cache_200[i][-1]/signal_max, color=energy_color(signal_cache_200[i][0]))

PEEK_energies = np.array([data_wheel_400['energies'][i] for i, j in enumerate(signal_cache_400[0:-3])])

signal_max = np.max(signal_cache_400[1][-1])
for i in range(len(signal_cache_400[0:-3])):
    x = signal_cache_400[i][1]
    ind = (28.25-16 <= x) & (x <= 28.25+16)
    x, sig = x[ind]-28.25+16, signal_cache_400[i][-1][ind]
    # ax.plot(x, sig/signal_max, color=energy_color(signal_cache_200[i][0]))
    ax.plot([i*0.25 for i in range(len(signal_cache_400[i][-1]))], signal_cache_400[i][-1]/signal_max, color=energy_color(PEEK_energies[i]))

ax.set_xlabel(r'Position (mm)')
ax.set_ylabel(f'Normed signal current')

ax.set_xlim(ax.get_xlim()), ax.set_ylim(ax.get_ylim())

shape = LineShape([[-4, 0], [36, 10]], distance_mode=False)
shape.print_shape()
shape.add_to_plot(0.0, 0.5, color='grey', alpha=1, zorder=-1, edgecolor='k', add_angle=True, ax=ax)

ax.set_xlim(3, 36)

PEEK_energies = np.array([data_wheel_400['energies'][i] for i, j in enumerate(signal_cache_400[0:-3])])
start, end = [0.83, 0.80], [0.83, 0.67]
txtend = end[1] - 0.075 if end[1] < start[1] else end[1]+0.075
frac1 = 1 - (PEEK_energies.min() - energies.min()) / (energies.max()- energies.min())
frac2 = 1 - (PEEK_energies.max() - energies.min()) / (energies.max()- energies.min())
print(f"frac1: {frac1}, frac2: {frac2}")
new_cmap = truncated_colormap(energy_cmap, frac1, frac2, reverse=True)

gradient_arrow(ax, transform_axis_to_data_coordinates(ax, start),
                       transform_axis_to_data_coordinates(ax, end), cmap=new_cmap, lw=10, zorder=5)
ax.text(*transform_axis_to_data_coordinates(ax, start),
        f"{PEEK_energies.min(): .2f}$\\,${'MeV'}", fontsize=11,
        c=energy_color(PEEK_energies.min()), zorder=3, va='bottom', ha='center')  # , bbox={'facecolor': freq_colour(1033), 'alpha': 0.2, 'pad': 2})
ax.text(*transform_axis_to_data_coordinates(ax, [end[0], txtend]),
        f"{PEEK_energies.max(): .2f}$\\,${'MeV'}", fontsize=11,
        c=energy_color(PEEK_energies.max()), zorder=3, va='top', ha='center')  # , bbox={'facecolor': freq_colour(32033), 'alpha': 0.2, 'pad': 2})

# ------------------------------------------------------------------------------------------------------------------
# Ax4: New 5° wedge - high obtained energy resolution + comparison with simulation
# ------------------------------------------------------------------------------------------------------------------
ax = ax4

colors = sns.color_palette("crest_r", as_cmap=True)

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
measurements = []
# '''
energy_meas = [23, 23, 24, 24, 25, 25]
# measurements += ['23MeV_at_Control_1nA.csv', '23MeV_at_Control_bis_1nA.csv', '24MeV_at_Control_1nA.csv', '24MeV_at_Control_bis_1nA.csv', '25MeV_at_Control_1nA.csv', '25MeV_at_Control_bis_1nA.csv']
measurements += ['23MeV_at_Control_bis_1nA.csv', '24MeV_at_Control_bis_1nA.csv', '25MeV_at_Control_bis_1nA.csv']

dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_161225/')
dark_paths_array1 = ['dark_current']
norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_260325/')
norm_array1 = ['exp7_norm1,9V_']
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

print(':'*50)
print(cache)
for i, curve in enumerate(cache[0:]):
    print(curve)
    cache_max = np.max(curve)
    cache_max = np.max(cache[0:])
    ax.plot([0.25*i for i in range(len(curve))], curve / cache_max, color=colors(i/len(cache)), lw=1.5, zorder=3, label=f'{energy[i]:.1f} MeV')
    ax.axvline([0.25*i for i in range(len(curve))][np.argmax(curve / cache_max)], color=colors(i/len(cache)), zorder=2, alpha=1, lw=1.5)

x = 0
start, end = 9, 41
crit = f'5e+075degInitialNewPEEK_param'
param_list = [24.92, 23.92, 22.92]
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

ax.legend()

ax.set_xlim(ax.get_xlim()), ax.set_ylim(ax.get_ylim())

# Positioning: The measurements are aligned and compared with their centers overlapping!
# So we consider the 2x64 line array which has a length of 32.25 mm
# The simulation is considerend on a 50x50 mm block of GaN divided into 0.25 pixels
# -> If we consider the measurements from 0 to 32 mm we need to look at the center ± 16 mm around middle 25 mm: 9 - 41 mm
# The wedge is positioned with ± 20 mm from beam center in plot coordinates 16 mm
shape = LineShape([[-4, 2.14], [36, 5.64]], distance_mode=False)
shape.print_shape()
shape.add_to_plot(0.0, 0.28, color='grey', alpha=1, zorder=-1, edgecolor='k', add_angle=True, ax=ax)

for j, param in enumerate(param_list):
    data_cache, line_cache, line_std_cache = get_sim(crit, param=param)
    line_cache = line_cache[int(start/0.25):int(end/0.25)]
    print(param)
    print(shape.calculate_value([0.25*i+x for i in range(len(line_cache))][np.argmax(line_cache / np.max(line_cache))]))

ax.set_xlabel(r'Position (mm)')
ax.set_ylabel(f'Normed signal Current')

add_png_icon(ax1, A, 'bottom left', translation=None, zoom=0.2)

ax1.text(*transform_axis_to_data_coordinates(ax1, [0.97, 0.97]), r'\textbf{(a)}', fontsize=10, ha='right',
         va='top', color='k')
ax2.text(*transform_axis_to_data_coordinates(ax2, [0.97, 0.97]), r'\textbf{(b)}', fontsize=10, ha='right',
         va='top', color='k')
ax3.text(*transform_axis_to_data_coordinates(ax3, [0.97, 0.97]), r'\textbf{(c)}', fontsize=10, ha='right',
         va='top', color='k')
ax4.text(*transform_axis_to_data_coordinates(ax4, [0.97, 0.97]), r'\textbf{(d)}', fontsize=10, ha='right',
         va='top', color='k')

format_save(save_path=results_path, save_name=f"Graph5_EnergyCharacteristics", dpi=dpi, plot_size=plot_size, save_format=format, fig=fig, legend=False)