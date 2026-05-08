from Consoles.StyleConsoles.Utils_ImageLoad import *
from EvaluationSoftware.simulation_connectors import *
from scipy.optimize import least_squares
import time

# --------------------------------------------------
# Result plots
# --------------------------------------------------
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black", "red", "yellow"])
diode_colour = sns.color_palette("icefire", as_cmap=True)

diodes_to_plot = [0, 50, 107]
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/ResultsPEEK_161225/')
# save_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/ResultsPEEK_161225/IdealSimulationComp/')
save_format = '.png'

# --------------------------------------------------
# Paths for simulation
# --------------------------------------------------
run_stem = '1e+085degActiveLayerSortieAir50PEEK'
output_path = Path("/Users/nico_brosda/GateSimulation/GATE10/Simulation/output/")
hdf5_filename = output_path / f"{run_stem}/{run_stem}_param24.92.h5"
run_name = f"{run_stem}_param24.92"
save_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/ResultsPEEK_161225/{run_stem}Comp/')


Np = 1e8   # <-- your number of primaries

dose_base_path = output_path / run_name[:run_name.rindex('_')]

# --------------------------------------------------
# Get SRIM for comp
# --------------------------------------------------
df = pd.read_excel(Path('/Users/nico_brosda/Cyrce_Messungen/SRIM_Simulations/') / 'SRIM_results.xlsx')
# df = pd.read_fwf(Path('/Users/nico_brosda/Desktop/AFP Stuff/Software/SRIM2013/SRIM Outputs/SRIM_results.txt'), header=1, names=['Ion Energy', 'Elec. dE/dx', 'Nuclear dE/dx', 'Projected Range', 'Longitudinal Straggling', 'Lateral Straggling', ])
df = convert_columns_to_units(df, {'Ion Energy': 'MeV', 'Projected Range': 'mm', 'Longitudinal Straggling': 'mm', 'Lateral Straggling': 'mm'})
df['Elec. dE/dx'], df['Nuclear dE/dx'] = df['Elec. dE/dx']/1e3, df['Nuclear dE/dx']/1e3

# --------------------------------------------------
# Define wedge geometry
# --------------------------------------------------
shape = LineShape([[5, 2.14], [45, 5.64]], distance_mode=False)
shape.print_shape()
wedge_thickness = []
for row in tqdm(range(200)):
    position = 0.25 * row + 0.125
    if position < 5 or position > 45:
        wedge_th = 0
        wedge_range = 0
    else:
        wedge_th_left = shape.calculate_value(position - 0.125)
        wedge_th = shape.calculate_value(position)
        wedge_th_right = shape.calculate_value(position + 0.125)

        wedge_range = np.abs(wedge_th_right - wedge_th_left) / 2
    wedge_thickness.append(wedge_th)

# --------------------------------------------------
# 1. Load Dose and Deposited Energy images
# --------------------------------------------------
output_file = dose_base_path / f"_{run_name}_dose.mhd"
dose_img = sitk.ReadImage(str(output_file).replace(".mhd", "_dose.mhd"))
edep_img = sitk.ReadImage(str(output_file).replace(".mhd", "_edep.mhd"))

dose = sitk.GetArrayFromImage(dose_img)   # Gy
edep = sitk.GetArrayFromImage(edep_img)  # MeV

# If single layer detector, remove z dimension
if dose.ndim == 3:
    dose = dose[0][::-1]
    edep = edep[0][::-1]

# --------------------------------------------------
# 2. Normalize to number of primaries
# --------------------------------------------------
cache_dir = dose_base_path / "cache/"
os.makedirs(cache_dir, exist_ok=True)
cache_file = cache_dir / f"{run_name}_phsp1_energy_stats.npz"
try:
    data = np.load(cache_file)
    mean_energy   = data["mean_energy"]
    median_energy = data["median_energy"]
    sigma_energy  = data["sigma_energy"]
    rel_spread    = data["rel_spread"]
    counts        = data["counts"]
    valid_mask    = data["valid_mask"]
    print("Loaded cached PHSP1 energy statistics.")

except (FileNotFoundError, KeyError):
    # If file doesn't exist or is corrupted/incomplete → recompute
    print("Cache not found. Computing PHSP1 energy statistics...")
    mean_energy, median_energy, sigma_energy, rel_spread, counts, valid_mask = \
        compute_phsp1_energy_stats(hdf5_filename, run_name)
    np.savez(
        cache_file,
        mean_energy=mean_energy,
        median_energy=median_energy,
        sigma_energy=sigma_energy,
        rel_spread=rel_spread,
        counts=counts,
        valid_mask=valid_mask,
    )
    print("Saved PHSP1 energy statistics to cache.")

proton_current_scaling = get_proton_counts(hdf5_filename, run_name)
print(proton_current_scaling[1]/proton_current_scaling[0])

mean_energy, median_energy, sigma_energy, rel_spread, counts, valid_mask = (mean_energy[::-1], median_energy[::-1],
                                                                            sigma_energy[::-1], rel_spread[::-1],
                                                                            counts[::-1], valid_mask[::-1])

filter_mask = counts <= 50
mean_energy[filter_mask] = 0
dose_norm = dose / counts
edep_norm = edep / counts
# Modified since we do not look at the number of primaries, but the number of protons in the setup:
proton_density_distribution = counts / proton_current_scaling[1]
dose_norm[filter_mask] = 0
edep_norm[filter_mask] = 0

median_energy_per_diode = []
for i in range(128):
    position = 25 - 16.25 + i * 0.25
    sim_pixel = int(position / 0.25)
    median_energy_per_diode.append(np.mean(median_energy[sim_pixel, 98:102]))

mean_energy_per_diode = []
for i in range(128):
    position = 25 - 16.25 + i * 0.25
    sim_pixel = int(position / 0.25)
    mean_energy_per_diode.append(np.mean(mean_energy[sim_pixel, 98:102]))

print('Total number of detector counts: ', np.sum(counts.flatten()))
print('Ratio of Total number of detector counts to total in simulation: ', np.sum(counts.flatten())/Np)

# --------------------------------------------------
# 3. Assign simulated pixels to real pixels (simulated grid had 0.25 mm x 0.25 mm pixel
# --------------------------------------------------
# For simulation the center is at y=25 mm
pixel_ycoords = np.arange(0, np.shape(dose)[0]) * 0.25 - 25
pixel_xcoords = np.arange(0, np.shape(dose)[1]) * 0.25

# For the experiment the center is at y=16 - half the array pixels - the array is loaded from

# --------------------------------------------------
# 4. Load long term irradiation data
# --------------------------------------------------
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/ResultsPEEK_161225/')

cache = np.loadtxt(results_path / 'long_term_data.txt', delimiter=',')
times = np.loadtxt(results_path / 'long_term_times.txt', delimiter=',') * 60

time_mark_8nA = times[np.argmin(np.abs(times - 39*60))]

proton_count = times * 25e-9 * 0.8874 / e
proton_count[:np.argmin(np.abs(times - 39*60))] *= 8/25

# --------------------------------------------------
# 5. Calculate the dose of individual pixels
# --------------------------------------------------
# For time until time mark ≈ 39 min 8 nA into setup
dose1 = dose_norm * time_mark_8nA * (8e-9 * 0.8874 / e)
# After that 25 nA into setup
dose2 = dose_norm * (times.max()-time_mark_8nA) * (25e-9 * 0.8874 / e)
dose_ges = (dose1 + dose2) * proton_density_distribution


# ----- Plot of the dose map with detector and wedge -----
fig, ax = plt.subplots()
extent = [0, 50, -25, 25]
im = ax.imshow(dose_ges.T/1e+6, cmap=cmap, extent=extent)
plt.colorbar(im, ax=ax, label="Dose per simulation pixel (MGy)")
ax.set_title("Dose per simulation pixel with diffuser")
ax.set_xlim(ax.get_xlim()), ax.set_ylim(ax.get_ylim())
shape.add_to_plot(0, 0.3, ax, add_angle=False, color='grey', alpha=0.5, edgecolor='k')
rect = matplotlib.patches.Rectangle((25-16, -0.5), 32, 0.5, linewidth=0, edgecolor="lime", facecolor="lime", alpha=0.25, label='Detector array')
ax.add_patch(rect)
rect = matplotlib.patches.Rectangle((25-16-0.25, 0), 32, 0.5, linewidth=0, edgecolor="lime", facecolor="lime", alpha=0.25)
ax.add_patch(rect)
ax.set_xlabel("Simulation y-axis (mm)")
ax.set_ylabel("Simulation x-axis (mm)")
ax.legend(fontsize=12)
format_save(save_path, "CumulativeDoseSim", save_format=save_format, legend=False)


# ----- Plot of the energy map with detector and wedge -----
fig, ax = plt.subplots()
extent = [0, 50, -25, 25]
im = ax.imshow(mean_energy.T, cmap=sns.color_palette("crest_r", as_cmap=True), extent=extent)
plt.colorbar(im, ax=ax, label="Mean proton energy per pixel (MeV)")
ax.set_title("Mean energy per simulation pixel with diffuser")
ax.set_xlim(ax.get_xlim()), ax.set_ylim(ax.get_ylim())
shape.add_to_plot(0, 0.3, ax, add_angle=False, color='grey', alpha=0.5, edgecolor='k')
rect = matplotlib.patches.Rectangle((25-16, -0.5), 32, 0.5, linewidth=0, edgecolor="lime", facecolor="lime", alpha=0.25, label='Detector array')
ax.add_patch(rect)
rect = matplotlib.patches.Rectangle((25-16-0.25, 0), 32, 0.5, linewidth=0, edgecolor="lime", facecolor="lime", alpha=0.25)
ax.add_patch(rect)
ax.set_xlabel("Simulation y-axis (mm)")
ax.set_ylabel("Simulation x-axis (mm)")
ax.legend(fontsize=12)
format_save(save_path, "MeanEnergyMap", save_format=save_format, legend=False)

# ----- Check alignment of detector and sim -----
fig, ax = plt.subplots()
ax2 = ax.twinx()

for i in range(128):
    print("Diode: ", i)
    position = 25-16.25+i*0.25
    print(f"Position: {position} mm")
    th = shape.calculate_value(position)
    print(f"Wedge thickness: {th:.2f} mm")
    sim_pixel = int(position/0.25)
    ax.plot(position, cache[0, i], marker='x', color='k')
    print(cache[0, i])
    ax2.plot(position, np.mean(dose_ges[sim_pixel, 98:102])/1e6, marker='o', color='r')
    print(f"Sim pixel: {sim_pixel}")
    print(np.mean(dose_ges[sim_pixel, 98:102])/1e6)

shape.add_to_plot(0, 0.3, ax, add_angle=True, color='grey', alpha=0.5, edgecolor='k')
ax.set_xlabel("Position (mm)")
ax.set_ylabel("Diode signal (nA)")
ax2.set_ylabel("Dose (MGy)")
format_save(save_path, "ArrayResponsevsDose", save_format=save_format, legend=False)

# ----- Check alignment of detector and sim -----
fig, ax = plt.subplots()
ax2 = ax.twinx()

for i in range(128):
    print("Diode: ", i)
    position = 25-16.25+i*0.25
    print(f"Position: {position} mm")
    th = shape.calculate_value(position)
    print(f"Wedge thickness: {th:.2f} mm")
    sim_pixel = int(position/0.25)
    ax.plot(i, cache[0, i], marker='x', color='k')
    print(cache[0, i])
    ax2.plot(i, np.mean(dose_ges[sim_pixel, 98:102])/1e6, marker='o', color='r')
    print(f"Sim pixel: {sim_pixel}")
    print(np.mean(dose_ges[sim_pixel, 98:102])/1e6)

ax.axvline(np.argmax(cache[0]), c='k')
# shape.add_to_plot(0, 0.3, ax, add_angle=True, color='grey', alpha=0.5, edgecolor='k')
ax.set_xlabel("Array Pixel")
ax.set_ylabel("Diode signal (nA)")
ax2.set_ylabel("Dose (MGy)")
format_save(save_path, "ArrayResponsevsDose_PixelPosition", save_format=save_format, legend=False)

# ----- Plot signal vs time -----
fig, ax = plt.subplots()
for i in range(128):
    if i in diodes_to_plot:
        print("Diode: ", i)
        position = 25-16.25+i*0.25
        print(f"Position: {position} mm")
        th = shape.calculate_value(position)
        print(f"Wedge thickness: {th:.2f} mm")
        sim_pixel = int(position/0.25)
        ax.plot(times/3600, cache[:, i], marker='x', color=diode_colour(i/128), ls='')
ax.set_xlabel("Irradiation time (h)")
ax.set_ylabel("Diode signal (nA)")
format_save(save_path, "SignalvsTime", save_format=save_format, legend=False)

# ----- Plot signal vs amount of protons -----
fig, ax = plt.subplots()

for i in range(128):
    if i in diodes_to_plot:
        print("Diode: ", i)
        position = 25 - 16.25 + i * 0.25
        print(f"Position: {position} mm")
        th = shape.calculate_value(position)
        print(f"Wedge thickness: {th:.2f} mm")
        sim_pixel = int(position / 0.25)
        ax.plot(proton_count*np.mean(proton_density_distribution[sim_pixel, 98:102])/1e12, cache[:, i], marker='x', color=diode_colour(i / 128), ls='')
ax.set_xlabel("Proton Count (1e12)")
ax.set_ylabel("Diode signal (nA)")
format_save(save_path, "SignalvsProtonCount", save_format=save_format, legend=False)


# ----- Plot signal vs dose -----
fig, ax = plt.subplots()

for i in range(128):
    if i in diodes_to_plot:
        print("Diode: ", i)
        position = 25 - 16.25 + i * 0.25
        print(f"Position: {position} mm")
        th = shape.calculate_value(position)
        print(f"Wedge thickness: {th:.2f} mm")
        sim_pixel = int(position / 0.25)
        dose_per_diode = proton_count*np.mean(dose_norm[sim_pixel-2:sim_pixel+2, 98:102]) * np.mean(proton_density_distribution[sim_pixel, 98:102]) /1e+6
        ax.plot(dose_per_diode, cache[:, i], marker='x', color=diode_colour(i / 128), ls='')
ax.set_xlabel("Total dose (MGy)")
ax.set_ylabel("Diode signal (nA)")
format_save(save_path, "SignalvsDose", save_format=save_format, legend=False)


# ----- Plot normed signal vs time -----
fig, ax = plt.subplots()

for i in range(128)[::-1]:
    if i in diodes_to_plot:
        print("Diode: ", i)
        position = 25-16.25+i*0.25
        print(f"Position: {position} mm")
        th = shape.calculate_value(position)
        print(f"Wedge thickness: {th:.2f} mm")
        sim_pixel = int(position/0.25)
        signal = cache[:, i] / cache[:, i].max()
        ax.plot(times/3600, signal, marker='x', color=diode_colour(i/128), ls='', alpha=0.5)
ax.set_xlabel("Irradiation time (h)")
ax.set_ylabel("Relative signal level of diode")
format_save(save_path, "NormedSignalvsTime", save_format=save_format, legend=False)

# ----- Plot signal vs amount of protons -----
fig, ax = plt.subplots()

for i in range(0, 128)[::-1]:
    if i in diodes_to_plot:
        print("Diode: ", i)
        position = 25 - 16.25 + i * 0.25
        print(f"Position: {position} mm")
        th = shape.calculate_value(position)
        print(f"Wedge thickness: {th:.2f} mm")
        sim_pixel = int(position / 0.25)
        signal = cache[:, i] / cache[:, i].max()
        ax.plot(proton_count*np.mean(proton_density_distribution[sim_pixel, 98:102])/1e12, signal, marker='x', color=diode_colour(i / 128), ls='', alpha=0.5)
ax.set_xlabel("Proton Count (1e12)")
ax.set_ylabel("Relative signal level of diode")
format_save(save_path, "NormedSignalvsProtonCount", save_format=save_format, legend=False)


# ----- Plot signal vs dose -----
fig, ax = plt.subplots()

for i in range(128)[::-1]:
    if i in diodes_to_plot:
        print("Diode: ", i)
        position = 25 - 16.25 + i * 0.25
        print(f"Position: {position} mm")
        th = shape.calculate_value(position)
        print(f"Wedge thickness: {th:.2f} mm")
        sim_pixel = int(position / 0.25)
        dose_per_diode = proton_count*np.mean(dose_norm[sim_pixel-2:sim_pixel+2, 98:102]) * np.mean(proton_density_distribution[sim_pixel, 98:102]) /1e+6
        signal = cache[:, i] / cache[:, i].max()
        ax.plot(dose_per_diode, signal, marker='x', color=diode_colour(i / 128), ls='', alpha=0.5)
ax.set_xlabel("Total dose (MGy)")
ax.set_ylabel("Relative signal level of diode")
format_save(save_path, "NormedSignalvsDose", save_format=save_format, legend=False)

# ----- Correction factor for degradation -----
correction_factor = []
for i in range(128):
    signal = cache[:, i] / cache[:, i].max()
    correction_factor.append(np.mean(signal[-20:]))

correction_factor = np.array(correction_factor)
np.save(results_path / 'compensation_factor.npy', correction_factor)
correction_factor /= correction_factor.mean()
np.save(results_path / 'correction_factor.npy', correction_factor)
print('Correction factor:')
print(correction_factor)

# ----- Dose estimate 2 out of array signal -----
array_signal = cache[0, :]
array_signal /= array_signal.max()

dose_per_diode = np.mean(dose_norm[:, 98:102], axis=1) * np.mean(proton_density_distribution[:, 98:102], axis=1) / 1e+6
array_signal *= dose_per_diode.max()

fig, ax = plt.subplots()

for i in range(128)[::-1]:
    if i in diodes_to_plot:
        print("Diode: ", i)
        position = 25 - 16.25 + i * 0.25
        print(f"Position: {position} mm")
        th = shape.calculate_value(position)
        print(f"Wedge thickness: {th:.2f} mm")
        sim_pixel = int(position / 0.25)
        dose_per_diode = proton_count*np.mean(dose_norm[sim_pixel-2:sim_pixel+2, 98:102]) * np.mean(proton_density_distribution[sim_pixel, 98:102]) /1e+6
        dose_per_diode = proton_count*array_signal[i]

        signal = cache[:, i] / cache[:, i].max()

        ax.plot(dose_per_diode, signal, marker='x', color=diode_colour(i / 128), ls='', alpha=0.5)
ax.set_xlabel("Total dose (MGy)")
ax.set_ylabel("Relative signal level of diode")
format_save(save_path, "NormedSignalvsDoseOutOfArraySignal", save_format=save_format, legend=False)

# '''
# ----- Plot signal vs dose per diode -----
plt.rcParams.update({
    "text.usetex": False,
})
plt.rcParams["font.family"] = ["Arial"]

def exp_decay(x, A, k, C):
    return C + A * np.exp(-k * x)

def r_squared(y, y_fit):
    ss_res = np.sum((y - y_fit)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    if ss_tot > 0:
        return 1 - ss_res/ss_tot
    else:
        return 1

def stretched_damage_model(x, A, k, beta):
    x = np.asarray(x, dtype=float)
    return 1.0 - A * (1.0 - np.exp(-np.power(np.clip(k * x, 0, None), beta)))


GLOBAL_DAMAGE_MODELS = {
    'shared_beta_only': {
        'shared_A': False,
        'description': 'Shared beta, individual A and k',
        'plot_kwargs': {'color': 'tab:blue', 'ls': '--', 'lw': 1.5},
    },
    'shared_beta_and_A': {
        'shared_A': True,
        'description': 'Shared beta and A, individual k',
        'plot_kwargs': {'color': 'tab:orange', 'ls': '-', 'lw': 1.5},
    },
    'shared_beta_individual_Ak': {
        'shared_A': False,
        'description': 'Alias of shared_beta_only for explicit individual A and k',
        'plot_kwargs': {'color': 'tab:green', 'ls': ':', 'lw': 1.5},
    },
}

# Add 'shared_beta_individual_Ak' here if you want to run the alias explicitly as a third comparison.
global_fit_modes = ['shared_beta_only', 'shared_beta_and_A']
primary_global_fit_mode = 'shared_beta_and_A'

GLOBAL_FIT_CONTROL = {
    'solver_max_points_per_curve': 250,
    'warmup_max_nfev': 400,
    'final_max_nfev': 1500,
    'solver_method': 'trf',
    'solver_jac': '2-point',
}


def normalize_relative_signal(signal):
    signal = np.asarray(signal, dtype=float)
    finite_mask = np.isfinite(signal)
    if not np.any(finite_mask):
        return signal
    first_idx = np.flatnonzero(finite_mask)[0]
    norm_reference = signal[first_idx]
    if np.isclose(norm_reference, 0):
        norm_reference = np.nanmax(signal[finite_mask])
    if np.isclose(norm_reference, 0):
        raise ValueError('Signal cannot be normalized because the reference value is zero.')
    return signal / norm_reference


def subsample_curve(x_data, y_data, max_points):
    if max_points is None or len(x_data) <= max_points:
        return x_data, y_data
    indices = np.linspace(0, len(x_data) - 1, max_points, dtype=int)
    indices = np.unique(indices)
    return x_data[indices], y_data[indices]


def build_irradiation_curve_datasets(signal_matrix, signal_threshold=0.1):
    datasets = []
    n_curves = signal_matrix.shape[1]
    for diode_idx in range(n_curves):
        position = 25 - 16.25 + diode_idx * 0.25
        sim_pixel = int(position / 0.25)
        local_dose = (
            proton_count
            * np.mean(dose_norm[sim_pixel - 2:sim_pixel + 2, 98:102])
            * np.mean(proton_density_distribution[sim_pixel, 98:102])
            / 1e+6
        )
        signal_full = normalize_relative_signal(signal_matrix[:, diode_idx])
        valid_mask = np.isfinite(local_dose) & np.isfinite(signal_full)
        fit_mask = valid_mask & (signal_full > signal_threshold)
        if np.count_nonzero(fit_mask) < 4:
            continue

        x_full = np.asarray(local_dose[valid_mask], dtype=float)
        y_full = np.asarray(signal_full[valid_mask], dtype=float)
        x_fit = np.asarray(local_dose[fit_mask], dtype=float)
        y_fit = np.asarray(signal_full[fit_mask], dtype=float)
        order_full = np.argsort(x_full)
        order_fit = np.argsort(x_fit)
        x_fit = x_fit[order_fit]
        y_fit = y_fit[order_fit]
        x_solver, y_solver = subsample_curve(
            x_fit,
            y_fit,
            GLOBAL_FIT_CONTROL['solver_max_points_per_curve'],
        )

        datasets.append({
            'curve_position': len(datasets),
            'diode_idx': diode_idx,
            'curve_label': f'Diode {diode_idx + 1}',
            'position_mm': position,
            'wedge_thickness_mm': shape.calculate_value(position),
            'mean_energy_mev': float(mean_energy_per_diode[diode_idx]),
            'median_energy_mev': float(median_energy_per_diode[diode_idx]),
            'x_full': x_full[order_full],
            'y_full': y_full[order_full],
            'x_fit': x_fit,
            'y_fit': y_fit,
            'x_solver': x_solver,
            'y_solver': y_solver,
            'color': diode_colour(diode_idx / n_curves),
        })
    return datasets


def build_global_damage_parameters(datasets, model_name):
    config = GLOBAL_DAMAGE_MODELS[model_name]
    n_curves = len(datasets)
    a_guess = np.array([np.clip(1.0 - dataset['y_fit'][-1], 1e-3, 0.95) for dataset in datasets])
    k_guess = np.array([1.0 / max(dataset['x_fit'].max() * 0.1, 1e-6) for dataset in datasets])
    beta_guess = 0.5

    if config['shared_A']:
        x0 = np.concatenate([[beta_guess, np.clip(np.mean(a_guess), 1e-3, 0.95)], k_guess])
        lower = np.concatenate([[1e-4, 0.0], np.full(n_curves, 1e-8)])
        upper = np.concatenate([[1.0, 1.0], np.full(n_curves, np.inf)])
    else:
        x0 = np.concatenate([[beta_guess], a_guess, k_guess])
        lower = np.concatenate([[1e-4], np.zeros(n_curves), np.full(n_curves, 1e-8)])
        upper = np.concatenate([[1.0], np.ones(n_curves), np.full(n_curves, np.inf)])
    return x0, lower, upper


def unpack_global_damage_parameters(parameter_vector, datasets, model_name):
    config = GLOBAL_DAMAGE_MODELS[model_name]
    n_curves = len(datasets)
    beta = float(parameter_vector[0])
    if config['shared_A']:
        shared_A = float(parameter_vector[1])
        curve_A = np.full(n_curves, shared_A)
        curve_k = np.asarray(parameter_vector[2:], dtype=float)
    else:
        shared_A = None
        curve_A = np.asarray(parameter_vector[1:1 + n_curves], dtype=float)
        curve_k = np.asarray(parameter_vector[1 + n_curves:], dtype=float)
    return {
        'beta': beta,
        'shared_A': shared_A,
        'curve_A': curve_A,
        'curve_k': curve_k,
    }


def global_damage_residuals(parameter_vector, datasets, model_name):
    unpacked = unpack_global_damage_parameters(parameter_vector, datasets, model_name)
    residuals = []
    for dataset, a_curve, k_curve in zip(datasets, unpacked['curve_A'], unpacked['curve_k']):
        y_model = stretched_damage_model(dataset['x_solver'], a_curve, k_curve, unpacked['beta'])
        residuals.append(y_model - dataset['y_solver'])
    return np.concatenate(residuals)


def estimate_least_squares_uncertainties(lsq_result):
    jacobian = lsq_result.jac
    m_points, n_params = jacobian.shape
    if m_points <= n_params:
        return np.full(n_params, np.nan)
    _, singular_values, vt = np.linalg.svd(jacobian, full_matrices=False)
    threshold = np.finfo(float).eps * max(jacobian.shape) * singular_values[0]
    valid = singular_values > threshold
    if not np.any(valid):
        return np.full(n_params, np.nan)
    covariance = (vt[valid].T / (singular_values[valid] ** 2)) @ vt[valid]
    variance_scale = 2 * lsq_result.cost / max(m_points - n_params, 1)
    covariance *= variance_scale
    diagonal = np.diag(covariance).copy()
    diagonal[diagonal < 0] = np.nan
    return np.sqrt(diagonal)


def run_global_damage_fit(datasets, model_name):
    x0, lower_bounds, upper_bounds = build_global_damage_parameters(datasets, model_name)
    print(f"Starting global fit '{model_name}' on {len(datasets)} curves "
          f"with up to {GLOBAL_FIT_CONTROL['solver_max_points_per_curve']} solver points per curve.")

    fit_start = time.perf_counter()
    lsq_result = least_squares(
        global_damage_residuals,
        x0=x0,
        bounds=(lower_bounds, upper_bounds),
        args=(datasets, model_name),
        method=GLOBAL_FIT_CONTROL['solver_method'],
        jac=GLOBAL_FIT_CONTROL['solver_jac'],
        max_nfev=GLOBAL_FIT_CONTROL['warmup_max_nfev'],
        x_scale='jac',
    )

    if (not lsq_result.success) or (lsq_result.nfev >= GLOBAL_FIT_CONTROL['warmup_max_nfev']):
        print(
            f"Warm-up fit for '{model_name}' stopped after {lsq_result.nfev} evaluations "
            f"and {time.perf_counter() - fit_start:.1f}s. Retrying once with a higher limit."
        )
        retry_start = time.perf_counter()
        lsq_result = least_squares(
            global_damage_residuals,
            x0=lsq_result.x,
            bounds=(lower_bounds, upper_bounds),
            args=(datasets, model_name),
            method=GLOBAL_FIT_CONTROL['solver_method'],
            jac=GLOBAL_FIT_CONTROL['solver_jac'],
            max_nfev=GLOBAL_FIT_CONTROL['final_max_nfev'],
            x_scale='jac',
        )
        print(
            f"Retry for '{model_name}' finished after {lsq_result.nfev} evaluations "
            f"and {time.perf_counter() - retry_start:.1f}s."
        )
    else:
        print(
            f"Fit '{model_name}' converged in {lsq_result.nfev} evaluations "
            f"and {time.perf_counter() - fit_start:.1f}s."
        )

    parameter_uncertainties = estimate_least_squares_uncertainties(lsq_result)
    unpacked = unpack_global_damage_parameters(lsq_result.x, datasets, model_name)
    unpacked_unc = unpack_global_damage_parameters(parameter_uncertainties, datasets, model_name)

    total_sse = 0.0
    total_points = 0
    curve_results = []
    for dataset, a_curve, k_curve, a_unc, k_unc in zip(
        datasets,
        unpacked['curve_A'],
        unpacked['curve_k'],
        unpacked_unc['curve_A'],
        unpacked_unc['curve_k'],
    ):
        y_model_fit = stretched_damage_model(dataset['x_fit'], a_curve, k_curve, unpacked['beta'])
        y_model_full = stretched_damage_model(dataset['x_full'], a_curve, k_curve, unpacked['beta'])
        residuals = y_model_fit - dataset['y_fit']
        sse = float(np.sum(residuals ** 2))
        total_sse += sse
        total_points += residuals.size
        curve_results.append({
            'curve_position': dataset['curve_position'],
            'diode_idx': dataset['diode_idx'],
            'curve_label': dataset['curve_label'],
            'mean_energy_mev': dataset['mean_energy_mev'],
            'median_energy_mev': dataset['median_energy_mev'],
            'x_fit': dataset['x_fit'],
            'y_fit': dataset['y_fit'],
            'x_full': dataset['x_full'],
            'y_full': dataset['y_full'],
            'y_model_fit': y_model_fit,
            'y_model_full': y_model_full,
            'A': float(a_curve),
            'A_unc': float(a_unc),
            'k': float(k_curve),
            'k_unc': float(k_unc),
            'beta': float(unpacked['beta']),
            'beta_unc': float(unpacked_unc['beta']),
            'rmse': float(np.sqrt(np.mean(residuals ** 2))),
            'r_squared': float(r_squared(dataset['y_fit'], y_model_fit)),
        })

    n_params = lsq_result.x.size
    global_rmse = float(np.sqrt(total_sse / max(total_points, 1)))
    global_reduced_chi2 = float(total_sse / max(total_points - n_params, 1))

    return {
        'model_name': model_name,
        'description': GLOBAL_DAMAGE_MODELS[model_name]['description'],
        'shared_A': GLOBAL_DAMAGE_MODELS[model_name]['shared_A'],
        'plot_kwargs': GLOBAL_DAMAGE_MODELS[model_name]['plot_kwargs'],
        'success': lsq_result.success,
        'message': lsq_result.message,
        'beta': float(unpacked['beta']),
        'beta_unc': float(unpacked_unc['beta']),
        'shared_A_value': None if unpacked['shared_A'] is None else float(unpacked['shared_A']),
        'shared_A_unc': None if unpacked_unc['shared_A'] is None else float(unpacked_unc['shared_A']),
        'curve_results': curve_results,
        'global_rmse': global_rmse,
        'global_reduced_chi2': global_reduced_chi2,
        'n_points': total_points,
        'n_params': n_params,
    }


def format_value_with_uncertainty(value, uncertainty):
    if np.isfinite(uncertainty):
        return f'{value:.4g} +/- {uncertainty:.2g}'
    return f'{value:.4g}'


def print_global_fit_summary(fit_result):
    print(f"\nGlobal stretched-damage fit: {fit_result['model_name']}")
    print(f"  Description: {fit_result['description']}")
    print(f"  Success: {fit_result['success']} ({fit_result['message']})")
    print(f"  Shared beta: {format_value_with_uncertainty(fit_result['beta'], fit_result['beta_unc'])}")
    if fit_result['shared_A']:
        print(f"  Shared A: {format_value_with_uncertainty(fit_result['shared_A_value'], fit_result['shared_A_unc'])}")
    print(f"  Global RMSE: {fit_result['global_rmse']:.5f}")
    print(f"  Global reduced chi2: {fit_result['global_reduced_chi2']:.5f}")

    header = f"{'Curve':<10} {'E_mean':>8} {'k':>18} {'A':>18} {'beta':>14} {'RMSE':>10}"
    print(header)
    print('-' * len(header))
    for curve_result in fit_result['curve_results']:
        a_text = 'shared' if fit_result['shared_A'] else format_value_with_uncertainty(curve_result['A'], curve_result['A_unc'])
        k_text = format_value_with_uncertainty(curve_result['k'], curve_result['k_unc'])
        beta_text = format_value_with_uncertainty(curve_result['beta'], curve_result['beta_unc'])
        print(
            f"{curve_result['curve_label']:<10} "
            f"{curve_result['mean_energy_mev']:>8.2f} "
            f"{k_text:>18} "
            f"{a_text:>18} "
            f"{beta_text:>14} "
            f"{curve_result['rmse']:>10.4f}"
        )


# Replace this call if you want to fit a different collection of irradiation curves.
# Each dataset entry only needs x_full, y_full, x_fit, y_fit, curve_label, and energy metadata.
irradiation_datasets = build_irradiation_curve_datasets(cache)
global_fit_results = {
    model_name: run_global_damage_fit(irradiation_datasets, model_name)
    for model_name in global_fit_modes
}

for fit_result in global_fit_results.values():
    print_global_fit_summary(fit_result)

for dataset in irradiation_datasets:
    fig, ax = plt.subplots()
    ax.plot(dataset['x_full'], dataset['y_full'], marker='x', color=dataset['color'], ls='', alpha=0.5, zorder=-1)
    for fit_result in global_fit_results.values():
        curve_result = fit_result['curve_results'][dataset['curve_position']]
        if fit_result['shared_A']:
            label = (
                f"{fit_result['model_name']}: "
                f"k={curve_result['k']:.3g}, "
                f"A={fit_result['shared_A_value']:.3g}, "
                f"beta={fit_result['beta']:.3g}, "
                f"RMSE={curve_result['rmse']:.4f}"
            )
        else:
            label = (
                f"{fit_result['model_name']}: "
                f"k={curve_result['k']:.3g}, "
                f"A={curve_result['A']:.3g}, "
                f"beta={fit_result['beta']:.3g}, "
                f"RMSE={curve_result['rmse']:.4f}"
            )
        ax.plot(curve_result['x_full'], curve_result['y_model_full'], label=label, **fit_result['plot_kwargs'])

    ax.set_xlabel("Total dose (MGy)")
    ax.set_ylabel("Relative signal level of diode")
    ax.set_title(
        f"{dataset['curve_label']} ({dataset['x_full'][-1]:.2f} MGy), "
        f"E_mean: {dataset['mean_energy_mev']:.2f} MeV"
    )
    ax.legend(fontsize=7)
    format_save(save_path/'DegCurvePerDiodeGlobal/', f"DegradationDiode{dataset['diode_idx']}", save_format=save_format, legend=False)

# '''
plt.rcParams.update({
    "text.usetex": True,
})
plt.rcParams["font.family"] = ["Latin Modern Roman"]


# ----- Extract signal decay at certain dose and plot vs energy -----
comp_dose = 1.3  # MGy
primary_fit_result = global_fit_results[primary_global_fit_mode]
fig, ax = plt.subplots()

for curve_result in primary_fit_result['curve_results']:
    if np.max(curve_result['x_full']) < comp_dose:
        continue
    x_ind = np.argmin(np.abs(comp_dose - curve_result['x_full']))
    ax.plot(curve_result['median_energy_mev'], curve_result['y_model_full'][x_ind] * 100, marker='x', c='k', alpha=1)

ax.set_xlabel("Median Proton Energy (MeV)")
ax.set_ylabel(f"Rest signal after {comp_dose}$\\,$MGy ($\\%$)")
format_save(save_path , f"DegradationvsEnergy", save_format=save_format, legend=False)

fig, ax = plt.subplots()

for curve_result in primary_fit_result['curve_results']:
    if np.max(curve_result['x_full']) < comp_dose:
        continue
    x_ind = np.argmin(np.abs(comp_dose - curve_result['x_full']))
    ax.plot(curve_result['mean_energy_mev'], curve_result['y_model_full'][x_ind] * 100, marker='x', c='k', alpha=1)

ax.set_xlabel("Mean Proton Energy (MeV)")
ax.set_ylabel(f"Rest signal after {comp_dose}$\\,$MGy ($\\%$)") 
format_save(save_path , f"DegradationvsMeanEnergy", save_format=save_format, legend=False)

fig, ax = plt.subplots()

for curve_result in primary_fit_result['curve_results']:
    yerr = curve_result['k_unc'] if np.isfinite(curve_result['k_unc']) else None
    ax.errorbar(
        curve_result['mean_energy_mev'],
        curve_result['k'],
        yerr=yerr,
        marker='x',
        color='k',
        ls='',
        alpha=1,
        capsize=2 if yerr is not None else 0,
    )

ax.set_xlabel("Mean Proton Energy (MeV)")
ax.set_ylabel("Damage-rate parameter $k$")
ax.set_ylim((0, 2.3))
format_save(save_path, "DamageRateKvsMeanEnergy", save_format=save_format, legend=False)

'''
plt.rcParams.update({
    "text.usetex": False,
})
plt.rcParams["font.family"] = ["Arial"]
deg_curves = []
for i in range(128):
    fig, ax = plt.subplots()
    print("Diode: ", i)
    position = 25 - 16.25 + i * 0.25
    print(f"Position: {position} mm")
    th = shape.calculate_value(position)
    print(f"Wedge thickness: {th:.2f} mm")
    sim_pixel = int(position / 0.25)
    dose_per_diode = proton_count * array_signal[i]
    signal = cache[:, i] / cache[:, i].max()

    # --- select first and last 1000 points ---
    x_data = dose_per_diode[signal > 0.1]
    # x_data = np.concatenate([dose_per_diode[:1000], dose_per_diode[-1000:]])
    y_data = signal[signal > 0.1]
    # y_data = np.concatenate([y_data[:1000], y_data[-1000:]])

    # --- 1. Single exponential ---
    p0 = [y_data.max(), 1 / (x_data.max() if x_data.max() > 0 else 1), y_data.min()]
    bounds = ([0, 0, 0], [1, np.inf, 1])
    popt1, _ = curve_fit(exp_decay, x_data, y_data, p0=p0, bounds=bounds, maxfev=10000)
    y_fit1 = exp_decay(dose_per_diode, *popt1)
    label = f"Single exp: A={popt1[0]:.3f}, k={popt1[1]:.3f}, C={popt1[2]:.3f}, R$^2$={r_squared(y_data, exp_decay(x_data, *popt1)):.4f}"
    ax.plot(dose_per_diode, y_fit1, ls='--', label=label)

    # --- 2. Double exponential ---
    A1_0 = (y_data[0] - y_data[-1]) * 0.6
    A2_0 = (y_data[0] - y_data[-1]) * 0.4
    k1_0 = 1 / (x_data.max() * 0.1 if x_data.max() > 0 else 1)
    k2_0 = 1 / (x_data.max() * 0.8 if x_data.max() > 0 else 1)
    C0 = y_data[-1]
    p0_double = [A1_0, k1_0, A2_0, k2_0, C0]
    bounds_double = ([0, 0, 0, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf])
    popt2, _ = curve_fit(double_exp, x_data, y_data, p0=p0_double, bounds=bounds_double, maxfev=10000)
    y_fit2 = double_exp(dose_per_diode, *popt2)
    label = (
        f"Double exp: A1={popt2[0]:.3f}, k1={popt2[1]:.3f}, A2={popt2[2]:.3f}, k2={popt2[3]:.3f}, C={popt2[4]:.3f}, R$^2$={r_squared(y_data, double_exp(x_data, *popt2)):.4f}")
    ax.plot(dose_per_diode, y_fit2, ls='--', label=label)

    # --- 3. Stretched exponential ---
    A0 = y_data[0] - y_data[-1]
    k0 = 1 / (x_data.max() * 0.1 if x_data.max() > 0 else 1)
    beta0 = 0.5
    C0 = y_data[-1]
    p0_stretched = [A0, k0, beta0, C0]
    bounds_stretched = ([0, 0, 0, -np.inf], [np.inf, np.inf, 1, np.inf])
    popt3, _ = curve_fit(stretched_exp, x_data, y_data, p0=p0_stretched, bounds=bounds_stretched, maxfev=10000)
    y_fit3 = stretched_exp(dose_per_diode, *popt3)
    label = (
        f"Stretched exp: A={popt3[0]:.3f}, k={popt3[1]:.3f}, beta={popt3[2]:.3f}, C={popt3[3]:.3f}, R$^2$={r_squared(y_data, stretched_exp(x_data, *popt3)):.4f}")
    ax.plot(dose_per_diode, y_fit3, ls='--', label=label)

    # --- 4. Exp + Lin decay ---
    A0 = y_data[0] - y_data[-1]
    k0 = 1 / (x_data.max() * 0.1 if x_data.max() > 0 else 1)
    m0 = (y_data[-1] - y_data[-5]) / (x_data[-1] - x_data[-5])
    m0 = m0 if np.inf > m0 >= 0 else 0
    C0 = y_data[-1]
    p0_stretched = [A0, k0, m0, C0]
    bounds_stretched = ([0, 0, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf])
    popt4, _ = curve_fit(explin_decay, x_data, y_data, p0=p0_stretched, bounds=bounds_stretched, maxfev=10000)
    y_fit4 = explin_decay(dose_per_diode, *popt4)
    label = (
        f"Exp+Lin: A={popt4[0]:.3f}, k={popt4[1]:.3f}, beta={popt4[2]:.3f}, C={popt4[3]:.3f}, R$^2$={r_squared(y_data, explin_decay(x_data, *popt4)):.4f}")
    ax.plot(dose_per_diode, y_fit3, ls='--', label=label)

    deg_curves.append([x_data, y_fit4])

    # --- plot original data ---
    ax.plot(dose_per_diode, signal, marker='x', color=diode_colour(i / 128), ls='', alpha=0.5, zorder=-1)

    ax.set_xlabel("Total dose (MGy)")
    ax.set_ylabel("Diode signal (nA)")
    ax.set_title(f'Diode {i+1} ({dose_per_diode[-1]:.2f} MGy), E_mean: {np.mean(median_energy[sim_pixel, 98:102]):.2f} MeV')
    ax.legend(fontsize=7)
    format_save(save_path/'DegCurvePerDiodeArrayScaled/', f"DegradationDiode{i}", save_format=save_format, legend=False)

plt.rcParams.update({
    "text.usetex": True,
})
plt.rcParams["font.family"] = ["Latin Modern Roman"]


# ----- Extract signal decay at certain dose and plot vs energy -----
comp_dose = 1.3  # MGy
deg_at_dose = []
fig, ax = plt.subplots()

for i, curve in enumerate(deg_curves):
    if np.max(curve[0]) < comp_dose:
        continue
    x_ind = np.argmin(np.abs(comp_dose-curve[0]))
    deg_at_dose.append(curve[1][x_ind])
    ax.plot(median_energy_per_diode[i], curve[1][x_ind]*100, marker='x', c='k', alpha=1)

ax.set_xlabel("Proton Energy (MeV)")
ax.set_ylabel(f"Rest signal after {comp_dose}$\\,$MGy ($\\%$)")
format_save(save_path , f"DegradationvsEnergyArrayScaled", save_format=save_format, legend=False)
'''
