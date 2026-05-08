from Consoles.StyleConsoles.Utils_ImageLoad import *
from EvaluationSoftware.simulation_connectors import *
from EvaluationSoftware.standard_processes import linearity_return
from scipy.optimize import least_squares

# --------------------------------------------------
# Result plots
# --------------------------------------------------
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black", "red", "yellow"])
diode_colour = sns.color_palette("icefire", as_cmap=True)

diodes_to_plot = [0, 50, 106]
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
sigma_energy_per_diode = []
for i in range(128):
    position = 25 - 16.25 + i * 0.25
    sim_pixel = int(position / 0.25)
    median_energy_per_diode.append(np.mean(median_energy[sim_pixel, 98:102]))
    sigma_energy_per_diode.append(np.mean(sigma_energy[sim_pixel, 98:102]))

mean_energy_per_diode = []
for i in range(128):
    position = 25 - 16.25 + i * 0.25
    sim_pixel = int(position / 0.25)
    mean_energy_per_diode.append(np.mean(mean_energy[sim_pixel, 98:102]))

median_energy_per_diode = np.array(median_energy_per_diode)
sigma_energy_per_diode = np.array(sigma_energy_per_diode)
mean_energy_per_diode = np.array(mean_energy_per_diode)

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

exp_data = np.loadtxt(results_path / 'long_term_data.txt', delimiter=',')
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



# --------------------------------------------------------------------------------------------------------------------
# Helpers and Models
# --------------------------------------------------------------------------------------------------------------------

from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerBase

class SigmaLegendItem:
    def __init__(self, color='r', alpha=0.2, lw=1.5):
        self.color = color
        self.alpha = alpha
        self.lw = lw


class HandlerSigmaLegendItem(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        color = orig_handle.color

        patch = Rectangle(
            (xdescent, ydescent + 0.15 * height),
            width,
            0.7 * height,
            facecolor=color,
            edgecolor='none',
            alpha=orig_handle.alpha,
            transform=trans,
        )

        line = Line2D(
            [xdescent, xdescent + width],
            [ydescent + 0.5 * height, ydescent + 0.5 * height],
            color=color,
            lw=orig_handle.lw,
            transform=trans,
        )

        return [patch, line]


def exp_decay(x, A, k, C):
    return C + A * np.exp(-k * x)


def explin_decay(x, A, k, m, C):
    return C + A * np.exp(-k * x) - m*x


def double_exp(x, A1, k1, A2, k2, C):
    """
    Double exponential decay:
    y(x) = C + A1*exp(-k1*x) + A2*exp(-k2*x)
    """
    return C + A1*np.exp(-k1*x) + A2*np.exp(-k2*x)


def stretched_exp(x, A, k, beta, C):
    """
    Stretched exponential decay (Kohlrausch):
    y(x) = C + A * exp(-(k*x)**beta)
    """
    return C + A * np.exp(-(k*x)**beta)


def r_squared(y, y_fit):
    ss_res = np.sum((y - y_fit)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    if ss_tot > 0:
        return 1 - ss_res/ss_tot
    else:
        return 1


GLOBAL_DAMAGE_FIT_CONTROL = {
    'signal_threshold': 0.1,
    'mask_half_window': 5,
    'solver_max_points_per_curve': 250,
    'solver_method': 'trf',
    'solver_jac': '2-point',
    'solver_max_nfev': 1500,
}


def stretched_damage_model(x, A, k, beta):
    x = np.asarray(x, dtype=float)
    return 1.0 - A * (1.0 - np.exp(-np.power(np.clip(k * x, 0, None), beta)))


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


def build_global_damage_datasets(signal_matrix):
    datasets = []
    signal_threshold = GLOBAL_DAMAGE_FIT_CONTROL['signal_threshold']
    mask_half_window = GLOBAL_DAMAGE_FIT_CONTROL['mask_half_window']
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
        base_mask = valid_mask & (signal_full > signal_threshold)
        support_mask = (
            np.convolve((~base_mask).astype(int), np.ones(2 * mask_half_window + 1, int), mode='same') == 0
        )
        fit_mask = base_mask & support_mask
        if np.count_nonzero(fit_mask) < 4:
            continue

        x_full = np.asarray(local_dose[valid_mask], dtype=float)
        y_full = np.asarray(signal_full[valid_mask], dtype=float)
        x_fit = np.asarray(local_dose[fit_mask], dtype=float)
        y_fit = np.asarray(signal_full[fit_mask], dtype=float)

        order_full = np.argsort(x_full)
        order_fit = np.argsort(x_fit)
        x_full = x_full[order_full]
        y_full = y_full[order_full]
        x_fit = x_fit[order_fit]
        y_fit = y_fit[order_fit]
        x_solver, y_solver = subsample_curve(
            x_fit,
            y_fit,
            GLOBAL_DAMAGE_FIT_CONTROL['solver_max_points_per_curve'],
        )

        datasets.append({
            'curve_position': len(datasets),
            'diode_idx': diode_idx,
            'mean_energy_mev': float(mean_energy_per_diode[diode_idx]),
            'x_full': x_full,
            'y_full': y_full,
            'x_fit': x_fit,
            'y_fit': y_fit,
            'x_solver': x_solver,
            'y_solver': y_solver,
            'color': ddc(td[diode_idx]) if 'td' in globals() else None,
        })

    return datasets


def _highest_dose_dataset(datasets):
    return max(datasets, key=lambda dataset: dataset['x_fit'][-1])


def _unpack_global_damage_parameters(parameter_vector, datasets):
    n_curves = len(datasets)
    beta = float(parameter_vector[0])
    shared_A = float(parameter_vector[1])
    curve_k = np.asarray(parameter_vector[2:2 + n_curves], dtype=float)
    return beta, shared_A, curve_k


def _global_damage_residuals(parameter_vector, datasets):
    beta, shared_A, curve_k = _unpack_global_damage_parameters(parameter_vector, datasets)
    residuals = []
    for dataset, k_value in zip(datasets, curve_k):
        y_model = stretched_damage_model(dataset['x_solver'], shared_A, k_value, beta)
        residuals.append(y_model - dataset['y_solver'])
    return np.concatenate(residuals)


def _fit_global_shared_A_beta_k(datasets, x0=None):
    n_curves = len(datasets)
    a_guess = np.array([np.clip(1.0 - dataset['y_fit'][-1], 1e-3, 0.95) for dataset in datasets])
    k_guess = np.array([1.0 / max(dataset['x_fit'].max() * 0.1, 1e-6) for dataset in datasets])
    beta_guess = 0.5
    shared_a_guess = float(np.clip(np.max(a_guess), 1e-3, 0.95))

    if x0 is None:
        x0 = np.concatenate([[beta_guess, shared_a_guess], k_guess])

    lower_bounds = np.concatenate([[1e-4, 1e-4], np.full(n_curves, 1e-8)])
    upper_bounds = np.concatenate([[1.0, 1.0], np.full(n_curves, np.inf)])
    return least_squares(
        _global_damage_residuals,
        x0=x0,
        bounds=(lower_bounds, upper_bounds),
        args=(datasets,),
        method=GLOBAL_DAMAGE_FIT_CONTROL['solver_method'],
        jac=GLOBAL_DAMAGE_FIT_CONTROL['solver_jac'],
        max_nfev=GLOBAL_DAMAGE_FIT_CONTROL['solver_max_nfev'],
        x_scale='jac',
    )


def _highest_dose_curve_residuals(parameter_vector, dataset, beta):
    shared_A, k_value = parameter_vector
    return stretched_damage_model(dataset['x_solver'], shared_A, k_value, beta) - dataset['y_solver']


def _fit_highest_dose_curve_for_A(dataset, beta, x0=None):
    a_guess = np.clip(1.0 - dataset['y_fit'][-1], 1e-3, 0.95)
    k_guess = 1.0 / max(dataset['x_fit'].max() * 0.1, 1e-6)
    if x0 is None:
        x0 = np.array([a_guess, k_guess], dtype=float)

    lower_bounds = np.array([1e-4, 1e-8], dtype=float)
    upper_bounds = np.array([1.0, np.inf], dtype=float)
    return least_squares(
        _highest_dose_curve_residuals,
        x0=x0,
        bounds=(lower_bounds, upper_bounds),
        args=(dataset, beta),
        method=GLOBAL_DAMAGE_FIT_CONTROL['solver_method'],
        jac=GLOBAL_DAMAGE_FIT_CONTROL['solver_jac'],
        max_nfev=GLOBAL_DAMAGE_FIT_CONTROL['solver_max_nfev'],
        x_scale='jac',
    )


def _fixed_A_global_damage_residuals(parameter_vector, datasets, shared_A):
    beta = float(parameter_vector[0])
    curve_k = np.asarray(parameter_vector[1:], dtype=float)
    residuals = []
    for dataset, k_value in zip(datasets, curve_k):
        y_model = stretched_damage_model(dataset['x_solver'], shared_A, k_value, beta)
        residuals.append(y_model - dataset['y_solver'])
    return np.concatenate(residuals)


def _fit_global_fixed_A_beta_k(datasets, shared_A, x0=None):
    n_curves = len(datasets)
    k_guess = np.array([1.0 / max(dataset['x_fit'].max() * 0.1, 1e-6) for dataset in datasets])
    beta_guess = 0.5

    if x0 is None:
        x0 = np.concatenate([[beta_guess], k_guess])

    lower_bounds = np.concatenate([[1e-4], np.full(n_curves, 1e-8)])
    upper_bounds = np.concatenate([[1.0], np.full(n_curves, np.inf)])
    return least_squares(
        _fixed_A_global_damage_residuals,
        x0=x0,
        bounds=(lower_bounds, upper_bounds),
        args=(datasets, shared_A),
        method=GLOBAL_DAMAGE_FIT_CONTROL['solver_method'],
        jac=GLOBAL_DAMAGE_FIT_CONTROL['solver_jac'],
        max_nfev=GLOBAL_DAMAGE_FIT_CONTROL['solver_max_nfev'],
        x_scale='jac',
    )


def run_staged_global_damage_fit(datasets):
    stage1_result = _fit_global_shared_A_beta_k(datasets)
    stage1_beta, stage1_shared_A, stage1_curve_k = _unpack_global_damage_parameters(stage1_result.x, datasets)

    highest_dose_dataset = _highest_dose_dataset(datasets)
    highest_curve_position = highest_dose_dataset['curve_position']
    stage2_x0 = np.array([stage1_shared_A, stage1_curve_k[highest_curve_position]], dtype=float)
    stage2_result = _fit_highest_dose_curve_for_A(highest_dose_dataset, stage1_beta, x0=stage2_x0)
    anchored_A = float(stage2_result.x[0])
    anchored_k_highest_dose = float(stage2_result.x[1])

    stage3_x0 = np.concatenate([[stage1_beta], stage1_curve_k])
    stage3_x0[1 + highest_curve_position] = anchored_k_highest_dose
    stage3_result = _fit_global_fixed_A_beta_k(datasets, anchored_A, x0=stage3_x0)

    final_beta = float(stage3_result.x[0])
    final_curve_k = np.asarray(stage3_result.x[1:], dtype=float)
    curve_results = []
    for dataset, k_value in zip(datasets, final_curve_k):
        y_model_full = stretched_damage_model(dataset['x_full'], anchored_A, k_value, final_beta)
        y_model_fit = stretched_damage_model(dataset['x_fit'], anchored_A, k_value, final_beta)
        curve_results.append({
            'curve_position': dataset['curve_position'],
            'diode_idx': dataset['diode_idx'],
            'mean_energy_mev': dataset['mean_energy_mev'],
            'x_full': dataset['x_full'],
            'y_full': dataset['y_full'],
            'x_fit': dataset['x_fit'],
            'y_fit': dataset['y_fit'],
            'y_model_full': y_model_full,
            'y_model_fit': y_model_fit,
            'k': float(k_value),
            'r_squared': float(r_squared(dataset['y_fit'], y_model_fit)),
        })

    return {
        'success': bool(stage1_result.success and stage2_result.success and stage3_result.success),
        'message': (
            f"stage1={stage1_result.status}, "
            f"stage2={stage2_result.status}, "
            f"stage3={stage3_result.status}"
        ),
        'beta': final_beta,
        'shared_A': anchored_A,
        'highest_dose_diode_idx': highest_dose_dataset['diode_idx'],
        'curve_results': curve_results,
        'fit_strategy': 'stage1 shared A,beta,k_i; stage2 highest-dose A|beta_fixed; stage3 shared beta,k_i|A_fixed',
        'stage1': {
            'beta': stage1_beta,
            'shared_A': stage1_shared_A,
            'curve_k': stage1_curve_k,
            'success': bool(stage1_result.success),
        },
        'stage2': {
            'beta_fixed': stage1_beta,
            'shared_A': anchored_A,
            'highest_dose_k': anchored_k_highest_dose,
            'success': bool(stage2_result.success),
        },
        'stage3': {
            'beta': final_beta,
            'shared_A_fixed': anchored_A,
            'success': bool(stage3_result.success),
        },
    }


# --------------------------------------------------------------------------------------------------------------------
# Linearity
# --------------------------------------------------------------------------------------------------------------------
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_171225/')

before = 'exp1_25MeV_at_Control_'
after = 'exp5_25MeV_at_Control_'
mid = 'exp3_25MeV_at_Control_'

mapping = Path('../../Files/mapping.xlsx')
data = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:])-1 for k in data['direction_2']]
readout, position_parser, voltage_parser, current_parser = (
    lambda x, y: ams_2line_fast_avg(x, y, channel_assignment=channel_assignment),
    standard_position,
    set_voltage,
    current6
)
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
A.scale = 'pico'

def reorder_diodes(a):
    """
    Interleave diode axis (size 128):
    [0..63, 64..127] → [64,0,65,1,...]
    Works for:
    - (128,)
    - (N, 128)
    - (128, M)
    - (N, 128, M), etc.
    Leaves arrays without a 128-axis unchanged.
    """
    if not hasattr(a, "shape"):
        return a
    for axis, size in enumerate(a.shape):
        if size == 128:
            if size % 2 != 0:
                raise ValueError(f"Diode axis not even: {a.shape}")
            # --- 1D special case ---
            if a.ndim == 1:
                return (
                    a.reshape(2, -1)[::-1]
                     .T
                     .reshape(-1)
                )
            # --- general case ---
            a_swapped = np.moveaxis(a, axis, 0)  # bring diode axis to front
            a_reordered = (
                a_swapped.reshape(2, -1, *a_swapped.shape[1:])
                         [::-1, ...]
                         .transpose(1, 0, *range(2, a_swapped.ndim + 1))
                         .reshape(128, *a_swapped.shape[1:])
            )
            return np.moveaxis(a_reordered, 0, axis)
    return a


def process_linearity(path, dataset):
    currents, fit_currents, *rest = linearity_return(
        path, dataset, dark, A, per_diode=True
    )
    rest = [reorder_diodes(np.array(a)) for a in rest]
    return (currents, fit_currents, *rest)
(currents,  fit_currents,  signal,  fit,  std,  fit_std,  fit_r2,  std_r2)  = process_linearity(folder_path, before)
(currents2, fit_currents2, signal2, fit2, std2, fit_std2, fit_r22, std_r22) = process_linearity(folder_path, after)
(currents3, fit_currents3, signal3, fit3, std3, fit_std3, fit_r23, std_r23) = process_linearity(folder_path, mid)
print(np.shape(currents), np.shape(fit_currents), np.shape(signal), np.shape(fit), np.shape(std), np.shape(fit_std), np.shape(fit_r2), np.shape(std_r2))

currents = currents / (np.pi * (1.8)**2) * 1e3
fit_currents = fit_currents / (np.pi * (1.8)**2) * 1e3
currents2 = currents2 / (np.pi * (1.8)**2) * 1e3
fit_currents2 = fit_currents2 / (np.pi * (1.8)**2) * 1e3
currents3 = currents3 / (np.pi * (1.8)**2) * 1e3
fit_currents3 = fit_currents3 / (np.pi * (1.8)**2) * 1e3
