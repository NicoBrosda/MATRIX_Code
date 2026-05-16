import matplotlib
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
import numpy as np
import csv
from copy import deepcopy
from matplotlib.collections import LineCollection

from Consoles.StyleConsoles.Utils_ImageLoad import *
from EvaluationSoftware.simulation_connectors import *
from Consoles.Consoles_13_202511.PasteIn_DosePerDiode import *

linearity_signal_before = np.array(signal, copy=True)
linearity_signal_after = np.array(signal2, copy=True)
linearity_signal_middle = np.array(signal3, copy=True)

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

dpi = 300
save_format = 'png'

plot_size = (18 * cm, 3 / 2 * 18 / 1.2419 * cm)
# Setup of the final figure
# Structure: Ax1 Logo Line Super Res - Ax2 Logo Gaf - Ax3 BeamShape 2-Line - Ax4 BeamShape Gaf -
# Ax5 Logo Overlay - Ax6 Beam Overlay
fig, [[ax1, ax2], [ax3, ax4], [ax5, ax6]] = plt.subplots(3, 2, figsize=plot_size)
# Adjust spacing: left, right, bottom, top, wspace, hspace
fig.subplots_adjust(wspace=0.35, hspace=0.25)
# Axis limits
y_limits = []
x_limits = []
zero_scale = True
intensity_limits = None
intensity_limitsg = [0, 1]

energies = np.array(mean_energy_per_diode)
energy_cmap = sns.color_palette("crest_r", as_cmap=True)
energy_colormapper = lambda energy: color_mapper(energy, 0, np.max(energies))
energy_color = lambda energy: energy_cmap(energy_colormapper(energy))

# --- Precompute total dose per diode for the colour scale ---
shape = LineShape([[-4, 2.14], [36, 5.64]], distance_mode=False)
td = np.zeros(128)
for i in range(128):
    position = 25 - 16.25 + i * 0.25
    sim_pixel = int(position / 0.25)
    td[i] = (
        proton_count[-1]
        * np.mean(dose_norm[sim_pixel - 2:sim_pixel + 2, 98:102])
        * np.mean(proton_density_distribution[sim_pixel, 98:102])
        / 1e+6
    )  # in MGy

dose_cmap = sns.color_palette("flare_r", as_cmap=True)
# dose_cmap = mcolors.LinearSegmentedColormap.from_list("dose", dose_cmap(np.linspace(0.1, 0.9, 256)))
dose_min = np.min(td[td > 0])
dose_max = np.max(td)
dose_colormapper = lambda d: color_mapper(d, 1, dose_max)
dose_color = lambda d: dose_cmap(dose_colormapper(d))
# Convenience helpers for the dose-based colour mapping
ddc = lambda dose: dose_color(dose)
diode_dose_color = lambda idx: dose_color(td[idx])
fit_curve_color = '#0B2545'
fit_curve_pe = [pe.Stroke(linewidth=3.0, foreground=(1.0, 1.0, 1.0, 0.85)), pe.Normal()]

# ------------------------------------------------------------------------------------------------------------------
# Ax1: Response + Simulation + Proton energy vs wedge (response vs energy as inset?)
# ------------------------------------------------------------------------------------------------------------------
ax = ax1
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
energy_meas = [25]
measurements += ['25MeV_at_Control_bis_1nA.csv']
dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_161225/')
dark_paths_array1 = ['dark_current']
norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_260325/')
norm_array1 = ['exp7_norm1,9V_']
cache = []
comp_list = np.array(energy_meas)
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
cache_max = np.max(cache[0])
for i, curve in enumerate(cache[0]):
    if i == 0:
        ax.plot(0.25*i, curve / cache_max, color=ddc(td[i]), marker='o', markersize=3, ls='-', zorder=3, label=f'Experiment')

    ax.plot(0.25*i, curve / cache_max, color=ddc(td[i]), marker='o', markersize=3, ls='-', zorder=3)
ax.axvline([0.25*i for i in range(len(cache[0]))][np.argmax(cache[0] / cache_max)], color=ddc(td[np.argmax(cache[0] / cache_max)]), zorder=2, alpha=1, lw=1.5)

x = 0
start, end = 9 - 0.25, 41 - 0.25
crit = f'1e+085degActiveLayerSortieAir50PEEK_param'
param_list = [24.92]
max_line_cache = 0
for j, param in enumerate(param_list):
    data_cache, line_cache, line_std_cache = get_sim(crit, param=param)
    line_cache = line_cache[int(start/0.25):int(end/0.25)]
    max_line_cache = max(np.max(line_cache), max_line_cache)
for j, param in enumerate(param_list):
    data_cache, line_cache, line_std_cache = get_sim(crit, param=param)
    line_cache = line_cache[int(start/0.25):int(end/0.25)]
    if j == 0:
        ax.plot([0.25*i+x for i in range(len(line_cache))], line_cache / max_line_cache, color='k', ls='--',  lw=1.5, alpha=1, zorder=4, label='GATE')
    else:
        ax.plot([0.25*i+x for i in range(len(line_cache))], line_cache / max_line_cache, color='k', ls='--',  lw=1.5, alpha=1, zorder=4)

    ax.axvline([0.25*i+x for i in range(len(line_cache))][np.argmax(line_cache / np.max(line_cache))], color='k', ls='--', zorder=2, alpha=1, lw=1.2)

ax_r = ax.twinx()
x_vals = [0.25 * i + x for i in range(len(line_cache))]
ax_r.plot(x_vals, mean_energy_per_diode, color='r')
ax_r.fill_between(
    x_vals,
    mean_energy_per_diode - sigma_energy_per_diode,
    mean_energy_per_diode + sigma_energy_per_diode,
    color='r',
    alpha=0.2
)
# ax_r.set_ylim(0, ax_r.get_ylim()[1])

ax.set_xlim(ax.get_xlim()), ax.set_ylim(ax.get_ylim())

shape = LineShape([[-4, 2.14], [36, 5.64]], distance_mode=False)
shape.print_shape()
shape.add_to_plot(0.0, 0.28, fs=14, color='grey', alpha=1, zorder=-1, edgecolor='k', add_angle=True, ax=ax)

for j, param in enumerate(param_list):
    data_cache, line_cache, line_std_cache = get_sim(crit, param=param)
    line_cache = line_cache[int(start/0.25):int(end/0.25)]

ax.set_xlim(ax.get_xlim())
ax.set_ylim(ax.get_ylim())

handles, labels = ax.get_legend_handles_labels()
sigma_item = SigmaLegendItem(color='r', alpha=0.2, lw=1.5)
handles.append(sigma_item)
labels.append('E$_\\mathrm{Proton}$')
ax.legend(handles, labels, handler_map={SigmaLegendItem: HandlerSigmaLegendItem()}, loc='lower center')

ax.set_xlabel(r'Position along wedge (mm)')
ax.set_ylabel(r'Normalized signal/simulated E$_\mathrm{dep}$')
ax_r.set_ylabel(f'Simulated proton energy (MeV)')
ax_r.tick_params(axis='y', colors='red')
ax_r.yaxis.label.set_color('red')
ax_r.spines['right'].set_color('red')

# ------------------------------------------------------------------------------------------------------------------
# Ax2: Signal vs proton count (inset indication of which signal channels were picked)
# ------------------------------------------------------------------------------------------------------------------
ax = ax2
ax2_curve_data = {}

for i in range(0, 128)[::-1]:
    if i in diodes_to_plot:
        position = 25 - 16.25 + i * 0.25
        th = shape.calculate_value(position)
        sim_pixel = int(position / 0.25)
        long_term_signal = exp_data[:, i] / exp_data[:, i].max()

        n = 5
        base = long_term_signal > 0.1
        mask = base & (np.convolve(~base, np.ones(2 * n + 1, int), mode='same') == 0)
        # kernel = np.ones(5, dtype=int)  # Punkt selbst + 2 Nachbarn links/rechts
        # mask = np.convolve(base_mask.astype(int), kernel, mode='same') > 0

        x_data = (proton_count*np.mean(proton_density_distribution[sim_pixel, 98:102])/1e12)[mask]
        y_data = long_term_signal[mask]
        ax2_curve_data[i] = (x_data, y_data)
        ax.plot(x_data, y_data, marker='.', color=ddc(td[i]), ls='', alpha=0.5)

ax.set_xlim(ax.get_xlim()), ax.set_ylim(0, ax.get_ylim()[1])

ax.set_xlabel(r'Number of protons per pixel (1e12)')
ax.set_ylabel(f'Normalized pixel response')

# Manual annotation targets for panel (b), given as (x, y) in axis data coordinates.
drop_events = [
    (0.295, 0.3),
    (0.365, 0.31),
    (0.437, 0.32),
]

if drop_events:
    drop_points = drop_events
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_span = x_max - x_min
    y_span = y_max - y_min

    label_x = float(np.clip(np.mean([point[0] for point in drop_points]), x_min + 0.34 * x_span, x_min + 0.68 * x_span))
    label_y = y_min + 0.08 * y_span
    arrow_tail_y = label_y + 0.17 * y_span
    arrow_tail_offsets = np.linspace(-0.09 * x_span, 0.09 * x_span, len(drop_points))
    arrow_tail_rise = np.linspace(0.00, 0.04 * y_span, len(drop_points))

    ax.text(
        *transform_axis_to_data_coordinates(ax2, [0.5, 0.1]),
        'Readjustments of beam current',
        fontsize=9,
        color='0.25',
        ha='center',
        va='top',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8),
        zorder=6,
    )
    for (drop_x, drop_y), x_offset, y_offset in zip(drop_points, arrow_tail_offsets, arrow_tail_rise):
        ax.annotate(
            '',
            xy=(drop_x, drop_y),
            xytext=(label_x + x_offset, 0.1),
            arrowprops=dict(arrowstyle='->', lw=1.0, color='0.35', shrinkA=0, shrinkB=2),
            zorder=-1,
        )

signal_x = np.array([0.25 * i for i in range(cache.shape[1])])
signal_y = cache[0] / np.max(cache[0])
x_pdensity = np.array([0.25 * i for i in range(cache.shape[1]+32)]) - 16*0.25
proton_density_line = np.array([
    np.mean(proton_density_distribution[int((25 - 16.25 - 16*0.25 + i * 0.25) / 0.25), 98:102])
    for i in range(cache.shape[1]+32)
], dtype=float)
if np.max(proton_density_line) > 0:
    proton_density_line = proton_density_line / np.max(proton_density_line)


def add_wedge_signal_symbol(parent_ax, anchor_x, anchor_y, width_frac, height_frac,
                            show_dose_scale=False, scale_gap_frac=0.03, scale_width_frac=0.06,
                            show_proton_density=False):
    fig.canvas.draw()
    parent_bbox = parent_ax.get_position()
    inset_left = parent_bbox.x0 + anchor_x * parent_bbox.width
    inset_bottom = parent_bbox.y0 + anchor_y * parent_bbox.height
    inset_width = width_frac * parent_bbox.width
    inset_height = height_frac * parent_bbox.height

    symbol_ax = fig.add_axes([inset_left, inset_bottom, inset_width, inset_height])
    inset_wedge_ymin = 0.0
    inset_wedge_ymax = 0.28
    shape_inset = LineShape([[-4, 2.14], [36, 5.64]], distance_mode=False)
    shape_inset.add_to_plot(
        inset_wedge_ymin,
        inset_wedge_ymax,
        fs=14,
        color='grey',
        alpha=1,
        zorder=-1,
        edgecolor='k',
        add_angle=False,
        ax=symbol_ax,
    )

    points = np.array([signal_x, signal_y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors_per_segment = [diode_dose_color(i) for i in range(len(segments))]
    symbol_ax.add_collection(LineCollection(segments, colors=colors_per_segment, linewidths=1.2, zorder=3))

    symbol_ax.axis('off')
    symbol_ax.set_xlim(ax1.get_xlim())
    symbol_ax.set_ylim(ax1.get_ylim())

    if show_proton_density:
        wedge_bottom_data = transform_axis_to_data_coordinates(symbol_ax, [0.5, inset_wedge_ymin])[1]
        density_y = wedge_bottom_data + proton_density_line * (np.max(signal_y) - wedge_bottom_data)
        symbol_ax.plot(
            x_pdensity,
            density_y,
            ls='--',
            lw=1.1,
            color='0.2',
            zorder=2,
        )
        density_peak_idx = int(np.argmax(proton_density_line))
        density_peak_x = np.min(signal_x)
        density_peak_y = density_y[16]
        symbol_ax.annotate(
            'Proton density',
            xy=(density_peak_x, density_peak_y),
            xytext=transform_axis_to_data_coordinates(symbol_ax, [0.2, 0.97]),
            fontsize=9,
            color='0.2',
            ha='right',
            va='top',
            arrowprops=dict(arrowstyle='->', lw=0.9, color='0.2'),
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.75),
            zorder=4,
        )

    for ch_idx in diodes_to_plot:
        position = ch_idx * 0.25
        signal_val = signal_y[ch_idx] if ch_idx < len(signal_y) else signal_y[-1]
        wedge_bottom_data = transform_axis_to_data_coordinates(symbol_ax, [0.5, inset_wedge_ymin])[1]
        y_span = symbol_ax.get_ylim()[1] - symbol_ax.get_ylim()[0]
        arrow_tail_y = wedge_bottom_data - 0.10 * y_span
        symbol_ax.annotate(
            '',
            xy=(position, signal_val * 0.9),
            xytext=(position, arrow_tail_y),
            arrowprops=dict(arrowstyle='->', lw=1.5, color=diode_dose_color(ch_idx)),
            annotation_clip=False,
        )
        symbol_ax.plot(position, signal_val, marker='x', color=diode_dose_color(ch_idx), ls='', alpha=0.5)

    if not show_dose_scale:
        return symbol_ax

    scale_left = inset_left + inset_width + scale_gap_frac * parent_bbox.width
    scale_width = scale_width_frac * parent_bbox.width
    scale_ax = fig.add_axes([scale_left, inset_bottom, scale_width, inset_height])
    dose_norm_for_cbar = matplotlib.colors.Normalize(vmin=1.0, vmax=dose_max)
    dose_mappable = matplotlib.cm.ScalarMappable(norm=dose_norm_for_cbar, cmap=dose_cmap)
    dose_mappable.set_array([])
    cbar = fig.colorbar(dose_mappable, cax=scale_ax, orientation='vertical', extend='min')
    cbar.set_ticks([1.0, dose_max])
    cbar.set_ticklabels(['1 MGy', f'{dose_max:.1f} MGy'])
    cbar.ax.yaxis.tick_right()
    cbar.ax.tick_params(axis='y', length=0, pad=1, labelsize=9)
    return symbol_ax, scale_ax


ax2_symbol_ax = add_wedge_signal_symbol(ax2, 0.5, 0.6, 0.35, 0.35, show_proton_density=True)
# ------------------------------------------------------------------------------------------------------------------
# Ax4: Signals vs total diode dose + fit and model specification
# ------------------------------------------------------------------------------------------------------------------
ax = ax4

# Minimum total-dose threshold for channels included in the staged global degradation fit.
# Units: MGy
dose_reference_mgy = 1.0

irradiation_datasets_all = build_global_damage_datasets(exp_data)
irradiation_datasets = [
    dataset for dataset in irradiation_datasets_all
    if td[dataset['diode_idx']] >= dose_reference_mgy
]
excluded_diode_indices = sorted(
    dataset['diode_idx'] for dataset in irradiation_datasets_all
    if td[dataset['diode_idx']] < dose_reference_mgy
)

if not irradiation_datasets:
    raise RuntimeError(
        f"No channels pass dose_reference_mgy={dose_reference_mgy:.3f} MGy. "
        "Lower the threshold or check dose estimation."
    )

global_damage_fit = run_staged_global_damage_fit(irradiation_datasets)
deg_curve_map = {curve_result['diode_idx']: curve_result for curve_result in global_damage_fit['curve_results']}


def _rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _r_squared(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return np.nan
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 0:
        return np.nan
    return float(1.0 - ss_res / ss_tot)

print(
    "Axis 4 global damage fit:",
    f"success={global_damage_fit['success']}",
    f"beta={global_damage_fit['beta']:.4f}",
    f"shared_A={global_damage_fit['shared_A']:.4f}",
    f"highest_dose_diode={global_damage_fit['highest_dose_diode_idx'] + 1}",
    f"dose_reference_mgy={dose_reference_mgy:.3f}",
    f"n_fit_channels={len(irradiation_datasets)}",
    f"n_excluded_channels={len(excluded_diode_indices)}",
    f"strategy={global_damage_fit['fit_strategy']}",
)
if excluded_diode_indices:
    print(
        "Axis 4 excluded channels (dose below threshold):",
        [diode_idx + 1 for diode_idx in excluded_diode_indices],
    )

all_y_true = []
all_y_pred = []
n_channels_total = int(exp_data.shape[1])
nrmse_pct_per_channel = np.full(n_channels_total, np.nan, dtype=float)
print("Axis 4 per-channel fit quality (on fitted points):")
for curve_result in sorted(global_damage_fit['curve_results'], key=lambda item: item['diode_idx']):
    y_true = np.asarray(curve_result['y_fit'], dtype=float)
    y_pred = np.asarray(curve_result['y_model_fit'], dtype=float)
    if y_true.size == 0:
        continue

    curve_rmse = _rmse(y_true, y_pred)
    y_span = float(np.ptp(y_true))
    curve_nrmse_pct = np.nan if y_span <= 0 else float(100.0 * curve_rmse / y_span)
    curve_r2 = curve_result.get('r_squared', _r_squared(y_true, y_pred))
    nrmse_pct_per_channel[curve_result['diode_idx']] = curve_nrmse_pct

    all_y_true.append(y_true)
    all_y_pred.append(y_pred)

    print(
        f"  diode={curve_result['diode_idx'] + 1:3d}",
        f"k={curve_result['k']:.6g}",
        f"RMSE={curve_rmse:.6f}",
        f"NRMSE={curve_nrmse_pct:.2f}%",
        f"R2={curve_r2:.5f}",
    )

valid_nrmse_mask = np.isfinite(nrmse_pct_per_channel)
if np.any(valid_nrmse_mask):
    valid_nrmse = nrmse_pct_per_channel[valid_nrmse_mask]
    q25 = float(np.percentile(valid_nrmse, 25))
    q75 = float(np.percentile(valid_nrmse, 75))
    nrmse_mean = float(np.mean(valid_nrmse))
    nrmse_median = float(np.median(valid_nrmse))
    nrmse_std = float(np.std(valid_nrmse))
    nrmse_iqr = float(q75 - q25)
    nrmse_min = float(np.min(valid_nrmse))
    nrmse_max = float(np.max(valid_nrmse))
    nrmse_range = float(nrmse_max - nrmse_min)
    best_channel_idx = int(np.nanargmin(nrmse_pct_per_channel))
    worst_channel_idx = int(np.nanargmax(nrmse_pct_per_channel))
    print(
        "Axis 4 NRMSE summary (%):",
        f"mean={nrmse_mean:.4f}",
        f"median={nrmse_median:.4f}",
        f"std={nrmse_std:.4f}",
        f"IQR={nrmse_iqr:.4f}",
        f"min={nrmse_min:.4f}",
        f"max={nrmse_max:.4f}",
        f"range={nrmse_range:.4f}",
        f"n_valid={valid_nrmse.size}",
        f"n_total={n_channels_total}",
    )
    print(
        "Axis 4 NRMSE best/worst channel:",
        f"best={best_channel_idx + 1} (NRMSE={nrmse_pct_per_channel[best_channel_idx]:.4f}%)",
        f"worst={worst_channel_idx + 1} (NRMSE={nrmse_pct_per_channel[worst_channel_idx]:.4f}%)",
    )
else:
    print(
        "Axis 4 NRMSE summary: no valid channel NRMSE values available "
        f"(n_total={n_channels_total})."
    )

if all_y_true:
    global_y_true = np.concatenate(all_y_true)
    global_y_pred = np.concatenate(all_y_pred)
    global_rmse = _rmse(global_y_true, global_y_pred)
    global_span = float(np.ptp(global_y_true))
    global_nrmse_pct = np.nan if global_span <= 0 else float(100.0 * global_rmse / global_span)
    global_r2 = _r_squared(global_y_true, global_y_pred)
    print(
        "Axis 4 global fit quality:",
        f"RMSE={global_rmse:.6f}",
        f"NRMSE={global_nrmse_pct:.2f}%",
        f"R2={global_r2:.5f}",
        f"n_points={global_y_true.size}",
    )
else:
    print("Axis 4 global fit quality: no fitted points available.")

selected_curve_results = [
    deg_curve_map[diode_idx]
    for diode_idx in diodes_to_plot
    if diode_idx in deg_curve_map
]

y_offset = 0
for curve_result in selected_curve_results[::-1]:
    curve_color = ddc(td[curve_result['diode_idx']])
    ax.plot(curve_result['x_fit'], curve_result['y_fit'], marker='.', color=curve_color, ls='', alpha=0.5, zorder=3)
    (fit_line,) = ax.plot(
        curve_result['x_full'],
        curve_result['y_model_full'],
        ls=':',
        lw=1.8,
        c=fit_curve_color,
        zorder=4,
    )
    fit_line.set_path_effects(fit_curve_pe)

    annotation_x = 0.8 * curve_result['x_full'][-1]
    if annotation_x > ax.get_xlim()[1]*0.6:
        annotation_x = ax.get_xlim()[1]*0.6
    annotation_y = stretched_damage_model(
        annotation_x,
        global_damage_fit['shared_A'],
        curve_result['k'],
        global_damage_fit['beta'],
    )
    annotation_y = float(np.clip(annotation_y + y_offset, 0.12, 0.96))
    ax.text(
        annotation_x,
        annotation_y+0.05,
        f"$k={1/curve_result['k']:.2f}\\,$MGy",
        color=curve_color,
        fontsize=SMALL_SIZE - 1,
        ha='left',
        va='bottom',
        bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='none', alpha=0.65),
        zorder=5,
    )

fit_legend_handle = matplotlib.lines.Line2D(
    [0], [0],
    color=fit_curve_color,
    ls=':',
    lw=1.8,
    label='Kohlrausch fit of degradation',
)
ax.legend(handles=[fit_legend_handle], loc='lower left', frameon=True)
ax4_symbol_ax, ax4_scale_ax = add_wedge_signal_symbol(
    ax4,
    0.25,
    0.65,
    0.35,
    0.3,
    show_dose_scale=True,
    scale_gap_frac=0.02,
    scale_width_frac=0.03,
)

ax.set_xlabel("Total dose (MGy)")
ax.set_ylabel(f'Normalized pixel response')

ax.set_xlim(ax.get_xlim()), ax.set_ylim(0, ax.get_ylim()[1])

# ------------------------------------------------------------------------------------------------------------------
# Ax3: Degradation at same dose vs proton energy | Linearity Before after for highest dose?
# ------------------------------------------------------------------------------------------------------------------
ax = ax3

comp_dose = 1  # MGy
k_energy = []
k_values = []

for diode_idx in range(len(mean_energy_per_diode)):
    curve_result = deg_curve_map.get(diode_idx)
    if curve_result is None or np.max(curve_result['x_full']) < comp_dose:
        continue
    k_value = 1 / curve_result['k']
    k_energy.append(curve_result['mean_energy_mev'])
    k_values.append(k_value)
    ax.plot(
        curve_result['mean_energy_mev'],
        k_value,
        marker='.',
        c=ddc(td[diode_idx]),
        alpha=0.8,
    )

ax.set_xlabel("Mean Proton Energy behind wedge (MeV)")
ax.xaxis.label.set_color('red')
ax.tick_params(axis='x', colors='red')
ax.spines['bottom'].set_color('red')

# ax.set_ylabel(f"Residual signal after {comp_dose}$\\,$MGy ($\\%$)")
ax.set_ylabel(f"Damage scaling $k$ from fit (MGy)")
# Keep secondary axis reserved but empty for now.


x_vals = np.array([0.25 * i + x for i in range(len(line_cache))])
x_data_min = x_vals[0]
x_data_max = x_vals[-1]
frac_left = transform_data_to_axis_coordinates(ax1, [x_data_min, 0])[0]
frac_right = transform_data_to_axis_coordinates(ax1, [x_data_max, 0])[0]
e_min = np.min(mean_energy_per_diode)
e_max = np.max(mean_energy_per_diode)
energy_range = e_max - e_min
total_energy_range = energy_range / (frac_right - frac_left)
ax3_xlim_left = e_min - frac_left * total_energy_range
ax3_xlim_right = ax3_xlim_left + total_energy_range
ax3.set_xlim(ax3_xlim_left, ax3_xlim_right)
ax.invert_xaxis()

ax.set_xlim(ax.get_xlim()), ax.set_ylim(ax.get_ylim())

mean_signal_before = np.mean(linearity_signal_before, axis=1)
mean_signal_middle = np.mean(linearity_signal_middle, axis=1)
mean_signal_after = np.mean(linearity_signal_after, axis=1)
mean_fit_before = np.mean(fit, axis=0)
mean_fit_middle = np.mean(fit3, axis=0)
mean_fit_after = np.mean(fit2, axis=0)
mean_fit_r2_before = float(np.mean(fit_r2))
mean_fit_r2_middle = float(np.mean(fit_r23))
mean_fit_r2_after = float(np.mean(fit_r22))


def compute_linearity_nrmse_per_channel(currents_data, signal_data, fit_currents_data, fit_data):
    currents_data = np.asarray(currents_data, dtype=float)
    fit_currents_data = np.asarray(fit_currents_data, dtype=float)
    signal_data = np.asarray(signal_data, dtype=float)
    fit_data = np.asarray(fit_data, dtype=float)

    if signal_data.ndim != 2 or fit_data.ndim != 2:
        raise ValueError("Signal and fit data must be 2D arrays.")

    # Accept both layouts: (n_points, n_channels) and (n_channels, n_points)
    if signal_data.shape[0] == currents_data.size:
        signal_matrix = signal_data
    elif signal_data.shape[1] == currents_data.size:
        signal_matrix = signal_data.T
    else:
        raise ValueError(
            f"Signal data shape {signal_data.shape} does not match currents length {currents_data.size}."
        )

    if fit_data.shape[0] == fit_currents_data.size:
        fit_matrix = fit_data
    elif fit_data.shape[1] == fit_currents_data.size:
        fit_matrix = fit_data.T
    else:
        raise ValueError(
            f"Fit data shape {fit_data.shape} does not match fit-currents length {fit_currents_data.size}."
        )

    if fit_matrix.shape[1] != signal_matrix.shape[1]:
        raise ValueError(
            f"Channel mismatch between signal ({signal_matrix.shape[1]}) and fit ({fit_matrix.shape[1]})."
        )

    n_channels = signal_matrix.shape[1]
    nrmse = np.full(n_channels, np.nan, dtype=float)
    for diode_idx in range(n_channels):
        y_true = signal_matrix[:, diode_idx]
        y_fit_raw = fit_matrix[:, diode_idx]
        valid_fit = np.isfinite(fit_currents_data) & np.isfinite(y_fit_raw)
        if np.count_nonzero(valid_fit) < 2:
            continue

        x_fit = fit_currents_data[valid_fit]
        y_fit = y_fit_raw[valid_fit]
        order = np.argsort(x_fit)
        x_fit = x_fit[order]
        y_fit = y_fit[order]
        unique_x, unique_idx = np.unique(x_fit, return_index=True)
        if unique_x.size < 2:
            continue
        y_fit = y_fit[unique_idx]

        valid_true = np.isfinite(currents_data) & np.isfinite(y_true)
        if np.count_nonzero(valid_true) < 2:
            continue
        x_true = currents_data[valid_true]
        y_true = y_true[valid_true]

        y_pred = np.interp(x_true, unique_x, y_fit)
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        y_span = np.ptp(y_true)
        if y_span > 0:
            nrmse[diode_idx] = 100.0 * rmse / y_span

    return nrmse


def orient_linearity_matrix(x_data, y_data):
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)
    if y_data.ndim != 2:
        raise ValueError("Linearity matrix must be 2D.")
    if y_data.shape[0] == x_data.size:
        return y_data
    if y_data.shape[1] == x_data.size:
        return y_data.T
    raise ValueError(f"Linearity matrix shape {y_data.shape} does not match x-data length {x_data.size}.")


def compute_linearity_slope_per_channel(fit_currents_data, fit_data):
    fit_currents_data = np.asarray(fit_currents_data, dtype=float)
    fit_matrix = orient_linearity_matrix(fit_currents_data, fit_data)
    n_channels = fit_matrix.shape[1]
    slopes = np.full(n_channels, np.nan, dtype=float)
    for diode_idx in range(n_channels):
        y_fit = fit_matrix[:, diode_idx]
        valid = np.isfinite(fit_currents_data) & np.isfinite(y_fit)
        if np.count_nonzero(valid) < 2:
            continue
        x_valid = fit_currents_data[valid]
        y_valid = y_fit[valid]
        # Linear slope from fitted curve values (works for both a*x and a*x+b models)
        slopes[diode_idx] = np.polyfit(x_valid, y_valid, 1)[0]
    return slopes


nrmse_before = compute_linearity_nrmse_per_channel(currents, linearity_signal_before, fit_currents, fit)
nrmse_middle = compute_linearity_nrmse_per_channel(currents3, linearity_signal_middle, fit_currents3, fit3)
nrmse_after = compute_linearity_nrmse_per_channel(currents2, linearity_signal_after, fit_currents2, fit2)
slope_before = compute_linearity_slope_per_channel(fit_currents, fit)
slope_middle = compute_linearity_slope_per_channel(fit_currents3, fit3)
slope_after = compute_linearity_slope_per_channel(fit_currents2, fit2)
print("After linearity NRMSE vs diode channel (%):")
for diode_channel, nrmse_value in enumerate(nrmse_after, start=1):
    if np.isfinite(nrmse_value):
        print(f"  channel {diode_channel:3d}: {nrmse_value:.4f}")
    else:
        print(f"  channel {diode_channel:3d}: nan")

r2_before = np.asarray(fit_r2, dtype=float)
r2_middle = np.asarray(fit_r23, dtype=float)
r2_after = np.asarray(fit_r22, dtype=float)
n_channels = len(nrmse_after)
k_ax3 = np.full(n_channels, np.nan, dtype=float)
for diode_idx in range(n_channels):
    curve_result = deg_curve_map.get(diode_idx)
    if curve_result is None or np.max(curve_result['x_full']) < comp_dose:
        continue
    k_ax3[diode_idx] = 1.0 / curve_result['k']

if not (len(slope_before) == len(slope_middle) == len(slope_after) == len(r2_before) == len(r2_middle) == len(r2_after) == n_channels):
    raise RuntimeError(
        "Linearity export dimension mismatch: "
        f"slopes=({len(slope_before)}, {len(slope_middle)}, {len(slope_after)}), "
        f"r2=({len(r2_before)}, {len(r2_middle)}, {len(r2_after)}), nrmse={n_channels}."
    )

linearity_export_path = results_path / "Graph2X_LongTermWedge_LinearityChannelStats.csv"
with open(linearity_export_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "diode_channel",
        "slope_before",
        "slope_middle",
        "slope_after",
        "r2_before",
        "r2_middle",
        "r2_after",
        "nrmse_before_pct",
        "nrmse_middle_pct",
        "nrmse_after_pct",
        "k_ax3_mgy",
    ])
    for diode_channel in range(1, n_channels + 1):
        idx = diode_channel - 1
        writer.writerow([
            diode_channel,
            slope_before[idx],
            slope_middle[idx],
            slope_after[idx],
            r2_before[idx],
            r2_middle[idx],
            r2_after[idx],
            nrmse_before[idx],
            nrmse_middle[idx],
            nrmse_after[idx],
            k_ax3[idx],
        ])
print(f"Saved linearity channel comparison CSV to: {linearity_export_path}")

# ------------------------------------------------------------------------------------------------------------------
# Ax5: NRMSE of linearity fits vs proton energy
# ------------------------------------------------------------------------------------------------------------------
ax = ax5
energy_axis = np.asarray(mean_energy_per_diode, dtype=float)
valid_before = np.isfinite(energy_axis) & np.isfinite(nrmse_before)
valid_middle = np.isfinite(energy_axis) & np.isfinite(nrmse_middle)
valid_after = np.isfinite(energy_axis) & np.isfinite(nrmse_after)

ax.plot(energy_axis[valid_before][0:-5], nrmse_before[valid_before][0:-5], color='k', marker='.', ls='', alpha=0.8, label='Before')
# ax.plot(energy_axis[valid_middle], nrmse_middle[valid_middle], color='tab:orange', marker='.', ls='', alpha=0.8, label='Middle')
ax.plot(energy_axis[valid_after][1:-5], nrmse_after[valid_after][1:-5], color='b', marker='.', ls='', alpha=0.8, label='After')

ax.set_xlabel("Mean Proton Energy behind wedge (MeV)")
ax.xaxis.label.set_color('red')
ax.tick_params(axis='x', colors='red')
ax.spines['bottom'].set_color('red')
ax.set_ylabel("Linearity fit NRMSE ($\\%$)")
ax.set_xlim(ax3.get_xlim()), ax.set_ylim(0, 5.5)
ax.tick_params(axis='both', labelsize=SMALL_SIZE - 1)
# ax.legend(loc='lower right', frameon=True)

ax5_inset = ax.inset_axes([0.2, 0.43, 0.52, 0.5])
ax5_inset.plot(currents, mean_signal_before, color='k', marker='.', ls='', alpha=0.75)
ax5_inset.plot(fit_currents, mean_fit_before, color='k', ls='--', lw=1.0, alpha=0.85)
# ax5_inset.plot(currents3, mean_signal_middle, color='tab:orange', marker='.', ls='', alpha=0.75)
# ax5_inset.plot(fit_currents3, mean_fit_middle, color='tab:orange', ls='--', lw=1.0, alpha=0.85)
ax5_inset.plot(currents2, mean_signal_after, color='b', marker='.', ls='', alpha=0.75)
ax5_inset.plot(fit_currents2, mean_fit_after, color='b', ls='--', lw=1.0, alpha=0.85)
ax5_inset.set_xlabel(r'Beam current (pA$\,\mathrm{cm}^{-2}$)', fontsize=9, labelpad=0)
ax5_inset.set_ylabel(f'Signal current ({scale_dict[A.scale][1]}A)', fontsize=9, labelpad=0.8)
ax5_inset.set_xscale('log')
ax5_inset.set_yscale('log')
ax5_inset.set_xlim([1e+1, 1e+4])
ax5_inset.set_ylim(2e+1, 2e+4)
ax5_inset.xaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=4))
ax5_inset.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
ax5_inset.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=40))
ax5_inset.xaxis.set_minor_formatter(mticker.NullFormatter())
ax5_inset.tick_params(axis='x', which='major', length=3)
ax5_inset.tick_params(axis='x', which='minor', length=2)
ax5_inset.tick_params(axis='both', labelsize=9)

ax5_inset.text(
    *transform_axis_to_data_coordinates(ax5_inset, [0.04, 0.95]),
    rf'$R^2={mean_fit_r2_before:.4f}$' + '\n' +  rf'Before',
    color='k',
    fontsize=10,
    ha='left',
    va='top',
    # bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=0.75),
)
"""
ax5_inset.text(
    *transform_axis_to_data_coordinates(ax5_inset, [0.52, 0.95]),
    rf'Middle' + '\n' + rf'$R^2={mean_fit_r2_middle:.4f}$',
    color='tab:orange',
    fontsize=SMALL_SIZE - 3,
    ha='left',
    va='top',
    bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=0.75),
)
"""
ax5_inset.text(
    *transform_axis_to_data_coordinates(ax5_inset, [0.96, 0.04]),
    rf'After' + '\n' + rf'$R^2={mean_fit_r2_after:.4f}$',
    color='b',
    fontsize=10,
    ha='right',
    va='bottom',
    # bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=0.75),
)

# ------------------------------------------------------------------------------------------------------------------
# Ax6: Degraded vs compensated wedge map (half/half)
# ------------------------------------------------------------------------------------------------------------------
ax = ax6
map_folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_171225/')
map_results_path = Path('/Users/nico_brosda/Cyrce_Messungen/ResultsPEEK_161225/')
map_measurements = ['exp9_', 'exp10_']

image_path = Path('/Users/nico_brosda/Cyrce_Messungen/Info_Files/Wedge5deg.jpeg')

map_readout, map_position_parser, map_voltage_parser, map_current_parser = (
    lambda x, y: ams_2line_fast_avg(x, y, channel_assignment=channel_assignment),
    standard_position,
    standard_voltage,
    current5
)
map_A = Analyzer(
    (2, 64), (0.4, 0.4), (0.1, 0.1),
    readout=map_readout,
    diode_offset=[[0, -0.25], np.zeros(64)],
    position_parser=map_position_parser,
    voltage_parser=map_voltage_parser,
    current_parser=map_current_parser,
)

map_files = []
for map_crit in map_measurements:
    map_A.set_measurement(map_folder_path, map_crit)
    map_files += map_A.measurement_files
map_A.measurement_files = map_files
map_A.set_dark_measurement(map_folder_path, ['dark_current'])
map_A.normalization(
    Path('/Users/nico_brosda/Cyrce_Messungen/matrix_260325/'),
    ['exp7_norm1,9V_'],
    normalization_module=lambda list_of_files, instance, method='least_squares': normalization_from_translated_array_v3(
        list_of_files, instance, method, align_lines=True
    ),
)

map_A.name = 'ax6_degraded'
map_A.load_measurement()
map_A.create_map(inverse=[True, False])
raw_map = overlap_treatment(deepcopy(map_A.maps[0]), map_A, True)

def deinterleave_to_2x64(a):
    a = np.asarray(a)
    if a.shape[-1] != 128:
        raise ValueError(f"Expected last axis = 128, got {a.shape}")
    even = a[..., 0::2]
    odd = a[..., 1::2]
    return np.stack((odd, even), axis=-2)

correction_factor = np.load(map_results_path / 'correction_factor.npy')
correction_factor = deinterleave_to_2x64(correction_factor)
map_norm_factor = deepcopy(map_A.norm_factor)
map_A.norm_factor = map_norm_factor / correction_factor
map_A.name = 'ax6_compensated'
map_A.load_measurement()
map_A.create_map(inverse=[True, False])
comp_map = overlap_treatment(deepcopy(map_A.maps[0]), map_A, True)
map_A.norm_factor = map_norm_factor

raw_z = np.asarray(raw_map['z'], dtype=float)
comp_z = np.asarray(comp_map['z'], dtype=float)
rows = min(raw_z.shape[0], comp_z.shape[0])
cols = min(raw_z.shape[1], comp_z.shape[1])
raw_z = raw_z[:rows, :cols]
comp_z = comp_z[:rows, :cols]
mix_z = np.array(raw_z, copy=True)
split_col = cols // 2
mix_z[:, split_col:] = comp_z[:, split_col:]

mix_map = {
    'x': np.asarray(raw_map['x'])[:cols]-np.min(np.asarray(raw_map['x'])[:cols]),
    'y': np.asarray(raw_map['y'])[:rows][::-1],
    'z': mix_z,
    'position': '',
}
map_A.maps = [mix_map]
shared_vmax = float(np.nanmax([np.nanmax(raw_z), np.nanmax(comp_z)]))
map_A.plot_map(
    save_path=None,
    pixel='fill',
    intensity_limits=[0, shared_vmax],
    ax_in=ax,
    fig_in=fig,
    colorbar=True,
)

ax.set_xlabel("Transversal position (mm)")
ax.set_ylabel("Position along wedge (mm)")
ax.set_xlim(ax.get_xlim()), ax.set_ylim(ax.get_ylim())
ax.invert_yaxis()
ax.set_ylim(ax.get_ylim()[0]*0.95, ax.get_ylim()[1]*1.05)

'''
# Add setup image via OffsetImage/AnnotationBbox with ~0.3x0.3 axis-fraction footprint.
ax_bbox = ax.get_window_extent()
with Image.open(image_path) as setup_img:
    img_w_px, img_h_px = setup_img.size
target_w_frac = 0.40
target_h_frac = 0.40
zoom_w = target_w_frac * ax_bbox.width / img_w_px
zoom_h = target_h_frac * ax_bbox.height / img_h_px
# OffsetImage applies DPI correction (dpi_cor=True), so compensate for figure DPI.
setup_zoom = min(zoom_w, zoom_h) / (dpi / 72.0)
add_image(ax, image_path, location=(0.65, 0.98), align_corner=(0.5, 1), zoom=setup_zoom, background=True)
'''

ax.text(
    *transform_axis_to_data_coordinates(ax, [0.25, 0.02]),
    "Degraded",
    ha='center',
    va='bottom',
    fontsize=10,
    color='k',
)
ax.text(
    *transform_axis_to_data_coordinates(ax, [0.75, 0.02]),
    "Compensated",
    ha='center',
    va='bottom',
    fontsize=10,
    color='k',
)

# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

add_png_icon(ax1, A, 'custom', translation=None, zoom=0.17, custom_in=((0.02, 0.6), (0, 0.5)))
add_png_icon(ax6, A, 'top left', translation='x', zoom=0.17)

ax1.text(*transform_axis_to_data_coordinates(ax1, [0.97, 0.97]), r'\textbf{(a)}', fontsize=10, ha='right',
         va='top', color='k')
ax2.text(*transform_axis_to_data_coordinates(ax2, [0.97, 0.97]), r'\textbf{(b)}', fontsize=10, ha='right',
         va='top', color='k')
ax3.text(*transform_axis_to_data_coordinates(ax3, [0.97, 0.97]), r'\textbf{(c)}', fontsize=10, ha='right',
         va='top', color='k')
ax4.text(*transform_axis_to_data_coordinates(ax4, [0.97, 0.97]), r'\textbf{(d)}', fontsize=10, ha='right',
         va='top', color='k')
ax5.text(*transform_axis_to_data_coordinates(ax5, [0.97, 0.97]), r'\textbf{(e)}', fontsize=10, ha='right',
         va='top', color='k')
ax6.text(*transform_axis_to_data_coordinates(ax6, [0.97, 0.97]), r'\textbf{(f)}', fontsize=10, ha='right',
         va='top', color='k')

format_save(
    save_path=results_path,
    save_name=f"Graph2X_LongTermWedge",
    dpi=dpi,
    plot_size=plot_size,
    save_format=save_format,
    fig=fig,
    axes=[ax1, ax2, ax3, ax4, ax5, ax6, ax_r, ax2_symbol_ax, ax4_symbol_ax, ax5_inset],
    legend=False,
)
