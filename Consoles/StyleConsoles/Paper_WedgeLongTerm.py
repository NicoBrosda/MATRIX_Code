import matplotlib
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.collections import LineCollection

from Consoles.StyleConsoles.Utils_ImageLoad import *
from EvaluationSoftware.simulation_connectors import *
from Consoles.Consoles_13_202511.PasteIn_DosePerDiode import *

linearity_signal_before = np.array(signal, copy=True)
linearity_signal_after = np.array(signal2, copy=True)

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

plot_size = (18 * cm, 2 / 2 * 18 / 1.2419 * cm)
# Setup of the final figure
# Structure: Ax1 Logo Line Super Res - Ax2 Logo Gaf - Ax3 BeamShape 2-Line - Ax4 BeamShape Gaf -
# Ax5 Logo Overlay - Ax6 Beam Overlay
fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=plot_size)
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

ax.set_xlabel(r'Position (mm)')
ax.set_ylabel(f'Normed signal Current')

ax.set_xlim(ax.get_xlim())
ax.set_ylim(ax.get_ylim())

handles, labels = ax.get_legend_handles_labels()
sigma_item = SigmaLegendItem(color='r', alpha=0.2, lw=1.5)
handles.append(sigma_item)
labels.append('E$_\\mathrm{Proton}$')
ax.legend(handles, labels, handler_map={SigmaLegendItem: HandlerSigmaLegendItem()}, loc='lower center')

ax.set_xlabel(r'Position (mm)')
ax.set_ylabel(f'Relative response normed to 1')
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

ax.set_xlabel(r'Number of protons per diode (1e12)')
ax.set_ylabel(f'Relative response normed to 1')

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

irradiation_datasets = build_global_damage_datasets(exp_data)
global_damage_fit = run_staged_global_damage_fit(irradiation_datasets)
deg_curve_map = {curve_result['diode_idx']: curve_result for curve_result in global_damage_fit['curve_results']}

print(
    "Axis 4 global damage fit:",
    f"success={global_damage_fit['success']}",
    f"beta={global_damage_fit['beta']:.4f}",
    f"shared_A={global_damage_fit['shared_A']:.4f}",
    f"highest_dose_diode={global_damage_fit['highest_dose_diode_idx'] + 1}",
    f"strategy={global_damage_fit['fit_strategy']}",
)

selected_curve_results = [
    deg_curve_map[diode_idx]
    for diode_idx in diodes_to_plot
    if diode_idx in deg_curve_map
]
annotation_offsets = np.linspace(0.035, -0.035, len(selected_curve_results)) if selected_curve_results else []
for curve_result, y_offset in zip(selected_curve_results[::-1], annotation_offsets[::-1]):
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
        rf"$k={curve_result['k']:.2f}$",
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
ax.set_ylabel(f'Relative response normed to 1')

ax.set_xlim(ax.get_xlim()), ax.set_ylim(0, ax.get_ylim()[1])

# ------------------------------------------------------------------------------------------------------------------
# Ax3: Degradation at same dose vs proton energy | Linearity Before after for highest dose?
# ------------------------------------------------------------------------------------------------------------------
ax = ax3

comp_dose = 1  # MGy

for diode_idx in range(len(mean_energy_per_diode)):
    curve_result = deg_curve_map.get(diode_idx)
    if curve_result is None or np.max(curve_result['x_full']) < comp_dose:
        continue
    x_ind = np.argmin(np.abs(comp_dose - curve_result['x_full']))
    ax.plot(
        curve_result['mean_energy_mev'],
        curve_result['y_model_full'][x_ind] * 100,
        marker='.',
        c=ddc(td[diode_idx]),
        alpha=0.8,
    )

ax.set_xlabel("Mean Proton Energy behind wedge (MeV)")
ax.set_ylabel(f"Residual signal after {comp_dose}$\\,$MGy ($\\%$)")

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
mean_signal_after = np.mean(linearity_signal_after, axis=1)
mean_fit_before = np.mean(fit, axis=0)
mean_fit_after = np.mean(fit2, axis=0)
mean_fit_r2_before = float(np.mean(fit_r2))
mean_fit_r2_after = float(np.mean(fit_r22))

ax3_inset = ax.inset_axes([0.22, 0.21, 0.42, 0.42])
ax3_inset.plot(currents, mean_signal_before, color='k', marker='.', ls='', alpha=0.75)
ax3_inset.plot(fit_currents, mean_fit_before, color='k', ls='--', lw=1.0, alpha=0.85)
ax3_inset.plot(currents2, mean_signal_after, color='b', marker='.', ls='', alpha=0.75)
ax3_inset.plot(fit_currents2, mean_fit_after, color='b', ls='--', lw=1.0, alpha=0.85)
ax3_inset.set_xlabel(r'Beam current (pA$\,\mathrm{cm}^{-2}$)', fontsize=SMALL_SIZE - 1, labelpad=2)
ax3_inset.set_ylabel(f'Signal current ({scale_dict[A.scale][1]}A)', fontsize=SMALL_SIZE - 1)
ax3_inset.set_xscale('log')
ax3_inset.set_yscale('log')
ax3_inset.set_xlim([0.9e+1, 1.7e+3])
ax3_inset.set_ylim(0.5e-2, ax3_inset.get_ylim()[1] * 2)
ax3_inset.tick_params(axis='both', labelsize=SMALL_SIZE - 2)
ax3_inset.text(
    *transform_axis_to_data_coordinates(ax3_inset, [0.98, 0.14]),
    rf'Before $R^2={mean_fit_r2_before:.4f}$',
    color='k',
    fontsize=SMALL_SIZE - 2,
    ha='right',
    va='bottom',
    bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.75),
)
ax3_inset.text(
    *transform_axis_to_data_coordinates(ax3_inset, [0.98, 0.06]),
    rf'After $R^2={mean_fit_r2_after:.4f}$',
    color='b',
    fontsize=SMALL_SIZE - 2,
    ha='right',
    va='bottom',
    bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.75),
)

# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

# add_png_icon(ax1, A, 'top left', translation=None, zoom=0.2)

ax1.text(*transform_axis_to_data_coordinates(ax1, [0.97, 0.97]), r'\textbf{(a)}', fontsize=10, ha='right',
         va='top', color='k')
ax2.text(*transform_axis_to_data_coordinates(ax2, [0.97, 0.97]), r'\textbf{(b)}', fontsize=10, ha='right',
         va='top', color='k')
ax3.text(*transform_axis_to_data_coordinates(ax3, [0.97, 0.97]), r'\textbf{(c)}', fontsize=10, ha='right',
         va='top', color='k')
ax4.text(*transform_axis_to_data_coordinates(ax4, [0.97, 0.97]), r'\textbf{(d)}', fontsize=10, ha='right',
         va='top', color='k')

format_save(
    save_path=results_path,
    save_name=f"Graph2X_LongTermWedge",
    dpi=dpi,
    plot_size=plot_size,
    save_format=save_format,
    fig=fig,
    axes=[ax1, ax2, ax3, ax4, ax_r, ax3_inset, ax2_symbol_ax, ax4_symbol_ax],
    legend=False,
)
