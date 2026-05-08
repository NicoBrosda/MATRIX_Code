"""
ReAnalysis_MetricComparison.py

Compare experiment signal-extraction methods against three simulation references,
across multiple simulation sets.

Experiment metrics (all ROI-restricted):
    max           — 99th-percentile peak
    top200        — mean of top-200 ROI signals
    gaussian      — full MLE Gaussian fit μ
    half_gaussian — adaptive: full Gaussian unless distribution is right-truncated
                    (symmetry ratio < 0.6); then mirror left bins around the mode
                    and fit a full Gaussian to the synthetic distribution
    roi_mean      — current default
    roi_median    — current default

Simulation references (one x-value per measurement):
    equivalent — same metric applied to sim ROI distribution (1:1 pairing)
    gaussian   — sim Gaussian centre (same x for all exp methods)
    old        — old sim_response2('EnergyVariation1e6_param') interpolated at table energies

For each (exp_method, sim_ref) pair: fit y = a·x and y = a·x + b, record R² + slope.

Outputs in Results_260325/ReAnalysisEnergyResponse/MetricComparison/<sim_set_tag>/.
"""
from EvaluationSoftware.main import *
from EvaluationSoftware.simulation_connectors import *
import h5py
import pickle
import matplotlib
matplotlib.rcParams['text.usetex'] = False
from scipy.stats import norm as scipy_norm
from scipy.optimize import curve_fit
from scipy.constants import e

save_format = '.png'

# ── paths ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
_FILES = _HERE.parent.parent / 'Files'

results_stem = Path('/Users/nico_brosda/Cyrce_Messungen/Results_260325/')
metric_root = Path('/Users/nico_brosda/Cyrce_Messungen/Results_260325/ReAnalysisEnergyResponse/MetricComparison/')
metric_root.mkdir(parents=True, exist_ok=True)

# ── simulation sets to process ────────────────────────────────────────────────
sim_root = Path('/Users/nico_brosda/GateSimulation/GATE10/Simulation/output')
SIM_CONFIGS = [
    {'tag': 'Normal50um',
     'sim_path_200': sim_root / '1e+07ALDensityNormal50umwindow200diff',
     'sim_path_400': sim_root / '1e+07ALDensityNormal50umwindow400diff'},
    {'tag': 'Var50um',
     'sim_path_200': sim_root / '1e+08ALDensityVar50umwindow200diff',
     'sim_path_400': sim_root / '1e+08ALDensityVar50umwindow400diff'},
    {'tag': 'Var200um',
     'sim_path_200': sim_root / '1e+08ALDensityVar200umwindow200diff',
     'sim_path_400': sim_root / '1e+08ALDensityVar200umwindow400diff'},
]

# ── analysis constants ────────────────────────────────────────────────────────
ROI_RADIUS_MM = 8.0
SIM_PIXEL_SIZE_MM = 50 / 200
P0_MANUAL_CENTER = (18.0, 67.5)
_PEEK_CX, _PEEK_CY = (11.5 + 24.0) / 2, 65.5
_PEEK_RX, _PEEK_RY = (24.0 - 11.5) / 2, 6.5

MAX_PERCENTILE = 99.0
HALF_SYM_THRESHOLD = 0.6
HIST_BINS = 40

aperture_radii = np.array([15, 16, 17, 18, 19, 20], dtype=float)
r_baseline = 15.0

flare_cmap = sns.color_palette("flare", as_cmap=True)

# ── wheel reference tables ────────────────────────────────────────────────────
data_wheel_200 = pd.read_csv(_FILES / 'energies_after_wheel_diffusor200.txt', sep='\t', header=4,
                             names=['position', 'thickness', 'energies'])
data_wheel_400 = pd.read_csv(_FILES / 'energies_after_wheel_diffusor400.txt', sep='\t', header=4,
                             names=['position', 'thickness', 'energies'])

# ── load shared cached measurement and EnergyVariation simulation data ────────
cache_meas_raw_path = results_stem / 'Fast_Mode/cache_meas_raw.pkl'
with open(cache_meas_raw_path, 'rb') as _f:
    all_meas_stored = pickle.load(_f)
print(f'Loaded {len(all_meas_stored)} cached measurements.')

_cache_simresp_path = results_stem / 'Fast_Mode/cache_simresp_EnergyVariation.pkl'
with open(_cache_simresp_path, 'rb') as _f:
    _sr = pickle.load(_f)
sim_energy_old = np.array(_sr['sim_energy'])
sim_res_old_raw = np.array(_sr['sim_res_raw'])

rescale_sim = 1e3
simn = 1e6 / rescale_sim / ((np.pi * (10e-3) ** 2) / (0.25e-3) ** 2)
sim_res_old = sim_res_old_raw / simn / 2  # keV/primary
print(f'Loaded EnergyVariation simulation, {len(sim_energy_old)} points.')

# ── Faraday currents (hardcoded as in Dummy_Test3.py) ─────────────────────────
raw_400 = np.array([887, 888, 885, 880, 876, 872, 884, 880, 876, 871,
                    888, 887, 884, 881, 881, 877, 882, 880, 879], dtype=float)
raw_200 = np.array([1.73, 1.72, 1.72, 1.70, 1.71, 1.70, 1.72, 1.71, 1.70,
                    1.72, 1.72, 1.71, 1.70, 1.72, 1.72, 1.71, 1.70, 1.69, 1.69, 1.76])

r_aperture = 15e-3
additional_scale = (1 / e) * 1e-18 * (np.pi * r_aperture ** 2) / (0.47e-3) ** 2
rescale_current = 1e6 * additional_scale
currents_400 = raw_400 * 1e-12 * 0.568 / e / rescale_current
currents_200 = raw_200 * 1e-9 * 0.568 / e / rescale_current

# ── WPE preliminary points ────────────────────────────────────────────────────
wpe_y = np.array([77, 87, 73, 88, 75, 75, 78, 61, 67, 63, 51, 54, 46, 42], dtype=float)
wpe_x = np.interp([100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 226.7],
                  sim_energy_old, sim_res_old)

# ── ROI mask helpers ──────────────────────────────────────────────────────────
_ii, _jj = np.meshgrid(np.arange(2), np.arange(64), indexing='ij')
DX_CH = (_ii - 0.5) * 0.5
DY_CH = (_jj - 31.5) * 0.5 + np.array([0, -0.25])[_ii]


def make_peek_mask(pos1_y, scale=1.0):
    cy_phys = 2 * pos1_y - 0.25 - _PEEK_CY
    rx, ry = _PEEK_RX * scale, _PEEK_RY * scale
    def mask(x_ch, y_ch):
        return (((x_ch - _PEEK_CX) / rx) ** 2 +
                ((y_ch - cy_phys) / ry) ** 2 <= 1) & (y_ch >= cy_phys)
    return mask


def build_exp_dist(meas, cx_default, cy_default, peek_variant='standard'):
    is_peek = meas['is_peek']
    sig_list = []

    if is_peek and peek_variant == 'none':
        for _, sig, _ in meas['entries']:
            sig_list.extend(sig.flatten().tolist())
        return np.array(sig_list)

    if is_peek:
        pos1_y = meas['entries'][0][0][1]
        scale = 1.5 if peek_variant == 'larger' else 1.0
        roi_mask = make_peek_mask(pos1_y, scale=scale)
    else:
        cx, cy = cx_default, cy_default
        def roi_mask(x_ch, y_ch):
            return np.sqrt((x_ch - cx) ** 2 + (y_ch - cy) ** 2) <= ROI_RADIUS_MM

    for pos, sig, _ in meas['entries']:
        x_ch = pos[0] + DX_CH
        y_ch = pos[1] + DY_CH
        in_roi = roi_mask(x_ch, y_ch)
        sig_list.extend(sig[in_roi].tolist())

    return np.array(sig_list)


def build_sim_dist(sim_map, roi_radius_mm=ROI_RADIUS_MM):
    yy, xx = np.indices(sim_map.shape)
    cx_px = (sim_map.shape[1] - 1) / 2.0
    cy_px = (sim_map.shape[0] - 1) / 2.0
    radius_px = roi_radius_mm / SIM_PIXEL_SIZE_MM
    mask = np.sqrt((xx - cx_px) ** 2 + (yy - cy_px) ** 2) <= radius_px
    return sim_map[mask].astype(float)


# ── metric functions ──────────────────────────────────────────────────────────
def metric_max(values):
    if len(values) == 0:
        return np.nan
    return float(np.percentile(values, MAX_PERCENTILE))


def metric_top200(values, n=200):
    if len(values) == 0:
        return np.nan
    n = min(n, len(values))
    return float(np.mean(np.sort(values)[-n:]))


def metric_gaussian(values):
    if len(values) < 5:
        return np.nan
    mu, _ = scipy_norm.fit(values)
    return float(mu)


def _gauss(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _half_gauss_mirrored_bins(centers, counts, mode_idx):
    left_centers = centers[:mode_idx + 1]
    left_counts = counts[:mode_idx + 1].astype(float)
    mode_center = centers[mode_idx]
    synth_right_centers = 2 * mode_center - left_centers[:-1][::-1]
    synth_right_counts = left_counts[:-1][::-1]
    return (np.concatenate([left_centers, synth_right_centers]),
            np.concatenate([left_counts, synth_right_counts]))


def metric_half_gaussian(values, n_bins=HIST_BINS, sym_threshold=HALF_SYM_THRESHOLD):
    if len(values) < 10:
        return np.nan, False
    counts, edges = np.histogram(values, bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if counts.sum() == 0:
        return np.nan, False

    mode_idx = int(np.argmax(counts))
    n_left = counts[:mode_idx + 1].sum()
    n_right = counts[mode_idx + 1:].sum()
    sym_ratio = n_right / max(n_left, 1.0)
    truncated = sym_ratio < sym_threshold

    if truncated and mode_idx >= 2:
        all_centers, all_counts = _half_gauss_mirrored_bins(centers, counts, mode_idx)
        if all_counts.sum() == 0:
            return float(centers[mode_idx]), True
        try:
            p0 = [all_counts.max(), centers[mode_idx],
                  max((centers[mode_idx] - centers[0]) * 0.5, abs(centers[1] - centers[0]))]
            popt, _ = curve_fit(_gauss, all_centers, all_counts, p0=p0, maxfev=5000)
            return float(popt[1]), True
        except Exception:
            return float(centers[mode_idx]), True

    mu, _ = scipy_norm.fit(values)
    return float(mu), False


def metric_roi_mean(values):
    if len(values) == 0:
        return np.nan
    return float(np.mean(values))


def metric_roi_median(values):
    if len(values) == 0:
        return np.nan
    return float(np.median(values))


METRICS = {
    'max':           metric_max,
    'top200':        metric_top200,
    'gaussian':      metric_gaussian,
    'half_gaussian': metric_half_gaussian,
    'roi_mean':      metric_roi_mean,
    'roi_median':    metric_roi_median,
}
METRIC_NAMES = list(METRICS.keys())

method_colors = {
    'max':           'red',
    'top200':        'orange',
    'gaussian':      'green',
    'half_gaussian': 'magenta',
    'roi_mean':      'blue',
    'roi_median':    'purple',
}

# ── consensus centers ─────────────────────────────────────────────────────────
c400 = [(m['x_c'], m['y_c']) for m in all_meas_stored if not m['is_peek'] and m['diffuser'] == 400]
c200 = [(m['x_c'], m['y_c']) for m in all_meas_stored if not m['is_peek'] and m['diffuser'] == 200]
cx_400 = float(np.median([c[0] for c in c400]))
cy_400 = float(np.median([c[1] for c in c400]))
cx_200 = float(np.median([c[0] for c in c200]))
cy_200 = float(np.median([c[1] for c in c200]))
print(f'Centers: 400µm=({cx_400:.2f}, {cy_400:.2f})  200µm=({cx_200:.2f}, {cy_200:.2f})')


def get_center(diffuser, wheel_pos):
    if wheel_pos == 0:
        return P0_MANUAL_CENTER
    return (cx_400, cy_400) if diffuser == 400 else (cx_200, cy_200)


def lookup_sim_map(sim_maps, thickness, atol=1e-6):
    if thickness in sim_maps:
        return sim_maps[thickness]
    for k, v in sim_maps.items():
        if np.isclose(float(k), thickness, atol=atol):
            return v
    return None


# ── fit + plot helpers ────────────────────────────────────────────────────────
def proportional(x, a):
    return a * x


def linear(x, a, b):
    return a * x + b


def df_to_grid(df, value_col):
    rows = METRIC_NAMES
    cols = ['equivalent', 'gaussian', 'old']
    grid = np.full((len(rows), len(cols)), np.nan)
    for i, m in enumerate(rows):
        for j, ref in enumerate(cols):
            sel = df[(df['exp_method'] == m) & (df['sim_ref'] == ref)]
            if len(sel) > 0:
                grid[i, j] = sel[value_col].values[0]
    return grid, rows, cols


def _adaptive_text_color(v, vmin, vmax):
    if not np.isfinite(v) or vmax == vmin:
        return 'black'
    norm = (v - vmin) / (vmax - vmin)
    return 'black' if norm < 0.5 else 'white'


def draw_heatmap(ax, grid, rows, cols, title, fmt='{:.4f}'):
    im = ax.imshow(grid, aspect='auto', cmap=flare_cmap)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_title(title)
    vmin = np.nanmin(grid)
    vmax = np.nanmax(grid)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = grid[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha='center', va='center',
                        color=_adaptive_text_color(v, vmin, vmax),
                        fontsize=9, fontweight='bold')
    return im


def load_sim_maps(sim_path_200, sim_path_400):
    sim_tag = f'{sim_path_200.name}__{sim_path_400.name}'
    cache_sim_path = results_stem / f'Fast_Mode/cache_sim_{sim_tag}.pkl'
    if cache_sim_path.exists():
        with open(cache_sim_path, 'rb') as f:
            sc = pickle.load(f)
        return sc['sim_maps_200'], sc['sim_maps_400']
    print(f'  No sim cache for {sim_tag}; loading from MHD/H5...')
    sim_maps_200 = load_normalized_edep_maps(
        sim_path_200.parent, sim_path_200.name, f'{sim_path_200.name}_param',
        data_wheel_200['thickness'].tolist())
    sim_maps_400 = load_normalized_edep_maps(
        sim_path_400.parent, sim_path_400.name, f'{sim_path_400.name}_param',
        data_wheel_400['thickness'].tolist())
    return sim_maps_200, sim_maps_400


# ══════════════════════════════════════════════════════════════════════════════
# Per-sim-set analysis
# ══════════════════════════════════════════════════════════════════════════════
def run_for_sim_set(sim_path_200, sim_path_400, save_folder):
    save_folder.mkdir(parents=True, exist_ok=True)
    (save_folder / 'process_plots').mkdir(exist_ok=True)
    (save_folder / 'half_gaussian_fits').mkdir(exist_ok=True)
    (save_folder / 'peek_roi_sensitivity').mkdir(exist_ok=True)

    print('\n' + '=' * 70)
    print(f'  SIM SET: {sim_path_200.name} / {sim_path_400.name}')
    print(f'  OUTPUT:  {save_folder}')
    print('=' * 70)

    sim_maps_200, sim_maps_400 = load_sim_maps(sim_path_200, sim_path_400)
    print(f'  Sim maps loaded: 200µm={len(sim_maps_200)}, 400µm={len(sim_maps_400)}')

    # ── compute per-measurement distributions and metrics (cached per sim set) ─
    metric_cache_path = save_folder / 'cache_metric_comparison.pkl'
    try:
        with open(metric_cache_path, 'rb') as _f:
            records = pickle.load(_f)
        print(f'  Loaded metric cache ({len(records)} records).')
    except FileNotFoundError:
        print('  Building distributions and computing metrics...')
        records = []
        for meas in all_meas_stored:
            diffuser = meas['diffuser']
            wp = meas['wheel_pos']
            is_peek = meas['is_peek']

            if diffuser == 400:
                thickness    = float(data_wheel_400['thickness'].iloc[wp])
                table_energy = float(data_wheel_400['energies'].iloc[wp])
                current      = float(currents_400[wp])
                sim_maps     = sim_maps_400
            else:
                thickness    = float(data_wheel_200['thickness'].iloc[wp])
                table_energy = float(data_wheel_200['energies'].iloc[wp])
                current      = float(currents_200[wp])
                sim_maps     = sim_maps_200

            cx, cy = get_center(diffuser, wp)
            exp_dist = build_exp_dist(meas, cx, cy, peek_variant='standard')

            sim_map = lookup_sim_map(sim_maps, thickness)
            if sim_map is None:
                print(f'  WARN: no sim map for ({diffuser}µm, P{wp}, t={thickness}); skipped')
                continue
            sim_dist = build_sim_dist(sim_map)

            exp_metrics, sim_metrics = {}, {}
            truncation = {}
            for name, fn in METRICS.items():
                if name == 'half_gaussian':
                    exp_v, exp_trunc = fn(exp_dist)
                    sim_v, sim_trunc = fn(sim_dist)
                    truncation['exp'] = exp_trunc
                    truncation['sim'] = sim_trunc
                else:
                    exp_v = fn(exp_dist)
                    sim_v = fn(sim_dist)
                exp_metrics[name] = exp_v
                sim_metrics[name] = sim_v

            sim_old_value = float(np.interp(table_energy, sim_energy_old, sim_res_old))

            records.append({
                'diffuser':     diffuser,
                'wheel_pos':    wp,
                'is_peek':      is_peek,
                'thickness':    thickness,
                'table_energy': table_energy,
                'current':      current,
                'exp_dist':     exp_dist,
                'sim_dist':     sim_dist,
                'exp_metrics':  exp_metrics,
                'sim_metrics':  sim_metrics,
                'sim_old':      sim_old_value,
                'truncation':   truncation,
                'crit':         meas['crit'],
            })
        with open(metric_cache_path, 'wb') as _f:
            pickle.dump(records, _f)
        print(f'  Cached {len(records)} records.')

    records_sorted = sorted(records, key=lambda r: r['table_energy'])

    energies = np.array([r['table_energy'] for r in records_sorted])
    peek_mask = np.array([r['is_peek'] for r in records_sorted])

    y_arrays = {m: np.array([r['exp_metrics'][m] / r['current'] for r in records_sorted])
                for m in METRIC_NAMES}
    x_arrays_equivalent = {m: np.array([r['sim_metrics'][m] * 1e3 for r in records_sorted])
                           for m in METRIC_NAMES}
    x_array_gaussian = np.array([r['sim_metrics']['gaussian'] * 1e3 for r in records_sorted])
    x_array_old = np.array([r['sim_old'] for r in records_sorted])

    # ── fit all 18 combinations ───────────────────────────────────────────────
    fit_results = []
    for m in METRIC_NAMES:
        for ref in ['equivalent', 'gaussian', 'old']:
            if ref == 'equivalent':
                x = x_arrays_equivalent[m]
            elif ref == 'gaussian':
                x = x_array_gaussian
            else:
                x = x_array_old
            y = y_arrays[m]

            valid = np.isfinite(x) & np.isfinite(y)
            x_f, y_f = x[valid], y[valid]
            n = int(valid.sum())

            record = {'exp_method': m, 'sim_ref': ref, 'n': n,
                      'r2_prop': np.nan, 'slope_prop': np.nan,
                      'r2_lin':  np.nan, 'slope_lin':  np.nan, 'intercept_lin': np.nan}

            if n >= 3:
                try:
                    popt_p, _ = curve_fit(proportional, x_f, y_f)
                    ss_res = np.sum((y_f - proportional(x_f, *popt_p)) ** 2)
                    ss_tot = np.sum((y_f - np.mean(y_f)) ** 2)
                    record['r2_prop'] = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
                    record['slope_prop'] = float(popt_p[0])
                except Exception:
                    pass
                try:
                    popt_l, _ = curve_fit(linear, x_f, y_f)
                    ss_res = np.sum((y_f - linear(x_f, *popt_l)) ** 2)
                    ss_tot = np.sum((y_f - np.mean(y_f)) ** 2)
                    record['r2_lin'] = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
                    record['slope_lin'] = float(popt_l[0])
                    record['intercept_lin'] = float(popt_l[1])
                except Exception:
                    pass

            fit_results.append(record)

    df = pd.DataFrame(fit_results)
    df.to_csv(save_folder / 'summary_table.csv', index=False)
    print(f'  Saved {save_folder / "summary_table.csv"}')

    # ── summary heatmap (2×2) ─────────────────────────────────────────────────
    fig_hm, axes_hm = plt.subplots(2, 2, figsize=(14, 12))
    panels_hm = [
        (axes_hm[0, 0], 'r2_prop',    'R² (proportional fit)',    '{:.4f}'),
        (axes_hm[0, 1], 'slope_prop', 'Slope (proportional fit)', '{:.2f}'),
        (axes_hm[1, 0], 'r2_lin',     'R² (linear fit)',          '{:.4f}'),
        (axes_hm[1, 1], 'slope_lin',  'Slope (linear fit)',       '{:.2f}'),
    ]
    for ax, value_col, title, fmt in panels_hm:
        grid, rows, cols = df_to_grid(df, value_col)
        im = draw_heatmap(ax, grid, rows, cols, title, fmt=fmt)
        fig_hm.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig_hm.savefig(save_folder / f'summary_heatmap{save_format}', dpi=150, bbox_inches='tight')
    plt.close(fig_hm)

    # ── WPE agreement analysis ────────────────────────────────────────────────
    wpe_records = []
    for fr in fit_results:
        rec = {'exp_method': fr['exp_method'], 'sim_ref': fr['sim_ref']}
        if np.isfinite(fr['slope_prop']):
            pred_p = fr['slope_prop'] * wpe_x
            rec['wpe_rmse_prop'] = float(np.sqrt(np.mean((wpe_y - pred_p) ** 2)))
            ss_tot = float(np.sum((wpe_y - np.mean(wpe_y)) ** 2))
            rec['wpe_r2_prop'] = float(1 - np.sum((wpe_y - pred_p) ** 2) / ss_tot) if ss_tot > 0 else np.nan
        else:
            rec['wpe_rmse_prop'] = np.nan
            rec['wpe_r2_prop'] = np.nan
        if np.isfinite(fr['slope_lin']):
            pred_l = fr['slope_lin'] * wpe_x + fr['intercept_lin']
            rec['wpe_rmse_lin'] = float(np.sqrt(np.mean((wpe_y - pred_l) ** 2)))
            ss_tot = float(np.sum((wpe_y - np.mean(wpe_y)) ** 2))
            rec['wpe_r2_lin'] = float(1 - np.sum((wpe_y - pred_l) ** 2) / ss_tot) if ss_tot > 0 else np.nan
        else:
            rec['wpe_rmse_lin'] = np.nan
            rec['wpe_r2_lin'] = np.nan
        wpe_records.append(rec)

    wpe_df = pd.DataFrame(wpe_records)
    wpe_df.to_csv(save_folder / 'wpe_agreement.csv', index=False)

    fig_wpe, axes_wpe = plt.subplots(2, 2, figsize=(14, 12))
    panels_wpe = [
        (axes_wpe[0, 0], 'wpe_rmse_prop', 'WPE RMSE (proportional fit)', '{:.2f}'),
        (axes_wpe[0, 1], 'wpe_r2_prop',   'WPE R² (proportional fit)',    '{:.3f}'),
        (axes_wpe[1, 0], 'wpe_rmse_lin',  'WPE RMSE (linear fit)',        '{:.2f}'),
        (axes_wpe[1, 1], 'wpe_r2_lin',    'WPE R² (linear fit)',          '{:.3f}'),
    ]
    for ax, value_col, title, fmt in panels_wpe:
        grid, rows, cols = df_to_grid(wpe_df, value_col)
        im = draw_heatmap(ax, grid, rows, cols, title, fmt=fmt)
        fig_wpe.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig_wpe.savefig(save_folder / f'wpe_agreement_heatmap{save_format}', dpi=150, bbox_inches='tight')
    plt.close(fig_wpe)

    # ── aperture sweep (analytic) ─────────────────────────────────────────────
    aperture_records = []
    for r in aperture_radii:
        k = (r / r_baseline) ** 2
        for fr in fit_results:
            slope_p = fr['slope_prop'] * k if np.isfinite(fr['slope_prop']) else np.nan
            slope_l = fr['slope_lin']  * k if np.isfinite(fr['slope_lin'])  else np.nan
            intercept_l = fr['intercept_lin'] * k if np.isfinite(fr['intercept_lin']) else np.nan

            rec = {
                'exp_method':    fr['exp_method'],
                'sim_ref':       fr['sim_ref'],
                'r_mm':          float(r),
                'r2_prop':       fr['r2_prop'],
                'slope_prop':    float(slope_p) if np.isfinite(slope_p) else np.nan,
                'r2_lin':        fr['r2_lin'],
                'slope_lin':     float(slope_l) if np.isfinite(slope_l) else np.nan,
                'intercept_lin': float(intercept_l) if np.isfinite(intercept_l) else np.nan,
                'n':             fr['n'],
            }

            if np.isfinite(slope_p):
                pred_p = slope_p * wpe_x
                rec['wpe_rmse_prop'] = float(np.sqrt(np.mean((wpe_y - pred_p) ** 2)))
                ss_tot = float(np.sum((wpe_y - np.mean(wpe_y)) ** 2))
                rec['wpe_r2_prop'] = float(1 - np.sum((wpe_y - pred_p) ** 2) / ss_tot) if ss_tot > 0 else np.nan
            else:
                rec['wpe_rmse_prop'] = np.nan
                rec['wpe_r2_prop'] = np.nan
            if np.isfinite(slope_l):
                pred_l = slope_l * wpe_x + intercept_l
                rec['wpe_rmse_lin'] = float(np.sqrt(np.mean((wpe_y - pred_l) ** 2)))
                ss_tot = float(np.sum((wpe_y - np.mean(wpe_y)) ** 2))
                rec['wpe_r2_lin'] = float(1 - np.sum((wpe_y - pred_l) ** 2) / ss_tot) if ss_tot > 0 else np.nan
            else:
                rec['wpe_rmse_lin'] = np.nan
                rec['wpe_r2_lin'] = np.nan

            aperture_records.append(rec)

    ap_df = pd.DataFrame(aperture_records)
    ap_df.to_csv(save_folder / 'aperture_sweep_table.csv', index=False)
    print(f'  Saved {save_folder / "aperture_sweep_table.csv"} ({len(ap_df)} rows)')

    # Best (exp_method, sim_ref, r) for WPE_RMSE_prop
    best_idx = ap_df['wpe_rmse_prop'].idxmin()
    print('  Best aperture/method/sim_ref by wpe_rmse_prop:')
    print('  ' + str(ap_df.loc[best_idx]).replace('\n', '\n  '))

    # WPE_RMSE vs r line plot
    fig_ap, axes_ap = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    exp_palette = sns.color_palette("tab10", n_colors=len(METRIC_NAMES))
    for ax, ref in zip(axes_ap, ['equivalent', 'gaussian', 'old']):
        sub = ap_df[ap_df['sim_ref'] == ref]
        for col, m in zip(exp_palette, METRIC_NAMES):
            sm = sub[sub['exp_method'] == m].sort_values('r_mm')
            ax.plot(sm['r_mm'], sm['wpe_rmse_prop'], marker='o', color=col, label=m)
        ax.set_xlabel('Aperture radius (mm)')
        ax.set_ylabel('WPE RMSE (proportional fit)')
        ax.set_title(f'sim_ref = {ref}')
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig_ap.savefig(save_folder / f'aperture_sweep_wpe_vs_r{save_format}',
                   dpi=150, bbox_inches='tight')
    plt.close(fig_ap)

    # WPE_RMSE heatmaps faceted by r (shared color scale)
    fig_aph, axes_aph = plt.subplots(2, 3, figsize=(18, 12))
    all_grids = []
    for r in aperture_radii:
        sub = ap_df[ap_df['r_mm'] == r]
        grid, _, _ = df_to_grid(sub, 'wpe_rmse_prop')
        all_grids.append(grid)
    all_grids_arr = np.array(all_grids)
    vmin_global = float(np.nanmin(all_grids_arr))
    vmax_global = float(np.nanmax(all_grids_arr))

    for ax, r, grid in zip(axes_aph.flat, aperture_radii, all_grids):
        im = ax.imshow(grid, aspect='auto', cmap=flare_cmap, vmin=vmin_global, vmax=vmax_global)
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(['equivalent', 'gaussian', 'old'])
        ax.set_yticks(np.arange(len(METRIC_NAMES)))
        ax.set_yticklabels(METRIC_NAMES)
        ax.set_title(f'r = {r:.0f} mm — WPE RMSE (prop fit)')
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                            color=_adaptive_text_color(v, vmin_global, vmax_global),
                            fontsize=8, fontweight='bold')

    plt.tight_layout()
    fig_aph.savefig(save_folder / f'aperture_sweep_wpe_heatmaps{save_format}',
                    dpi=150, bbox_inches='tight')
    plt.close(fig_aph)

    # Aperture sweep — proportional fit (no WPE)
    fig_apr, axes_apr = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    for col_idx, ref in enumerate(['equivalent', 'gaussian', 'old']):
        sub = ap_df[ap_df['sim_ref'] == ref]
        ax_r2 = axes_apr[0, col_idx]
        ax_sl = axes_apr[1, col_idx]
        for col, m in zip(exp_palette, METRIC_NAMES):
            sm = sub[sub['exp_method'] == m].sort_values('r_mm')
            ax_r2.plot(sm['r_mm'], sm['r2_prop'],    marker='o', color=col, label=m)
            ax_sl.plot(sm['r_mm'], sm['slope_prop'], marker='o', color=col, label=m)
        ax_r2.set_ylabel('R² (proportional fit)')
        ax_r2.set_title(f'sim_ref = {ref}')
        ax_r2.grid(alpha=0.2)
        ax_sl.set_xlabel('Aperture radius (mm)')
        ax_sl.set_ylabel('Slope (proportional fit)')
        ax_sl.grid(alpha=0.2)
    axes_apr[0, 0].legend(fontsize=8, loc='lower right')
    plt.tight_layout()
    fig_apr.savefig(save_folder / f'aperture_sweep_prop_fit{save_format}',
                    dpi=150, bbox_inches='tight')
    plt.close(fig_apr)

    # Aperture sweep — linear fit (no WPE)
    fig_apl, axes_apl = plt.subplots(3, 3, figsize=(18, 14), sharex=True)
    for col_idx, ref in enumerate(['equivalent', 'gaussian', 'old']):
        sub = ap_df[ap_df['sim_ref'] == ref]
        ax_r2  = axes_apl[0, col_idx]
        ax_sl  = axes_apl[1, col_idx]
        ax_int = axes_apl[2, col_idx]
        for col, m in zip(exp_palette, METRIC_NAMES):
            sm = sub[sub['exp_method'] == m].sort_values('r_mm')
            ax_r2.plot( sm['r_mm'], sm['r2_lin'],        marker='o', color=col, label=m)
            ax_sl.plot( sm['r_mm'], sm['slope_lin'],     marker='o', color=col, label=m)
            ax_int.plot(sm['r_mm'], sm['intercept_lin'], marker='o', color=col, label=m)
        ax_r2.set_ylabel('R² (linear fit)')
        ax_r2.set_title(f'sim_ref = {ref}')
        ax_r2.grid(alpha=0.2)
        ax_sl.set_ylabel('Slope (linear fit)')
        ax_sl.grid(alpha=0.2)
        ax_int.set_xlabel('Aperture radius (mm)')
        ax_int.set_ylabel('Intercept (linear fit)')
        ax_int.grid(alpha=0.2)
    axes_apl[0, 0].legend(fontsize=8, loc='lower right')
    plt.tight_layout()
    fig_apl.savefig(save_folder / f'aperture_sweep_lin_fit{save_format}',
                    dpi=150, bbox_inches='tight')
    plt.close(fig_apl)

    # ── signal-vs-edep plots (one folder per aperture, with WPE overlaid) ─────
    print('  Generating signal-vs-edep plots (108 = 18 × 6 apertures)...')
    energy_cmap = sns.color_palette("crest_r", as_cmap=True)
    e_min, e_max = float(np.min(energies)), float(np.max(energies))
    def energy_color(en):
        return energy_cmap((en - e_min) / max(e_max - e_min, 1e-9))

    for r in aperture_radii:
        k = (r / r_baseline) ** 2
        folder_r = save_folder / f'signal_vs_edep_r{int(r)}'
        folder_r.mkdir(exist_ok=True)

        for fr in fit_results:
            m = fr['exp_method']
            ref = fr['sim_ref']

            if ref == 'equivalent':
                x = x_arrays_equivalent[m]
            elif ref == 'gaussian':
                x = x_array_gaussian
            else:
                x = x_array_old
            y = y_arrays[m] * k

            slope_p_r = fr['slope_prop']    * k if np.isfinite(fr['slope_prop'])    else np.nan
            slope_l_r = fr['slope_lin']     * k if np.isfinite(fr['slope_lin'])     else np.nan
            intercept_l_r = fr['intercept_lin'] * k if np.isfinite(fr['intercept_lin']) else np.nan

            valid = np.isfinite(x) & np.isfinite(y)
            sub = ap_df[(ap_df['exp_method'] == m) & (ap_df['sim_ref'] == ref) &
                        (ap_df['r_mm'] == float(r))].iloc[0]

            fig, ax = plt.subplots(figsize=(8.5, 6))
            for i in range(len(x)):
                if not valid[i]:
                    continue
                col = energy_color(energies[i])
                if peek_mask[i]:
                    ax.plot(x[i], y[i], marker='D', ms=11, mfc='none', mec=col, mew=2,
                            ls='', zorder=4)
                else:
                    ax.plot(x[i], y[i], marker='o', ms=7, mfc=col, mec=col, ls='', zorder=3)

            ax.plot(wpe_x, wpe_y, marker='+', ms=12, color='orange', mew=2, ls='',
                    zorder=5, label='WPE preliminary')

            x_max = max(float(np.max(x[valid])), float(np.max(wpe_x))) * 1.05
            x_fit = np.linspace(0, x_max, 100)
            if np.isfinite(slope_p_r):
                ax.plot(x_fit, slope_p_r * x_fit, 'r--', lw=1.5,
                        label=f'Prop:  a={slope_p_r:.2f}, R²={fr["r2_prop"]:.4f}\n'
                              f'        WPE_RMSE={sub["wpe_rmse_prop"]:.2f}, WPE_R²={sub["wpe_r2_prop"]:.3f}')
            if np.isfinite(slope_l_r):
                ax.plot(x_fit, slope_l_r * x_fit + intercept_l_r, 'b--', lw=1.5,
                        label=f'Lin:   a={slope_l_r:.2f}, b={intercept_l_r:.2f}, R²={fr["r2_lin"]:.4f}\n'
                              f'        WPE_RMSE={sub["wpe_rmse_lin"]:.2f}, WPE_R²={sub["wpe_r2_lin"]:.3f}')

            ax.set_xlabel('Simulated Edep (keV/primary)')
            ax.set_ylabel(f'Signal / current (a.u.)   [r = {r:.0f} mm]')
            ax.set_title(f'exp: {m}  ×  sim_ref: {ref}   (r = {r:.0f} mm, n={fr["n"]})')
            ax.legend(fontsize=8, loc='lower right')
            ax.grid(alpha=0.2)
            ax.set_xlim(0, ax.get_xlim()[1])
            ax.set_ylim(0, ax.get_ylim()[1])

            fname = f'{m}_vs_{ref}'
            fig.savefig(folder_r / f'{fname}{save_format}', dpi=120, bbox_inches='tight')
            plt.close(fig)

    # ── process plots ──────────────────────────────────────────────────────────
    print('  Generating process plots per measurement...')
    for r in records_sorted:
        diff_label = f'P{r["wheel_pos"]:02d}_{r["diffuser"]}um'
        if r['is_peek']:
            diff_label += '_PEEK'

        fig, (ax_e, ax_s) = plt.subplots(2, 1, figsize=(10, 8))

        if len(r['exp_dist']) > 0:
            ax_e.hist(r['exp_dist'], bins=HIST_BINS, color='steelblue', alpha=0.7,
                      label=f'ROI signal (N={len(r["exp_dist"])})')
            for name in METRIC_NAMES:
                v = r['exp_metrics'][name]
                if np.isfinite(v):
                    ls = '-' if name in ('roi_mean', 'roi_median') else '--'
                    ax_e.axvline(v, color=method_colors[name], lw=1.5, ls=ls,
                                 label=f'{name}={v:.3g}')
        title_e = f'Exp ROI - {diff_label}, E_table={r["table_energy"]:.2f} MeV'
        if r['truncation'].get('exp', False):
            title_e += '  [TRUNCATED -> half-Gaussian]'
        ax_e.set_xlabel('Signal (a.u.)')
        ax_e.set_ylabel('Counts')
        ax_e.set_title(title_e, fontsize=10)
        ax_e.legend(fontsize=7, loc='upper right')

        if len(r['sim_dist']) > 0:
            sim_kev = r['sim_dist'] * 1e3
            ax_s.hist(sim_kev, bins=HIST_BINS, color='firebrick', alpha=0.7,
                      label=f'Sim ROI Edep (N={len(sim_kev)})')
            for name in METRIC_NAMES:
                v = r['sim_metrics'][name]
                if np.isfinite(v):
                    ls = '-' if name in ('roi_mean', 'roi_median') else '--'
                    ax_s.axvline(v * 1e3, color=method_colors[name], lw=1.5, ls=ls,
                                 label=f'{name}={v*1e3:.3g}')
        title_s = 'Sim ROI Edep - same wheel position'
        if r['truncation'].get('sim', False):
            title_s += '  [TRUNCATED -> half-Gaussian]'
        ax_s.set_xlabel('Edep (keV/primary)')
        ax_s.set_ylabel('Counts')
        ax_s.set_title(title_s, fontsize=10)
        ax_s.legend(fontsize=7, loc='upper right')

        plt.tight_layout()
        fig.savefig(save_folder / 'process_plots' / f'{diff_label}{save_format}',
                    dpi=120, bbox_inches='tight')
        plt.close(fig)

    # ── half-Gaussian fit diagnostics ─────────────────────────────────────────
    print('  Generating half-Gaussian fit diagnostics...')
    for r in records_sorted:
        if len(r['exp_dist']) < 10:
            continue

        diff_label = f'P{r["wheel_pos"]:02d}_{r["diffuser"]}um'
        if r['is_peek']:
            diff_label += '_PEEK'

        counts, edges = np.histogram(r['exp_dist'], bins=HIST_BINS)
        centers = 0.5 * (edges[:-1] + edges[1:])
        width = edges[1] - edges[0]
        mode_idx = int(np.argmax(counts))

        truncated = r['truncation'].get('exp', False)
        half_mu = r['exp_metrics']['half_gaussian']
        full_mu = r['exp_metrics']['gaussian']

        n_left = counts[:mode_idx + 1].sum()
        n_right = counts[mode_idx + 1:].sum()
        sym = n_right / max(n_left, 1)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(centers, counts, width=width, color='steelblue', alpha=0.6,
               label='ROI signal histogram')

        if truncated and mode_idx >= 2:
            all_centers, all_counts = _half_gauss_mirrored_bins(centers, counts, mode_idx)
            synth_centers = all_centers[mode_idx + 1:]
            synth_counts = all_counts[mode_idx + 1:]
            ax.bar(synth_centers, synth_counts, width=width, color='orange', alpha=0.6,
                   edgecolor='red', linewidth=0.5, hatch='//',
                   label=f'Mirrored synthetic bins ({len(synth_centers)})')

        full_sigma = np.std(r['exp_dist'])
        if full_sigma > 0 and counts.max() > 0:
            x_smooth = np.linspace(centers[0], 2 * centers[mode_idx] - centers[0], 300) \
                if truncated else np.linspace(centers[0], centers[-1], 300)
            if np.isfinite(full_mu):
                A_full = counts.max()
                ax.plot(x_smooth, _gauss(x_smooth, A_full, full_mu, full_sigma), 'g--', lw=1.5,
                        label=f'Full Gaussian (raw values): mu={full_mu:.3g}')

            if truncated and mode_idx >= 2:
                try:
                    p0 = [all_counts.max(), centers[mode_idx],
                          max((centers[mode_idx] - centers[0]) * 0.5, abs(width))]
                    popt, _ = curve_fit(_gauss, all_centers, all_counts, p0=p0, maxfev=5000)
                    ax.plot(x_smooth, _gauss(x_smooth, *popt), 'r-', lw=2,
                            label=f'Mirrored half-Gaussian: mu={popt[1]:.3g}')
                except Exception:
                    pass

        ax.axvline(centers[mode_idx], color='k', ls=':', lw=1, label='Mode bin')
        flag = 'TRUNCATED -> mirrored half-fit' if truncated else 'untruncated -> full fit'
        ax.set_xlabel('Signal (a.u.)')
        ax.set_ylabel('Counts')
        ax.set_title(f'{diff_label} - symmetry={sym:.2f} ({flag}), '
                     f'E_table={r["table_energy"]:.2f} MeV', fontsize=10)
        ax.legend(fontsize=8, loc='upper right')

        fig.savefig(save_folder / 'half_gaussian_fits' / f'{diff_label}{save_format}',
                    dpi=120, bbox_inches='tight')
        plt.close(fig)

    # ── PEEK ROI sensitivity ──────────────────────────────────────────────────
    print('  PEEK ROI sensitivity test...')
    peek_records_l = [m for m in all_meas_stored if m['is_peek']]
    if peek_records_l:
        peek_meas = peek_records_l[0]
        peek_results = {}
        for variant in ['standard', 'larger', 'none']:
            dist = build_exp_dist(peek_meas, None, None, peek_variant=variant)
            ms = {}
            for name, fn in METRICS.items():
                if name == 'half_gaussian':
                    v, _ = fn(dist)
                else:
                    v = fn(dist)
                ms[name] = v
            peek_results[variant] = {'dist': dist, 'metrics': ms}

        peek_table = pd.DataFrame({v: peek_results[v]['metrics']
                                   for v in ['standard', 'larger', 'none']})
        peek_table.to_csv(save_folder / 'peek_roi_sensitivity' / 'peek_metric_table.csv')

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, variant in zip(axes, ['standard', 'larger', 'none']):
            dist = peek_results[variant]['dist']
            if len(dist) > 0:
                ax.hist(dist, bins=50, color='steelblue', alpha=0.7,
                        label=f'N={len(dist)}')
                for name in METRIC_NAMES:
                    v = peek_results[variant]['metrics'][name]
                    if np.isfinite(v):
                        ax.axvline(v, color=method_colors[name], lw=1.2, ls='--',
                                   label=f'{name}={v:.2g}')
            ax.set_xlabel('Signal (a.u.)')
            ax.set_ylabel('Counts')
            ax.set_title(f'PEEK - ROI variant: {variant}')
            ax.legend(fontsize=6, loc='upper right')

        plt.tight_layout()
        fig.savefig(save_folder / 'peek_roi_sensitivity' / f'peek_roi_overlays{save_format}',
                    dpi=120, bbox_inches='tight')
        plt.close(fig)

    print(f'  Done. Outputs in: {save_folder}\n')

    # Return key fit summary for cross-set comparison
    return {'tag': save_folder.name, 'df': df, 'wpe_df': wpe_df, 'ap_df': ap_df}


# ══════════════════════════════════════════════════════════════════════════════
# Main loop over simulation sets
# ══════════════════════════════════════════════════════════════════════════════
all_summaries = []
for cfg in SIM_CONFIGS:
    summary = run_for_sim_set(cfg['sim_path_200'], cfg['sim_path_400'],
                              metric_root / cfg['tag'])
    all_summaries.append({**summary, **cfg})

# ── cross-sim-set comparison ──────────────────────────────────────────────────
print('\n' + '=' * 70)
print('CROSS-SIM-SET COMPARISON')
print('=' * 70)

# Best WPE-RMSE per sim set
print('\nBest WPE_RMSE_prop per sim set (across all aperture × method × ref):')
for s in all_summaries:
    best_idx = s['ap_df']['wpe_rmse_prop'].idxmin()
    row = s['ap_df'].loc[best_idx]
    print(f"  [{s['tag']:>10}]  exp={row['exp_method']:<14} ref={row['sim_ref']:<10} "
          f"r={row['r_mm']:.0f}mm  WPE_RMSE={row['wpe_rmse_prop']:.2f}  "
          f"WPE_R²={row['wpe_r2_prop']:.3f}")

# Save a combined R²_prop summary across sim sets
rows = []
for s in all_summaries:
    for _, r in s['df'].iterrows():
        rows.append({**r, 'sim_set': s['tag']})
combined_df = pd.DataFrame(rows)
combined_df.to_csv(metric_root / 'combined_summary_table.csv', index=False)
print(f'\nSaved combined summary: {metric_root / "combined_summary_table.csv"}')
print(f'\nAll outputs in: {metric_root}')