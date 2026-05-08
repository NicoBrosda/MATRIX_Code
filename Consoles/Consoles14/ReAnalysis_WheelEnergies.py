from EvaluationSoftware.main import *
from EvaluationSoftware.simulation_connectors import *
import h5py

save_format = '.png'

# ── paths ──────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
_FILES = _HERE.parent.parent / 'Files'

sim_path_200 = Path('/Users/nico_brosda/GateSimulation/GATE10/Simulation/output/1e+07ALDensityNormal50umwindow200diff')
sim_path_400 = Path('/Users/nico_brosda/GateSimulation/GATE10/Simulation/output/1e+07ALDensityNormal50umwindow400diff')
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Results_260325/WheelEnergies/')
results_path.mkdir(parents=True, exist_ok=True)

# ── wheel energy reference tables ──────────────────────────────────────────────
data_wheel_200 = pd.read_csv(_FILES / 'energies_after_wheel_diffusor200.txt', sep='\t', header=4,
                             names=['position', 'thickness', 'energies'])
data_wheel_400 = pd.read_csv(_FILES / 'energies_after_wheel_diffusor400.txt', sep='\t', header=4,
                             names=['position', 'thickness', 'energies'])


def load_phsp_energy_stats(sim_dir, phsp_ids=('phsp1', 'phsp2', 'phsp3', 'phsp4')):
    """Read E_stat_* attrs for each phsp group across all param files in sim_dir."""
    sim_dir = Path(sim_dir)
    result = {}
    for h5 in sorted(sim_dir.glob('*.h5')):
        try:
            param = float(h5.stem.split('_param')[-1])
        except ValueError:
            continue
        with h5py.File(h5, 'r') as f:
            run = list(f.keys())[0]
            result[param] = {}
            for pid in phsp_ids:
                gp = f'{run}/{pid}'
                if gp not in f:
                    continue
                g = f[gp]
                result[param][pid] = {
                    'mean':   float(g.attrs.get('E_stat_mean',   np.nan)),
                    'median': float(g.attrs.get('E_stat_median', np.nan)),
                    'std':    float(g.attrs.get('E_stat_std',    np.nan)),
                    'count':  int(  g.attrs.get('E_stat_count',  0)),
                }
    return result


def build_arrays(stats, params, phsp_id, key):
    return np.array([stats[p][phsp_id][key] for p in params])


print('Loading simulation phsp stats...')
stats_200 = load_phsp_energy_stats(sim_path_200)
stats_400 = load_phsp_energy_stats(sim_path_400)

params_200 = np.array(sorted(stats_200.keys()))
params_400 = np.array(sorted(stats_400.keys()))

# ── build numpy arrays for all phsps ──────────────────────────────────────────
phsp1_mean_200 = build_arrays(stats_200, params_200, 'phsp1', 'mean')
phsp2_mean_200 = build_arrays(stats_200, params_200, 'phsp2', 'mean')
phsp3_mean_200 = build_arrays(stats_200, params_200, 'phsp3', 'mean')
phsp3_med_200  = build_arrays(stats_200, params_200, 'phsp3', 'median')
phsp3_std_200  = build_arrays(stats_200, params_200, 'phsp3', 'std')
phsp4_mean_200 = build_arrays(stats_200, params_200, 'phsp4', 'mean')
phsp4_med_200  = build_arrays(stats_200, params_200, 'phsp4', 'median')
phsp4_std_200  = build_arrays(stats_200, params_200, 'phsp4', 'std')

phsp1_mean_400 = build_arrays(stats_400, params_400, 'phsp1', 'mean')
phsp2_mean_400 = build_arrays(stats_400, params_400, 'phsp2', 'mean')
phsp3_mean_400 = build_arrays(stats_400, params_400, 'phsp3', 'mean')
phsp3_med_400  = build_arrays(stats_400, params_400, 'phsp3', 'median')
phsp3_std_400  = build_arrays(stats_400, params_400, 'phsp3', 'std')
phsp4_mean_400 = build_arrays(stats_400, params_400, 'phsp4', 'mean')
phsp4_med_400  = build_arrays(stats_400, params_400, 'phsp4', 'median')
phsp4_std_400  = build_arrays(stats_400, params_400, 'phsp4', 'std')

# ── table reference energies matched to sorted param order ────────────────────
ref_e_200 = dict(zip(data_wheel_200['thickness'], data_wheel_200['energies']))
ref_e_400 = dict(zip(data_wheel_400['thickness'], data_wheel_400['energies']))
table_200 = np.array([ref_e_200.get(p, np.nan) for p in params_200])
table_400 = np.array([ref_e_400.get(p, np.nan) for p in params_400])

# Absolute and relative differences
diff_mean_200  = phsp3_mean_200 - phsp4_mean_200
diff_mean_400  = phsp3_mean_400 - phsp4_mean_400
reldiff_200    = diff_mean_200 / phsp4_mean_200 * 100   # %
reldiff_400    = diff_mean_400 / phsp4_mean_400 * 100   # %

# ── Figure 1: all four phsp energy levels vs. wheel thickness ─────────────────
# Gives context: phsp1/2 are constant (source/before wheel), phsp3/4 vary.
fig1, axes1 = plt.subplots(1, 2, figsize=(14 * 1/2.54 * 2, 14 * 1/2.54 / 1.3))
phsp_styles = {
    'phsp1': dict(color='C0', ls='--', marker='o', ms=4, label='phsp1 (source)'),
    'phsp2': dict(color='C1', ls='--', marker='s', ms=4, label='phsp2'),
    'phsp3': dict(color='C2', ls='-',  marker='^', ms=5, label='phsp3 (before det.)'),
    'phsp4': dict(color='C3', ls='-',  marker='v', ms=5, label='phsp4 (at det.)'),
}

for ax, params, stats, table, label in [
    (axes1[0], params_200, stats_200, table_200, '200 µm diffuser'),
    (axes1[1], params_400, stats_400, table_400, '400 µm diffuser'),
]:
    for pid, kw in phsp_styles.items():
        means = build_arrays(stats, params, pid, 'mean')
        ax.plot(params, means, **kw)
    ax.plot(params, table, color='k', ls=':', marker='D', ms=4, lw=1.2, label='Table reference')
    ax.set_xlabel('Wheel thickness (mm)')
    ax.set_ylabel('Mean proton energy (MeV)')
    ax.set_title(label)
    ax.legend(fontsize=7)

format_save(save_path=results_path, save_name='fig1_all_phsp_energies',
            save_format=save_format, legend=False, fig=fig1)

# ── Figure 2: phsp3 vs. phsp4 — mean, median, table reference ─────────────────
# Shows whether the choice matters for the energy axis.
fig2, axes2 = plt.subplots(1, 2, figsize=(14 * 1/2.54 * 2, 14 * 1/2.54 / 1.3))

for ax, params, p3m, p3med, p4m, p4med, table, label in [
    (axes2[0], params_200, phsp3_mean_200, phsp3_med_200, phsp4_mean_200, phsp4_med_200,
     table_200, '200 µm diffuser'),
    (axes2[1], params_400, phsp3_mean_400, phsp3_med_400, phsp4_mean_400, phsp4_med_400,
     table_400, '400 µm diffuser'),
]:
    ax.plot(params, p3m,   color='C2', ls='-',  marker='^', ms=5, label='phsp3 mean')
    ax.plot(params, p3med, color='C2', ls='--', marker='^', ms=4, alpha=0.7, label='phsp3 median')
    ax.plot(params, p4m,   color='C3', ls='-',  marker='v', ms=5, label='phsp4 mean')
    ax.plot(params, p4med, color='C3', ls='--', marker='v', ms=4, alpha=0.7, label='phsp4 median')
    ax.plot(params, table, color='k',  ls=':',  marker='D', ms=4, lw=1.2, label='Table reference')
    ax.set_xlabel('Wheel thickness (mm)')
    ax.set_ylabel('Energy (MeV)')
    ax.set_title(label)
    ax.legend(fontsize=7)

format_save(save_path=results_path, save_name='fig2_phsp3_vs_phsp4_energies',
            save_format=save_format, legend=False, fig=fig2)

# ── Figure 3: absolute and relative difference (phsp3 − phsp4) vs. energy ─────
# Key diagnostic: does the difference grow at low energy? How does it compare to the spread?
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14 * 1/2.54 * 2, 14 * 1/2.54 / 1.3))

ax3a.plot(phsp4_mean_200, diff_mean_200, color='C2', ls='-', marker='^', ms=5,
          label='200 µm diffuser')
ax3a.plot(phsp4_mean_400, diff_mean_400, color='C3', ls='-', marker='v', ms=5,
          label='400 µm diffuser')
ax3a.axhline(0, color='k', lw=0.8, ls='--')
ax3a.set_xlabel('phsp4 mean energy (MeV)')
ax3a.set_ylabel('phsp3 − phsp4 mean energy (MeV)')
ax3a.set_title('Absolute difference')
ax3a.legend(fontsize=8)

ax3b.plot(phsp4_mean_200, reldiff_200, color='C2', ls='-', marker='^', ms=5,
          label='200 µm diffuser')
ax3b.plot(phsp4_mean_400, reldiff_400, color='C3', ls='-', marker='v', ms=5,
          label='400 µm diffuser')
ax3b.axhline(0, color='k', lw=0.8, ls='--')
ax3b.set_xlabel('phsp4 mean energy (MeV)')
ax3b.set_ylabel(r'(phsp3 $-$ phsp4) / phsp4 × 100 \%')
ax3b.set_title('Relative difference')
ax3b.legend(fontsize=8)

format_save(save_path=results_path, save_name='fig3_phsp3_minus_phsp4',
            save_format=save_format, legend=False, fig=fig3)

# ── Figure 4: energy spread (std) — phsp3 vs phsp4 ───────────────────────────
# Shows whether the peak width changes between the two scoring planes.
fig4, axes4 = plt.subplots(1, 2, figsize=(14 * 1/2.54 * 2, 14 * 1/2.54 / 1.3))

for ax, p4m, p3s, p4s, label in [
    (axes4[0], phsp4_mean_200, phsp3_std_200, phsp4_std_200, '200 µm diffuser'),
    (axes4[1], phsp4_mean_400, phsp3_std_400, phsp4_std_400, '400 µm diffuser'),
]:
    ax.plot(p4m, p3s, color='C2', ls='-', marker='^', ms=5, label='phsp3 std')
    ax.plot(p4m, p4s, color='C3', ls='-', marker='v', ms=5, label='phsp4 std')
    ax.set_xlabel('phsp4 mean energy (MeV)')
    ax.set_ylabel('Energy std (MeV)')
    ax.set_title(label)
    ax.legend(fontsize=8)

format_save(save_path=results_path, save_name='fig4_energy_spread_phsp3_vs_phsp4',
            save_format=save_format, legend=False, fig=fig4)

# ── Figure 5: simulation (phsp3 & phsp4) vs. table reference ──────────────────
# Shows how much the table-based x-axis differs from the simulation-based one.
fig5, axes5 = plt.subplots(1, 2, figsize=(14 * 1/2.54 * 2, 14 * 1/2.54 / 1.3))

for ax, p3m, p4m, table, label in [
    (axes5[0], phsp3_mean_200, phsp4_mean_200, table_200, '200 µm diffuser'),
    (axes5[1], phsp3_mean_400, phsp4_mean_400, table_400, '400 µm diffuser'),
]:
    valid = np.isfinite(table)
    ax.plot(table[valid], p3m[valid] - table[valid], color='C2', ls='-', marker='^', ms=5,
            label='phsp3 − table')
    ax.plot(table[valid], p4m[valid] - table[valid], color='C3', ls='-', marker='v', ms=5,
            label='phsp4 − table')
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xlabel('Table reference energy (MeV)')
    ax.set_ylabel('Simulation − table (MeV)')
    ax.set_title(label)
    ax.legend(fontsize=8)

format_save(save_path=results_path, save_name='fig5_sim_vs_table_energies',
            save_format=save_format, legend=False, fig=fig5)

# ── Figure 6: combined overview — difference vs. energy spread ─────────────────
# Shows the difference relative to the std, i.e. how "significant" it is.
fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(14 * 1/2.54 * 2, 14 * 1/2.54 / 1.3))

ax6a.plot(phsp4_mean_200, diff_mean_200 / phsp4_std_200, color='C2', ls='-', marker='^', ms=5,
          label='200 µm diffuser')
ax6a.plot(phsp4_mean_400, diff_mean_400 / phsp4_std_400, color='C3', ls='-', marker='v', ms=5,
          label='400 µm diffuser')
ax6a.axhline(0, color='k', lw=0.8, ls='--')
ax6a.set_xlabel('phsp4 mean energy (MeV)')
ax6a.set_ylabel(r'$\Delta E_\mathrm{mean}$ / $\sigma_\mathrm{phsp4}$')
ax6a.set_title('Difference / energy spread')
ax6a.legend(fontsize=8)

# Scatter: phsp3 mean vs phsp4 mean, both diffusers, with 1:1 line
ax6b.scatter(phsp4_mean_200, phsp3_mean_200, color='C2', marker='^', s=30, label='200 µm diffuser')
ax6b.scatter(phsp4_mean_400, phsp3_mean_400, color='C3', marker='v', s=30, label='400 µm diffuser')
all_e = np.concatenate([phsp4_mean_200, phsp4_mean_400])
e_lim = (np.min(all_e) * 0.97, np.max(all_e) * 1.01)
ax6b.plot(e_lim, e_lim, 'k--', lw=1.0, label='1:1')
ax6b.set_xlim(e_lim)
ax6b.set_ylim(e_lim)
ax6b.set_xlabel('phsp4 mean energy (MeV)')
ax6b.set_ylabel('phsp3 mean energy (MeV)')
ax6b.set_title('phsp3 vs. phsp4 (scatter)')
ax6b.legend(fontsize=8)

format_save(save_path=results_path, save_name='fig6_difference_vs_spread',
            save_format=save_format, legend=False, fig=fig6)

# ── Print summary table ────────────────────────────────────────────────────────
print('\n=== 200 µm diffuser ===')
print(f'{"param":>6}  {"phsp3_mean":>10}  {"phsp4_mean":>10}  {"delta":>8}  {"rel%":>7}  {"phsp4_std":>9}  {"delta/std":>9}  {"table":>9}')
for i, p in enumerate(params_200):
    print(f'{p:6.3f}  {phsp3_mean_200[i]:10.4f}  {phsp4_mean_200[i]:10.4f}  '
          f'{diff_mean_200[i]:8.4f}  {reldiff_200[i]:7.3f}  '
          f'{phsp4_std_200[i]:9.4f}  {diff_mean_200[i]/phsp4_std_200[i]:9.4f}  '
          f'{table_200[i]:9.4f}')

print('\n=== 400 µm diffuser ===')
print(f'{"param":>6}  {"phsp3_mean":>10}  {"phsp4_mean":>10}  {"delta":>8}  {"rel%":>7}  {"phsp4_std":>9}  {"delta/std":>9}  {"table":>9}')
for i, p in enumerate(params_400):
    print(f'{p:6.3f}  {phsp3_mean_400[i]:10.4f}  {phsp4_mean_400[i]:10.4f}  '
          f'{diff_mean_400[i]:8.4f}  {reldiff_400[i]:7.3f}  '
          f'{phsp4_std_400[i]:9.4f}  {diff_mean_400[i]/phsp4_std_400[i]:9.4f}  '
          f'{table_400[i]:9.4f}')