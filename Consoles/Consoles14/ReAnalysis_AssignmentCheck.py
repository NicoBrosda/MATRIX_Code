"""
Verifies that measurements in Dummy_Test3.py are correctly paired with simulations.
Specific check: P0 (wheel_pos=0) must map to thickness=0 in both wheel tables and
simulation param lists.

Prints a full assignment table and plots signal/current vs. wheel position to confirm
monotonic Bragg-peak behaviour.
"""
from EvaluationSoftware.main import *
import h5py

save_format = '.png'

_HERE = Path(__file__).parent
_FILES = _HERE.parent.parent / 'Files'

results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Results_260325/WheelEnergies/')
results_path.mkdir(parents=True, exist_ok=True)

cache_400 = np.load('/Users/nico_brosda/Cyrce_Messungen/Results_260325/Fast_Mode/cache_400_roi.npy', allow_pickle=True)
cache_200 = np.load('/Users/nico_brosda/Cyrce_Messungen/Results_260325/Fast_Mode/cache_200_roi.npy', allow_pickle=True)

data_wheel_200 = pd.read_csv(_FILES / 'energies_after_wheel_diffusor200.txt', sep='\t', header=4,
                             names=['position', 'thickness', 'energies'])
data_wheel_400 = pd.read_csv(_FILES / 'energies_after_wheel_diffusor400.txt', sep='\t', header=4,
                             names=['position', 'thickness', 'energies'])

# Faraday currents as hardcoded in Dummy_Test3.py
raw_400 = np.array([887, 888, 885, 880, 876, 872, 884, 880, 876, 871,
                    888, 887, 884, 881, 881, 877, 882, 880, 879], dtype=float)
raw_200 = np.array([1.73, 1.72, 1.72, 1.70, 1.71, 1.70, 1.72, 1.71, 1.70,
                    1.72, 1.72, 1.71, 1.70, 1.72, 1.72, 1.71, 1.70, 1.69, 1.69, 1.76])

sim_path_200 = Path('/Users/nico_brosda/GateSimulation/GATE10/Simulation/output/1e+07ALDensityNormal50umwindow200diff')
sim_path_400 = Path('/Users/nico_brosda/GateSimulation/GATE10/Simulation/output/1e+07ALDensityNormal50umwindow400diff')


def get_phsp4_median(sim_dir, thickness):
    sim_name = sim_dir.name
    for t_str in [str(thickness), str(int(thickness))]:
        h5 = sim_dir / f'{sim_name}_param{t_str}.h5'
        if h5.exists():
            with h5py.File(h5, 'r') as f:
                run = list(f.keys())[0]
                return float(f[f'{run}/phsp4'].attrs['E_stat_median'])
    return float('nan')


# ── Build full assignment tables ───────────────────────────────────────────────
print('\n=== 400µm: cache → wheel_pos → thickness → energy → simulation ===')
print(f'  {"idx":>3} | {"wp":>3} | {"thickness":>10} | {"table_E":>9} | {"phsp4_E":>9} | {"faraday":>9} | {"signal":>8} | {"sig/I":>8} | note')
rows_400 = []
for i, row in enumerate(cache_400):
    wp = int(row[0])
    t  = data_wheel_400['thickness'].iloc[wp]
    e_tab = data_wheel_400['energies'].iloc[wp]
    e_sim = get_phsp4_median(sim_path_400, t)
    cur = raw_400[i]
    sig = row[1]
    ok = '✓' if wp == i else '✗ MISMATCH'
    rows_400.append((i, wp, t, e_tab, e_sim, cur, sig, sig/cur, ok))
    print(f'  {i:3d} | {wp:3d} | {t:10.3f} | {e_tab:9.4f} | {e_sim:9.4f} | {cur:9.0f} | {sig:8.3f} | {sig/cur:8.4f} | {ok}')

print('\n=== 200µm: cache → wheel_pos → thickness → energy → simulation ===')
print(f'  {"idx":>3} | {"wp":>3} | {"thickness":>10} | {"table_E":>9} | {"phsp4_E":>9} | {"faraday":>9} | {"signal":>8} | {"sig/I":>8} | note')
rows_200 = []
for i, row in enumerate(cache_200):
    wp = int(row[0])
    t  = data_wheel_200['thickness'].iloc[wp]
    e_tab = data_wheel_200['energies'].iloc[wp]
    e_sim = get_phsp4_median(sim_path_200, t)
    cur = raw_200[i]
    sig = row[1]
    ok = '✓' if wp == i else '✗ MISMATCH'
    note = ok + (' (PEEK)' if wp == 19 else '')
    rows_200.append((i, wp, t, e_tab, e_sim, cur, sig, sig/cur, note))
    print(f'  {i:3d} | {wp:3d} | {t:10.3f} | {e_tab:9.4f} | {e_sim:9.4f} | {cur:9.2f} | {sig:8.3f} | {sig/cur:8.4f} | {note}')

# Summary assertion
all_ok_400 = all(int(cache_400[i, 0]) == i for i in range(len(cache_400)))
all_ok_200 = all(int(cache_200[i, 0]) == i for i in range(len(cache_200)))
print(f'\n  400µm assignment: {"ALL CORRECT ✓" if all_ok_400 else "BUG DETECTED ✗"}')
print(f'  200µm assignment: {"ALL CORRECT ✓" if all_ok_200 else "BUG DETECTED ✗"}')
print(f'  P0 → thickness=0 (400µm): {data_wheel_400["thickness"].iloc[0]:.3f} mm → {data_wheel_400["energies"].iloc[0]:.4f} MeV  {"✓" if data_wheel_400["thickness"].iloc[0] == 0.0 else "✗"}')
print(f'  P0 → thickness=0 (200µm): {data_wheel_200["thickness"].iloc[0]:.3f} mm → {data_wheel_200["energies"].iloc[0]:.4f} MeV  {"✓" if data_wheel_200["thickness"].iloc[0] == 0.0 else "✗"}')

# ── Figure 1: assignment table heat map ────────────────────────────────────────
# Visual: idx (x) vs wp (y) — should be identity diagonal
fig1, axes1 = plt.subplots(1, 2, figsize=(10, 4))

idx_arr_400 = np.arange(len(cache_400))
wp_arr_400  = np.array([int(r[0]) for r in cache_400])
idx_arr_200 = np.arange(len(cache_200))
wp_arr_200  = np.array([int(r[0]) for r in cache_200])

for ax, idx_arr, wp_arr, label in [
    (axes1[0], idx_arr_400, wp_arr_400, '400 µm diffuser'),
    (axes1[1], idx_arr_200, wp_arr_200, '200 µm diffuser'),
]:
    ax.scatter(idx_arr, wp_arr, s=50, zorder=3)
    ax.plot([0, max(idx_arr)], [0, max(idx_arr)], 'r--', lw=1, label='Expected (identity)')
    ax.set_xlabel('Cache index')
    ax.set_ylabel('wheel_pos stored in cache')
    ax.set_title(label)
    ax.legend(fontsize=8)

format_save(save_path=results_path, save_name='assignment_fig1_index_vs_wheelpos',
            save_format=save_format, legend=False, fig=fig1)

# ── Figure 2: normalised signal vs table energy ────────────────────────────────
# Both series must be monotonically increasing from high energy (P0) to low energy (P18).
e_400 = np.array([data_wheel_400['energies'].iloc[int(r[0])] for r in cache_400])
e_200 = np.array([data_wheel_200['energies'].iloc[int(r[0])] for r in cache_200])

normed_400 = np.array([r[1] for r in cache_400]) / raw_400
normed_200 = np.array([r[1] for r in cache_200]) / raw_200

is_peek = np.array([int(r[0]) == 19 for r in cache_200])

fig2, ax2 = plt.subplots(figsize=(9, 6))
ax2.plot(e_400, normed_400, 'o-', color='C0', label='400 µm diffuser', zorder=3)
ax2.plot(e_200[~is_peek], normed_200[~is_peek], 's-', color='C1',
         label='200 µm diffuser', zorder=3)
ax2.plot(e_200[is_peek], normed_200[is_peek], 'D', color='C1', ms=8,
         markerfacecolor='none', markeredgewidth=2, zorder=4, label='PEEK (P19, 200 µm)')

# Annotate first and last point of each series
ax2.annotate('P0', (e_400[0], normed_400[0]), textcoords='offset points', xytext=(5, -12), fontsize=8)
ax2.annotate('P18', (e_400[-1], normed_400[-1]), textcoords='offset points', xytext=(5, 5), fontsize=8)
ax2.annotate('P0', (e_200[0], normed_200[0]), textcoords='offset points', xytext=(5, 5), fontsize=8)
ax2.annotate('P18', (e_200[18], normed_200[18]), textcoords='offset points', xytext=(-20, -14), fontsize=8)

ax2.set_xlabel('Table reference energy (MeV)')
ax2.set_ylabel('Signal / Faraday current (a.u.)')
ax2.set_title('Normalised signal vs. energy — monotonicity check')
ax2.legend(fontsize=9)

format_save(save_path=results_path, save_name='assignment_fig2_signal_vs_energy',
            save_format=save_format, legend=False, fig=fig2)

# ── Figure 3: phsp4 energy vs table energy ────────────────────────────────────
# Confirms that phsp4 energies (used as x-axis in corrected plot) match table energies.
phsp4_400 = np.array([get_phsp4_median(sim_path_400, data_wheel_400['thickness'].iloc[int(r[0])])
                      for r in cache_400])
phsp4_200 = np.array([get_phsp4_median(sim_path_200, data_wheel_200['thickness'].iloc[int(r[0])])
                      for r in cache_200])

fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))

ax3a.scatter(e_400, phsp4_400, color='C0', s=25, label='400 µm (P0–P18)', zorder=3)
ax3a.scatter(e_200[~is_peek], phsp4_200[~is_peek], color='C1', s=25,
             label='200 µm (P0–P18)', zorder=3)
ax3a.scatter(e_200[is_peek], phsp4_200[is_peek], color='C1', s=60, marker='D',
             facecolors='none', linewidths=2, label='PEEK (P19)', zorder=4)
all_e = np.concatenate([e_400, e_200])
valid = np.isfinite(all_e) & (all_e > 0)
e_lim = (0, np.nanmax(all_e) * 1.02)
ax3a.plot(e_lim, e_lim, 'k--', lw=0.8, label='1:1')
ax3a.set_xlim(e_lim)
ax3a.set_ylim(e_lim)
ax3a.set_xlabel('Table reference energy (MeV)')
ax3a.set_ylabel('phsp4 median energy (MeV)')
ax3a.set_title('phsp4 vs. table energy')
ax3a.legend(fontsize=8)

# Delta
delta_400 = phsp4_400 - e_400
delta_200_normeek = phsp4_200[~is_peek] - e_200[~is_peek]
ax3b.plot(e_400, delta_400, 'o-', color='C0', label='400 µm')
ax3b.plot(e_200[~is_peek], delta_200_normeek, 's-', color='C1', label='200 µm')
ax3b.axhline(0, color='k', lw=0.8, ls='--')
ax3b.set_xlabel('Table reference energy (MeV)')
ax3b.set_ylabel('phsp4 − table (MeV)')
ax3b.set_title('Energy difference (simulation vs. table)')
ax3b.legend(fontsize=8)

format_save(save_path=results_path, save_name='assignment_fig3_phsp4_vs_table',
            save_format=save_format, legend=False, fig=fig3)

print('\nAll figures saved to:', results_path)
