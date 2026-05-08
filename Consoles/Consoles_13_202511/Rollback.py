from Consoles.StyleConsoles.Utils_ImageLoad import *
from EvaluationSoftware.simulation_connectors import *

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

deg_curves = []
for i in range(128):
    fig, ax = plt.subplots()
    print("Diode: ", i)
    position = 25 - 16.25 + i * 0.25
    print(f"Position: {position} mm")
    th = shape.calculate_value(position)
    print(f"Wedge thickness: {th:.2f} mm")
    sim_pixel = int(position / 0.25)
    dose_per_diode = proton_count*np.mean(dose_norm[sim_pixel-2:sim_pixel+2, 98:102]) * np.mean(proton_density_distribution[sim_pixel, 98:102]) /1e+6
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
    ax.plot(dose_per_diode, y_fit4, ls='--', label=label)

    deg_curves.append([x_data, y_fit4])

    # --- plot original data ---
    ax.plot(dose_per_diode, signal, marker='x', color=diode_colour(i / 128), ls='', alpha=0.5, zorder=-1)

    ax.set_xlabel("Total dose (MGy)")
    ax.set_ylabel("Relative signal level of diode")
    ax.set_title(f'Diode {i+1} ({dose_per_diode[-1]:.2f} MGy), E_mean: {np.mean(median_energy[sim_pixel, 98:102]):.2f} MeV')
    ax.legend(fontsize=7)
    format_save(save_path/'DegCurvePerDiode/', f"DegradationDiode{i}", save_format=save_format, legend=False)

    if i == 107:
        dose_ext = np.linspace(0, 15, 10000)

        fig, ax = plt.subplots()
        y_fit1 = exp_decay(dose_ext, *popt1)
        label = f"Single exp: A={popt1[0]:.3f}, k={popt1[1]:.3f}, C={popt1[2]:.3f}, R$^2$={r_squared(y_data, exp_decay(x_data, *popt1)):.4f}"
        ax.plot(dose_ext, y_fit1, ls='--', label=label)

        y_fit2 = double_exp(dose_ext, *popt2)
        label = (
            f"Double exp: A1={popt2[0]:.3f}, k1={popt2[1]:.3f}, A2={popt2[2]:.3f}, k2={popt2[3]:.3f}, C={popt2[4]:.3f}, R$^2$={r_squared(y_data, double_exp(x_data, *popt2)):.4f}")
        ax.plot(dose_ext, y_fit2, ls='--', label=label)

        y_fit3 = stretched_exp(dose_ext, *popt3)
        label = (
            f"Stretched exp: A={popt3[0]:.3f}, k={popt3[1]:.3f}, beta={popt3[2]:.3f}, C={popt3[3]:.3f}, R$^2$={r_squared(y_data, stretched_exp(x_data, *popt3)):.4f}")
        ax.plot(dose_ext, y_fit3, ls='--', label=label)

        y_fit4 = explin_decay(dose_ext, *popt4)
        label = (
            f"Exp+Lin: A={popt4[0]:.3f}, k={popt4[1]:.3f}, beta={popt4[2]:.3f}, C={popt4[3]:.3f}, R$^2$={r_squared(y_data, explin_decay(x_data, *popt4)):.4f}")
        ax.plot(dose_ext, y_fit4, ls='--', label=label)

        ax.plot(dose_per_diode, signal, marker='x', color=diode_colour(i / 128), ls='', alpha=0.5, zorder=-1)
        ax.set_xlabel("Total dose (MGy)")
        ax.set_ylabel("Diode signal (nA)")
        ax.set_title(
            f'Diode {i + 1} ({dose_per_diode[-1]:.2f} MGy), E_mean: {np.mean(median_energy[sim_pixel, 98:102]):.2f} MeV')
        ax.legend(fontsize=7)
        ax.set_ylim(0, 1.05)
        format_save(save_path/'Extrapolation/', f"DegradationExtrapolation{i}", save_format=save_format, legend=False)

# '''
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

ax.set_xlabel("Median Proton Energy (MeV)")
ax.set_ylabel(f"Rest signal after {comp_dose}$\\,$MGy ($\\%$)")
format_save(save_path , f"DegradationvsEnergy", save_format=save_format, legend=False)

fig, ax = plt.subplots()

for i, curve in enumerate(deg_curves):
    if np.max(curve[0]) < comp_dose:
        continue
    x_ind = np.argmin(np.abs(comp_dose-curve[0]))
    ax.plot(mean_energy_per_diode[i], curve[1][x_ind]*100, marker='x', c='k', alpha=1)

ax.set_xlabel("Median Proton Energy (MeV)")
ax.set_ylabel(f"Rest signal after {comp_dose}$\\,$MGy ($\\%$)")
format_save(save_path , f"DegradationvsMeanEnergy", save_format=save_format, legend=False)

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



