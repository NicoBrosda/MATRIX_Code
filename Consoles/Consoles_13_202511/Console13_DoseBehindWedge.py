from Consoles.StyleConsoles.Utils_ImageLoad import *
from EvaluationSoftware.simulation_connectors import *

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black", "red", "yellow"])
# save_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/ResultsPEEK_161225/Simulation/')
# save_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/ResultsPEEK_161225/SimulationIdeal/')
# save_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/ResultsPEEK_161225/SimulationNoNuc2/')
save_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/ResultsPEEK_161225/SimulationActiveLayer/')


save_format = '.png'

output_path = Path("/Users/nico_brosda/GateSimulation/GATE10/Simulation/output/")
# hdf5_filename = output_path / "1e+085degDosePEEK/1e+085degDosePEEK_param24.92.h5"
# hdf5_filename = output_path / "1e+085degDoseidealPEEK/1e+085degDoseidealPEEK_param24.92.h5"
# hdf5_filename = output_path / "1e+085degNoNucPEEK/1e+085degNoNucPEEK_param24.92.h5"
hdf5_filename = output_path / "1e+085degActiveLayerPEEK/1e+085degActiveLayerPEEK_param24.92.h5"

# run_name = "1e+085degDosePEEK_param24.92"
# run_name = "1e+085degDoseidealPEEK_param24.92"
# run_name = "1e+085degNoNucPEEK_param24.92"
run_name = "1e+085degActiveLayerPEEK_param24.92"

dose_base_path = output_path / run_name[:run_name.rindex('_')]

# Get SRIM for comp
df = pd.read_excel(Path('/Users/nico_brosda/Cyrce_Messungen/SRIM_Simulations/') / 'SRIM_results.xlsx')
# df = pd.read_fwf(Path('/Users/nico_brosda/Desktop/AFP Stuff/Software/SRIM2013/SRIM Outputs/SRIM_results.txt'), header=1, names=['Ion Energy', 'Elec. dE/dx', 'Nuclear dE/dx', 'Projected Range', 'Longitudinal Straggling', 'Lateral Straggling', ])
df = convert_columns_to_units(df, {'Ion Energy': 'MeV', 'Projected Range': 'mm', 'Longitudinal Straggling': 'mm', 'Lateral Straggling': 'mm'})
df['Elec. dE/dx'], df['Nuclear dE/dx'] = df['Elec. dE/dx']/1e3, df['Nuclear dE/dx']/1e3

shape = LineShape([[45, 2.14], [5, 5.64]], distance_mode=False)
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
        print(f"wedge_th: {wedge_th:.2f} mm")
        print(f"wedge_th: {wedge_th_left:.2f} mm to {wedge_th_right:.2f} mm")
    wedge_thickness.append(wedge_th)

wedge_thickness = np.array(wedge_thickness)

fig, ax = plt.subplots()
ax.plot(wedge_thickness)
ax.fill_between(np.arange(len(wedge_thickness)), wedge_thickness-wedge_range, wedge_thickness+wedge_range, color='red', alpha=0.25)
ax.set_xlabel("Pixel Row")
ax.set_ylabel("Wedge thickness (mm)")
ax.set_title("Wedge thickness vs Pixel Row")
format_save(save_path, "WedgeTh_vs_row", save_format=save_format)

# --------------------------------------------------
# Function: plot map
# --------------------------------------------------

def plot_map(map_to_plot, title=""):
    fig, ax = plt.subplots()
    im = ax.imshow(map_to_plot, cmap=cmap)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Proton counts")

    return fig, ax


# --------------------------------------------------
# Compare all three PHSP planes
# --------------------------------------------------
cache_dir = dose_base_path / "cache/"
os.makedirs(cache_dir, exist_ok=True)
cache_file = cache_dir / f"{run_name}_row_energy_stats.npz"
try:
    data = np.load(cache_file)
    mean          = data["mean"]
    sigma         = data["sigma"]
    medians       = data["medians"]
    p16           = data["p16"]
    p84           = data["p84"]
    robust_width  = data["robust_width"]
    hist2d        = data["hist2d"]
    energy_edges  = data["energy_edges"]
    advanced_sigma = data["advanced_sigma"]
    ratio1        = data["ratio1"]
    ratio2        = data["ratio2"]
    peak_edges    = data["peak_edges"]
    print("Loaded cached row energy statistics.")
except (FileNotFoundError, KeyError):
    print("Cache not found. Computing row energy statistics...")
    (mean, sigma, medians, p16, p84, robust_width,
     hist2d, energy_edges, advanced_sigma,
     ratio1, ratio2, peak_edges) = compute_row_energy_statistics_fast(
        hdf5_filename,
        run_name,
        central_width_mm=10
    )
    np.savez(
        cache_file,
        mean=mean,
        sigma=sigma,
        medians=medians,
        p16=p16,
        p84=p84,
        robust_width=robust_width,
        hist2d=hist2d,
        energy_edges=energy_edges,
        advanced_sigma=advanced_sigma,
        ratio1=ratio1,
        ratio2=ratio2,
        peak_edges=peak_edges,
    )
    print("Saved row energy statistics to cache.")

energy_centers = 0.5 * (energy_edges[:-1] + energy_edges[1:])
energy_width = np.mean([energy_centers[i] - energy_centers[i + 1] for i in range(len(energy_centers) - 1)])

plt.rcParams.update({
    "text.usetex": False,
})
plt.rcParams["font.family"] = ["Arial"]


for row in tqdm(range(200)):
    fig, ax = plt.subplots()

    # Pre-create artists
    hist_line, = ax.step([], [], where="mid")
    # mean_line = ax.axvline(0, color="blue", linestyle="-")
    median_line = ax.axvline(0, color="green", linestyle="--")
    width_patch = ax.axvspan(0, 0, alpha=0.25, color="red")
    ax.set_xlabel("Energy (MeV)")
    ax.set_ylabel("Counts")

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    standard_layout([ax])

    position = 0.25 * row + 0.125
    if position < 5 or position > 45:
        wedge_th = 0
        wedge_range = 0
    else:
        wedge_th_left = shape.calculate_value(position - 0.125)
        wedge_th = shape.calculate_value(position)
        wedge_th_right = shape.calculate_value(position + 0.125)

        wedge_range = np.abs(wedge_th_right - wedge_th_left) / 2

    ydata = hist2d[row]
    counts = np.sum(ydata)

    if np.max(ydata) >= 1e3:
        ydata /= 1e3
        ax.set_ylabel("Counts (1e+3)")
    else:
        ax.set_ylabel("Counts")

    # Update histogram
    hist_line.set_data(energy_centers, ydata)

    # Update vertical lines
    '''
    mean_line.set_xdata([mean[row]])
    mean_line.set_label(f'Mean: {mean[row]:.2f} MeV')
    '''
    median_line.set_xdata([medians[row]])
    median_line.set_label(f'Median: {medians[row]:.2f} MeV')

    ax.axvline(energy_centers[np.argmax(ydata)], c='b', ls='--', label=f'Maximum bin: {energy_centers[np.argmax(ydata)]:.2f} MeV')
    '''
    # Update width band
    width_patch.remove()
    width_patch = ax.axvspan(
        p16[row], p84[row],
        alpha=0.25, color="red",
        label=f'P16 width: {robust_width[row]/2:.2f} MeV',
    )
    # '''
    width_patch.remove()
    width_patch = ax.axvspan(
        medians[row] - advanced_sigma[row], medians[row] + advanced_sigma[row],
        alpha=0.25, color="red", label=f'Sigma: {advanced_sigma[row]:.2f} MeV'
    )
    # '''

    '''
    sigma_line.remove()
    xmin = transform_data_to_axis_coordinates(ax, [medians[row] - sigma[row], 0])[0]
    xmax = transform_data_to_axis_coordinates(ax, [medians[row] + sigma[row], 0])[0]
    sigma_line = ax.axhline(np.max(ydata) *  0.607, xmin=xmin, xmax=xmax, label=f'Sigma: {sigma[row]:.2f} MeV', color='r')
    '''

    ax.axvline(energy_edges[peak_edges[row][0]], ls='--', c='k')
    ax.axvline(energy_edges[peak_edges[row][1]], ls='--', c='k')
    li, ri = peak_edges[row][0], peak_edges[row][1]
    ax.fill_between(energy_centers[li:ri], ydata[li:ri], color='yellow', label=f'Ratio: {ratio1[row]*100:.1f} %', alpha=0.4)
    ax.fill_between(energy_centers[:li], ydata[:li], color='pink', label=f'Ratio: {ratio2[row]*100: .1f} %', alpha=0.4)


    ax.set_ylim(0, ydata.max() * 1.1)
    ax.set_xlim(energy_edges[0], energy_edges[-1])

    xmin = transform_data_to_axis_coordinates(ax, [medians[row] - advanced_sigma[row], 0])[0]
    xmax = transform_data_to_axis_coordinates(ax, [medians[row] + advanced_sigma[row], 0])[0]
    sigma_line = ax.axhline(np.max(ydata) * 0.607, xmin=xmin, xmax=xmax, color='r')

    ax.set_title(f"Row {row} – wedge: {wedge_th:.2f}±{wedge_range:.2f} mm – counts: {counts:.2e}")
    ax.legend(fontsize=12)

    if save_path / (f"HistAdvanced_10mmCentral/") is not None and not os.path.exists(save_path / (f"HistAdvanced_10mmCentral/")):
        os.makedirs(save_path / (f"HistAdvanced_10mmCentral/"))
    fig.savefig(save_path / (f"HistAdvanced_10mmCentral/energy_histogram_row_{row}" + save_format), dpi=300)

    continue
    format_save(
        save_path,
        f"energy_histogram_row_{row}",
        save_format=save_format,
        legend=True
    )

plt.rcParams.update({
    "text.usetex": True,
})
plt.rcParams["font.family"] = ["Latin Modern Roman"]

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

filter_mask = counts <= 500
mean_energy[filter_mask] = 0

fig, ax = plt.subplots()
ax2 = ax.twinx()
for i, row in enumerate(mean_energy):
    if wedge_thickness[i] <= 0 or np.sum(counts[i]) < 500:
        continue
    ax.plot(i, medians[i], marker='x', color='k')
    ax2.plot(i, advanced_sigma[i], marker='o', color='r')

ax.set_xlabel("Pixel Row")
ax.set_ylabel("Median Proton energy per row (MeV)")
ax2.set_ylabel("Main population energy sigma (MeV)", color='r')
format_save(save_path, "EnergyMedianAndSigma_vs_row", save_format=save_format)

fig, ax = plt.subplots()
ax2 = ax.twinx()
for i, row in enumerate(mean_energy):
    if wedge_thickness[i] <= 0 or np.sum(counts[i]) < 500:
        continue
    ax.plot(wedge_thickness[i], medians[i], marker='x', color='k')
    ax2.plot(wedge_thickness[i], advanced_sigma[i], marker='o', color='r')

ax.set_xlabel("Wedge thickness (mm)")
ax.set_ylabel("Median Proton energy per row (MeV)")
ax2.set_ylabel("Main population energy sigma (MeV)", color='r')
format_save(save_path, "EnergyMedianAndSigma_vs_WedgeTh", save_format=save_format)


plot_map(counts, title="Counts2")
format_save(save_path, "Count_Map", save_format=save_format)


fig, ax = plt.subplots()
ax.hist(counts.flatten(), bins=50, color='k', alpha=0.7)
ax.set_xlabel("Counts per pixel")
format_save(save_path, "CountsPerPixelHist", save_format=save_format)


# --------------------------------------------------
# PARAMETERS
# --------------------------------------------------
Np = 1e8   # <-- your number of primaries


# --------------------------------------------------
# 1. Compute mean proton energy per pixel (PHSP1)
# --------------------------------------------------


# mean_energy = compute_phsp1_mean_energy(hdf5_filename, run_name)

fig, ax = plt.subplots()
im = ax.imshow(mean_energy, cmap=sns.color_palette("crest_r", as_cmap=True))
plt.colorbar(im, ax=ax, label="Mean Proton Energy (MeV)")
format_save(save_path, "Energy_Map", save_format=save_format)

# --------------------------------------------------
# 2. Load Dose and Deposited Energy images
# --------------------------------------------------

output_file = dose_base_path / f"_{run_name}_dose.mhd"
dose_img = sitk.ReadImage(str(output_file).replace(".mhd", "_dose.mhd"))
edep_img = sitk.ReadImage(str(output_file).replace(".mhd", "_edep.mhd"))

dose = sitk.GetArrayFromImage(dose_img)   # Gy
edep = sitk.GetArrayFromImage(edep_img)   # MeV

# If single layer detector, remove z dimension
if dose.ndim == 3:
    dose = dose[0]
    edep = edep[0]

# Transpose if necessary to match histogram orientation
dose = np.array(dose)
edep = np.array(edep)

fig, ax = plt.subplots()
im = ax.imshow(dose, cmap=cmap)
plt.colorbar(im, ax=ax, label="Dose per pixel (Gy)")
format_save(save_path, "Dose_Map", save_format=save_format)

fig, ax = plt.subplots()
ax.imshow(filter_mask, cmap="gray", alpha=0.5)
format_save(save_path, "CountFilterMask", save_format=save_format)

# --------------------------------------------------
# 3. Normalize to number of primaries
# --------------------------------------------------

dose_norm = dose / counts
edep_norm = edep / counts

dose_norm[filter_mask] = 0
edep_norm[filter_mask] = 0

fig, ax = plt.subplots()
for i, row in enumerate(mean_energy):
    if wedge_thickness[i] <= 0 or np.sum(counts[i]) < 500:
        continue
    edep_calc = edep_norm[i, 90:110]*1e3/4.5
    edep_mean = np.mean(edep_calc[edep_calc>0])
    edep_std = np.std(edep_calc[edep_calc>0])
    ax.errorbar(wedge_thickness[i], edep_mean, edep_std, color='k', marker='x', capsize=3)

ax.set_xlabel("Wedge thickness (mm)")
ax.set_ylabel("1 um GaN Deposited energy (keV/p+)")
format_save(save_path, "DepEnergy_vs_WedgeTh", save_format=save_format)

fig, ax = plt.subplots()
for i, row in enumerate(mean_energy):
    if wedge_thickness[i] <= 0 or np.sum(counts[i]) < 500:
        continue
    edep_calc = edep_norm[i, 90:110] * 1e3 / 4.5
    edep_mean = np.mean(edep_calc[edep_calc > 0])
    edep_std = np.std(edep_calc[edep_calc > 0])
    ax.errorbar(medians[i], edep_mean, edep_std, color='k', marker='x', capsize=3)

ax.set_xlabel("Medium proton energy behind wedge (MeV) ")
ax.set_ylabel("1 um GaN Deposited energy (keV/p+)")
format_save(save_path, "DepEnergy_vs_EMedian", save_format=save_format)

fig, ax = plt.subplots()
for i, row in enumerate(mean_energy):
    if wedge_thickness[i] <= 0 or np.sum(counts[i]) < 500:
        continue
    edep_calc = edep_norm[i, 90:110] * 1e3 / 4.5
    edep_mean = np.mean(edep_calc[edep_calc > 0])
    edep_std = np.std(edep_calc[edep_calc > 0])
    point = ax.errorbar(medians[i], edep_mean, edep_std, color='k', marker='x', capsize=3)

point.set_label('GATE behind wedge')
ax.set_xlim(ax.get_xlim())

ax.plot(df['Ion Energy'], df['Elec. dE/dx']+df['Nuclear dE/dx'], c='r', ls='--', label='SRIM in GaN')

ax.set_xlabel("Medium proton energy behind wedge (MeV) ")
ax.set_ylabel("1 um GaN Deposited energy (keV/p+)")
format_save(save_path, "DepEnergy_vs_EMedian_PlusSRIM", save_format=save_format, legend=True)

fig, ax = plt.subplots()
im = ax.imshow(dose_norm*1e3, cmap=cmap)
plt.colorbar(im, ax=ax, label="4.5 um GaN deposited dose (mGy/p+)")
format_save(save_path, "NormedDoseMap", save_format=save_format)

fig, ax = plt.subplots()
im = ax.imshow(edep_norm*1e3, cmap=cmap)
plt.colorbar(im, ax=ax, label="4.5 um GaN deposited Energy (keV/p+)")
format_save(save_path, "NormedEdepMap", save_format=save_format)

# -----------------
# ---------------------------------
# 4. Correlation Plot
# --------------------------------------------------

mean_E_flat = mean_energy.flatten()
median_E_flat = median_energy.flatten()
dose_flat = dose_norm.flatten() * 1e3
edep_flat = edep_norm.flatten() * 1e3

mask = dose_flat > 0

plt.figure()
plt.scatter(mean_E_flat[mask], dose_flat[mask], s=5, alpha=0.5)
plt.xlabel("Mean Proton Energy (MeV)")
plt.ylabel("Dose (mGy/p+)")
plt.title("4.5 um GaN Dose vs Mean Proton Energy")
format_save(save_path, "NormedDose_vs_MeanProtonEnergy", save_format=save_format)

plt.figure()
plt.scatter(median_E_flat[mask], dose_flat[mask], s=5, alpha=0.5)
plt.xlabel("Median Proton Energy (MeV)")
plt.ylabel("Dose (mGy/p+)")
plt.title("4.5 um GaN Dose vs Mean Proton Energy")
format_save(save_path, "NormedDose_vs_MedianProtonEnergy", save_format=save_format)

plt.figure()
plt.scatter(dose_flat[mask], counts.flatten()[mask], s=5, alpha=0.5)
plt.xlabel("Dose (mGy/p+)")
plt.ylabel("Counts per pixel")
plt.title("Counts per pixel vs Dose")
format_save(save_path, "CountsPerPixel_vs_NormedDose", save_format=save_format)

fig, ax = plt.subplots()
ax.plot(counts[100, :]/1e+3, c='k', ls='', marker='x')
ax.set_xlabel("Pixel")
ax.set_ylabel("Counts per pixel (1e+3)")
ax.set_title("IntensityDistributionOverAperture")
format_save(save_path, "IntensityDistributionOverAperture", save_format=save_format)

plt.figure()
plt.scatter(mean_E_flat[mask], edep_flat[mask], s=5, alpha=0.5)
plt.xlabel("Mean Proton Energy (MeV)")
plt.ylabel("Deposited Energy (keV/p+)")
plt.title("4.5 um GaN Edep vs Mean Proton Energy")
format_save(save_path, "NormedEdep_vs_MeanProtonEnergy", save_format=save_format)

plt.figure()
plt.scatter(median_E_flat[mask], edep_flat[mask], s=5, alpha=0.5)
plt.xlabel("Median Proton Energy (MeV)")
plt.ylabel("Deposited Energy (keV/p+)")
plt.title("4.5 um GaN Edep vs Median Proton Energy")
format_save(save_path, "NormedEdep_vs_MedianProtonEnergy", save_format=save_format)

fig, ax = plt.subplots()
ax.scatter(mean_E_flat[mask], sigma_energy.flatten()[mask], s=5, alpha=0.5)
plt.xlabel("Mean Proton Energy (MeV)")
plt.ylabel("Energy Sigma per pixel (MeV)")
plt.title("E sigma vs  E mean")
format_save(save_path, "Esigma_vs_Emean", save_format=save_format)

fig, ax = plt.subplots()
ax.scatter(mean_E_flat[mask], rel_spread.flatten()[mask], s=5, alpha=0.5)
plt.xlabel("Mean Proton Energy (MeV)")
plt.ylabel(r"Relative spread of energy per pixel (\%)")
plt.title("Energy Spread per pixel vs Mean Proton Energy")
format_save(save_path, "Espread_vs_Emean", save_format=save_format)

fig, ax = plt.subplots()
ax.scatter(dose_flat[mask], rel_spread.flatten()[mask], s=5, alpha=0.5)
plt.xlabel("Dose (mGy/p+)")
plt.ylabel(r"Relative spread of energy per pixel (\%)")
plt.title("Energy spread per pixel vs Dose")
format_save(save_path, "Espread_vs_Dose", save_format=save_format)
