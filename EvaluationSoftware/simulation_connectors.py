from EvaluationSoftware.helper_modules import *
import SimpleITK as sitk
import h5py
from scipy.optimize import curve_fit
from scipy.constants import e

def simulation_response(run_name, diff=200):
    dat = pd.read_csv(Path(f'../../Files/energies_after_wheel_diffusor{diff}.txt'), header=4, delimiter='\t', decimal='.',
                      names=['pos', 'thickness', 'energy'])
    comp_list = dat['thickness']

    # ----------------------------------------------------------------------------------------------------------------
    # Obtaining the data
    # ----------------------------------------------------------------------------------------------------------------
    data_cache = []
    response_cache = []
    std_cache = []
    energy_cache = []
    energy_std_cache = []
    for i, param in enumerate(comp_list):
        _run_name = f"{run_name}{param}"
        current_path = Path('/Users/nico_brosda/GateSimulation/GATE10/Simulation/output/')
        output_path = current_path / f'{run_name[0:run_name.index("_")]}/'
        output_file = output_path / f"_{_run_name}_dose.mhd"

        # ----------------- Load the Dose / Edep image ----------------------------------
        # img = sitk.ReadImage(str(output_file).replace(".mhd", "_dose.mhd"))
        img = sitk.ReadImage(str(output_file).replace(".mhd", "_edep.mhd"))
        img_std = sitk.ReadImage(str(output_file).replace(".mhd", "_edep_uncertainty.mhd"))
        data = np.array(sitk.GetArrayFromImage(img))[0][::-1, :]
        data_std = np.array(sitk.GetArrayFromImage(img_std))[0][::-1, :]
        response_ind = np.argsort(data.flatten())[-500:]
        response = data.flatten()[response_ind]
        response_std = data_std.flatten()[response_ind].mean()
        data_cache.append(data)
        response_cache.append(response.mean())
        std_cache.append(np.sqrt((response.std()/np.sqrt(len(response)))**2 + response_std**2))

        # ----------------- Load the energy information ----------------------------------
        hdf5_filename = f"{output_path}/{_run_name}.h5"
        try:
            with h5py.File(hdf5_filename, "r") as f:
                group_path = f"{_run_name}/phsp2"
                group = f[group_path]

                ''' Not needed right now
                E_sample = group["E_positive_sample"][...]
                hist_counts = group["E_hist_counts"][...]
                hist_bins = group["E_hist_bins"][...]
                hist_counts_comp = group["E_hist_counts_comp"][...]
                hist_bins_comp = group["E_hist_bins_comp"][...]
                '''
                if "E_stat_median" in group.attrs:
                    E_stat_median = group.attrs["E_stat_median"]
                    E_stat_std = group.attrs["E_stat_std"]
                else:
                    print('Distribution parameters are only calculated out of sampled set!')
                    E_sample = group["E_positive_sample"][...]
                    E_stat_median = np.median(E_sample)
                    E_stat_std = np.std(E_sample)
            energy_cache.append(E_stat_median)
            energy_std_cache.append(E_stat_std)
        except Exception as e:
            print(f"Error loading data from {hdf5_filename}: {e}")
            energy_cache.append(dat['energy'][i])
            energy_std_cache.append(0)
            continue

    return response_cache, std_cache, energy_cache, energy_std_cache


pixel_size = 50/200


def simulation_response2(run_name):
    folder_path = Path('/Users/nico_brosda/GateSimulation/GATE10/Simulation/output/')
    files = os.listdir(folder_path / f'{run_name[0:run_name.index("_")]}/')
    files = array_txt_file_search(files, blacklist=['.mhd', '.raw'], searchlist=[run_name], txt_file=False)
    # ----------------------------------------------------------------------------------------------------------------
    # Obtaining the data
    # ----------------------------------------------------------------------------------------------------------------
    param_list = []
    data_cache = []
    response_cache = []
    std_cache = []
    for i, file in enumerate(files):
        try:
            param = float(file[file.index('_param')+6:])
        except ValueError:
            continue

        # ----------------- Load the Dose / Edep image ----------------------------------
        # img = sitk.ReadImage(str(output_file).replace(".mhd", "_dose.mhd"))
        try:
            _run_name = f"{run_name}{param}"
            output_path = folder_path / f'{run_name[0:run_name.index("_")]}/'
            output_file = output_path / f"_{_run_name}_dose.mhd"

            img = sitk.ReadImage(str(output_file).replace(".mhd", "_edep.mhd"))
            img_std = sitk.ReadImage(str(output_file).replace(".mhd", "_edep_uncertainty.mhd"))
        except RuntimeError:
            param = int(param)
            _run_name = f"{run_name}{param}"
            output_path = folder_path / f'{run_name[0:run_name.index("_")]}/'
            output_file = output_path / f"_{_run_name}_dose.mhd"

            img = sitk.ReadImage(str(output_file).replace(".mhd", "_edep.mhd"))
            img_std = sitk.ReadImage(str(output_file).replace(".mhd", "_edep_uncertainty.mhd"))

        param_list.append(param)

        data = np.array(sitk.GetArrayFromImage(img))[0][::-1, :]
        data_std = np.array(sitk.GetArrayFromImage(img_std))[0][::-1, :]
        '''
        if np.max(data) < 0.1:
            response = 0
            response_std = 0
        else:
            # threshold = threshold_otsu(data)
            # response = np.mean(data[data > threshold])
            # response_std = np.sqrt(np.mean(data_std[data > threshold]) ** 2 + np.std(data[data > threshold]) ** 2)
        '''
        radius_mm = 8
        height, width = data.shape
        center_y, center_x = height // 2, width // 2
        radius_px = int(radius_mm / pixel_size)
        y, x = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        mask = dist_from_center <= radius_px
        response = np.mean(data[mask])
        print(np.mean(data_std[mask]), np.std(data[mask]))
        response_std = np.sqrt(np.mean(data_std[mask]) ** 2 + (np.std(data[mask])/np.sqrt(len(data[mask]))) ** 2)

        data_cache.append(data)
        response_cache.append(response)
        std_cache.append(response_std)

    indices = np.argsort(param_list)
    return np.array(param_list)[indices], np.array(response_cache)[indices], np.array(std_cache)[indices]


def get_sim(run_name, diff=200, pixel_size=50/200, mean_range=(20, 30), simn=1e+6, param=0, mirror=True):

    # ----------------------------------------------------------------------------------------------------------------
    # Obtaining the data
    # ----------------------------------------------------------------------------------------------------------------
    _run_name = f"{run_name}{param}"
    current_path = Path('/Users/nico_brosda/GateSimulation/GATE10/Simulation/output/')
    output_path = current_path / f'{run_name[0:run_name.index("_")]}/'
    output_file = output_path / f"_{_run_name}_dose.mhd"

    # ----------------- Load the Dose / Edep image ----------------------------------
    # img = sitk.ReadImage(str(output_file).replace(".mhd", "_dose.mhd"))
    print('-'*50)
    output_path2 = Path(str(output_file).replace(".mhd", "_edep.mhd"))
    print(f"Checking file: {output_path2}")
    print(f"Exists? {output_path2.exists()}")
    print(output_path)
    print(f"Exists? {output_path.exists()}")

    img = sitk.ReadImage(str(output_file).replace(".mhd", "_edep.mhd"))
    img_std = sitk.ReadImage(str(output_file).replace(".mhd", "_edep_uncertainty.mhd"))
    if mirror:
        data = np.array(sitk.GetArrayFromImage(img))[0][::-1, :] / simn
        data_std = np.array(sitk.GetArrayFromImage(img_std))[0][::-1, :] / simn
    else:
        data = np.array(sitk.GetArrayFromImage(img))[0][:, :] / simn
        data_std = np.array(sitk.GetArrayFromImage(img_std))[0][:, :] / simn
    line = np.mean(data[:, int(mean_range[0] / pixel_size):int(mean_range[1] / pixel_size)], axis=1)[:]
    line_std = np.mean(data_std[:, int(mean_range[0] / pixel_size):int(mean_range[1] / pixel_size)], axis=1)[:]

    return data, line, line_std


def compute_phsp1_energy_stats(
        hdf5_filename,
        run_name,
        detector_size_mm=50.0,
        n_pixels=200,
        min_counts=5):
    """
    Compute per-pixel proton energy statistics from PHSP1.

    Returns:
        mean_energy      : 2D array (MeV)
        sigma_energy     : 2D array (MeV)
        relative_spread  : 2D array (sigma / mean)
        counts           : 2D array (number of protons per pixel)
        valid_mask       : boolean 2D array (counts >= min_counts)
    """

    # --------------------------------------------------
    # 1. Load PHSP1 data
    # --------------------------------------------------
    with h5py.File(hdf5_filename, "r") as f:
        group = f[f"{run_name}/phsp1"]
        X = group["X"][:]
        Y = group["Y"][:]
        E = group["E"][:]  # MeV

    # --------------------------------------------------
    # 2. Define detector geometry
    # --------------------------------------------------
    half_size = detector_size_mm / 2.0
    edges = np.linspace(-half_size, half_size, n_pixels + 1)

    # --------------------------------------------------
    # 3. Histogramming
    # --------------------------------------------------
    counts, _, _ = np.histogram2d(X, Y, bins=[edges, edges])
    energy_sum, _, _ = np.histogram2d(X, Y, bins=[edges, edges], weights=E)
    energy_sq_sum, _, _ = np.histogram2d(X, Y, bins=[edges, edges], weights=E**2)

    # Transpose to (y, x) image convention
    counts = counts.T
    energy_sum = energy_sum.T
    energy_sq_sum = energy_sq_sum.T

    # --------------------------------------------------
    # 4. Compute statistics
    # --------------------------------------------------
    mean_energy = np.zeros_like(energy_sum)
    median_energy = np.zeros_like(energy_sum)
    sigma_energy = np.zeros_like(energy_sum)
    relative_spread = np.zeros_like(energy_sum)

    valid_mask = counts >= min_counts

    # Mean
    mean_energy[valid_mask] = energy_sum[valid_mask] / counts[valid_mask]

    # Population variance:  σ² = <E²> − <E>²
    mean_E2 = np.zeros_like(energy_sq_sum)
    mean_E2[valid_mask] = energy_sq_sum[valid_mask] / counts[valid_mask]

    variance = np.zeros_like(mean_energy)
    variance[valid_mask] = mean_E2[valid_mask] - mean_energy[valid_mask]**2

    # Numerical protection against tiny negative values
    variance[variance < 0] = 0.0

    sigma_energy[valid_mask] = np.sqrt(variance[valid_mask])

    # Relative spread (avoid division by zero)
    nonzero_mean = valid_mask & (mean_energy > 0)
    relative_spread[nonzero_mean] = (
        sigma_energy[nonzero_mean] / mean_energy[nonzero_mean]
    )

    # --------------------------------------------------
    # Median computation
    # --------------------------------------------------
    # Assign each proton to a pixel index
    x_idx = np.digitize(X, edges) - 1
    y_idx = np.digitize(Y, edges) - 1

    # Keep only protons inside detector bounds
    inside = (
            (x_idx >= 0) & (x_idx < n_pixels) &
            (y_idx >= 0) & (y_idx < n_pixels)
    )

    x_idx = x_idx[inside]
    y_idx = y_idx[inside]
    E_inside = E[inside]

    # Convert to flat pixel index
    flat_idx = y_idx * n_pixels + x_idx

    # Sort by pixel index (groups equal pixels together)
    order = np.argsort(flat_idx)
    flat_idx = flat_idx[order]
    E_sorted = E_inside[order]

    # Find pixel boundaries
    unique_pixels, start_indices, counts_per_pixel = np.unique(
        flat_idx, return_index=True, return_counts=True
    )

    for pix, start, c in zip(unique_pixels, start_indices, counts_per_pixel):
        if c >= min_counts:
            y = pix // n_pixels
            x = pix % n_pixels
            median_energy[y, x] = np.median(E_sorted[start:start + c])

    return mean_energy, median_energy, sigma_energy, relative_spread*100, counts, valid_mask


def compute_row_energy_statistics_fast(
        hdf5_filename,
        run_name,
        detector_size_mm=50.0,
        n_pixels=200,
        n_energy_bins=200,
        central_width_mm=None):

    # --------------------------------------------------
    # 1. Load PHSP1
    # --------------------------------------------------
    with h5py.File(hdf5_filename, "r") as f:
        group = f[f"{run_name}/phsp1"]
        X = group["X"][:]
        Y = group["Y"][:]
        E = group["E"][:]

    half_size = detector_size_mm / 2.0
    y_edges = np.linspace(-half_size, half_size, n_pixels + 1)

    # --------------------------------------------------
    # 2. Optional central band restriction
    # --------------------------------------------------
    if central_width_mm is not None:
        half_band = central_width_mm / 2.0
        band_mask = (X >= -half_band) & (X <= half_band)
        Y = Y[band_mask]
        E = E[band_mask]

    # --------------------------------------------------
    # 2. Row index (vectorized)
    # --------------------------------------------------
    row_indices = np.digitize(Y, y_edges) - 1
    valid = (row_indices >= 0) & (row_indices < n_pixels)

    row_indices = row_indices[valid]
    E = E[valid]

    # --------------------------------------------------
    # 3. Row statistics (mean + sigma via bincount)
    # --------------------------------------------------
    counts = np.bincount(row_indices, minlength=n_pixels)
    energy_sum = np.bincount(row_indices, weights=E, minlength=n_pixels)
    energy_sq_sum = np.bincount(row_indices, weights=E**2, minlength=n_pixels)

    mean = np.zeros(n_pixels)
    sigma = np.zeros(n_pixels)

    mask = counts > 0
    mean[mask] = energy_sum[mask] / counts[mask]

    variance = np.zeros(n_pixels)
    variance[mask] = (energy_sq_sum[mask] / counts[mask]) - mean[mask]**2
    variance[variance < 0] = 0.0
    sigma[mask] = np.sqrt(variance[mask])

    # --------------------------------------------------
    # 4. 2D histogram (row vs energy)
    # --------------------------------------------------
    e_min = np.min(E)
    e_max = np.max(E)
    energy_edges = np.linspace(e_min, e_max, n_energy_bins + 1)
    energy_centers = 0.5 * (energy_edges[:-1] + energy_edges[1:])

    hist2d, _, _ = np.histogram2d(
        row_indices,
        E,
        bins=[np.arange(n_pixels + 1), energy_edges]
    )

    # --------------------------------------------------
    # 5. Median + Percentiles from histogram
    # --------------------------------------------------
    medians = np.zeros(n_pixels)
    p16 = np.zeros(n_pixels)
    p84 = np.zeros(n_pixels)

    for row in range(n_pixels):

        hist = hist2d[row]
        total = np.sum(hist)

        if total == 0:
            continue

        cdf = np.cumsum(hist) / total

        medians[row] = np.interp(0.5, cdf, energy_centers)
        p16[row] = np.interp(0.16, cdf, energy_centers)
        p84[row] = np.interp(0.84, cdf, energy_centers)

    robust_width = p84 - p16

    # --------------------------------------------------
    # 6. Advanced statistics (FAST + histogram-only)
    # --------------------------------------------------

    advanced_sigma = np.zeros(n_pixels)
    ratio_main = np.zeros(n_pixels)
    ratio_parasitic = np.zeros(n_pixels)
    peak_edges = []

    threshold_fraction = 0.02  # 5% of peak height

    for row in range(n_pixels):

        ydata = hist2d[row]
        total_counts = np.sum(ydata)

        if total_counts == 0:
            peak_edges.append([0, 0])
            continue

        # -------- Edge detection --------
        peak_idx = np.argmax(ydata)
        threshold = ydata[peak_idx] * threshold_fraction

        # ---- walk left from peak ----
        left_edge = peak_idx
        while left_edge > 0 and ydata[left_edge - 1] > threshold:
            left_edge -= 1

        # ---- walk right from peak ----
        right_edge = peak_idx
        while right_edge < len(ydata) - 1 and ydata[right_edge + 1] > threshold:
            right_edge += 1

        peak_edges.append([left_edge, right_edge])

        # --- Main vs parasitic counts ---
        main_counts = np.sum(ydata[left_edge:right_edge + 1])
        parasitic_counts = total_counts - main_counts

        ratio_main[row] = main_counts / total_counts
        ratio_parasitic[row] = parasitic_counts / total_counts

        # --- Weighted sigma from histogram bins ---
        energies = energy_centers[left_edge:right_edge + 1]
        weights = ydata[left_edge:right_edge + 1]

        if np.sum(weights) > 0:
            mean_local = np.average(energies, weights=weights)
            variance_local = np.average((energies - mean_local) ** 2, weights=weights)
            advanced_sigma[row] = np.sqrt(variance_local)

    return (
        mean,
        sigma,
        medians,
        p16,
        p84,
        robust_width,
        hist2d,
        energy_edges,
        advanced_sigma,
        ratio_main,
        ratio_parasitic,
        peak_edges
    )


# SRIM connectors
UNIT_CONVERSIONS = {
    'MeV': {'MeV': 1, 'keV': 1e-3, 'GeV': 1e3, 'eV': 1e-6},
    'mm': {'mm': 1, 'um': 1e-3, 'µm': 1e-3, 'cm': 10, 'nm': 1e-6, 'A': 1e-7, 'm': 1e3},
    # Add more base units as needed
}

def convert_with_unit(val, target_unit):
    """Convert a string like '500.0 keV' to float in the target unit (e.g., 'MeV')."""
    if pd.isna(val):
        return None
    try:
        parts = val.strip().split()
        if len(parts) != 2:
            raise ValueError(f"Cannot parse value: '{val}'")
        number_str, unit = parts
        number = float(number_str)
        factor = UNIT_CONVERSIONS.get(target_unit, {}).get(unit)
        if factor is None:
            raise ValueError(f"No conversion from '{unit}' to '{target_unit}'")
        return number * factor
    except Exception as e:
        raise ValueError(f"Error processing '{val}': {e}")


def convert_columns_to_units(df, column_unit_map):
    """Convert specified DataFrame columns to target units.

    Args:
        df: pandas DataFrame
        column_unit_map: dict like {'Energy': 'MeV', 'Distance': 'mm'}

    Returns:
        New DataFrame with converted values (same columns overwritten).
    """
    df = df.copy()
    for col, target_unit in column_unit_map.items():
        df[col] = df[col].apply(lambda val: convert_with_unit(val, target_unit))
    return df