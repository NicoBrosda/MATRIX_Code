from EvaluationSoftware.main import *
import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu
import time
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit


def gaussian(x, a, mu, sigma):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def estimate_distribution_center(image_data, area, method="hist_peak"):
    """
    Estimate the center of a distribution in image data using different methods.

    Parameters:
    - image_data (np.ndarray): Input image data.
    - area (tuple): (threshold1, upper_border) defining the range of interest.
    - method (str): One of ['hist_peak', 'median', 'kde_peak', 'fwhm'].

    Returns:
    - float: Estimated center of the distribution.
    """
    threshold1, upper_border = area
    filtered_data = image_data[(image_data > threshold1) & (image_data <= upper_border)]
    if filtered_data.size == 0:
        return None  # Return None if no data in the range

    start_time = time.time()
    center = None

    if method == "hist_peak":
        # Histogram-based peak detection + Gaussian fit
        hist, bin_edges = np.histogram(filtered_data, bins=30)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        peak_idx = np.argmax(hist)
        initial_mu = bin_centers[peak_idx]

        try:
            popt, _ = curve_fit(gaussian, bin_centers, hist, p0=[np.max(hist), initial_mu, np.std(filtered_data)])
            center = popt[1]  # Mean of the Gaussian fit
        except:
            center = initial_mu  # Fallback to histogram peak if fit fails

    elif method == "median":
        center = np.median(filtered_data)

    elif method == "kde_peak":
        # Limit KDE computation to a subset if too many points
        sample_size = min(len(filtered_data), 5000)
        sample_data = np.random.choice(filtered_data, sample_size, replace=False)
        kde = gaussian_kde(sample_data)
        x_vals = np.linspace(threshold1, upper_border, 500)  # Reduce evaluation points
        density = kde(x_vals)
        center = x_vals[np.argmax(density)]

    elif method == "fwhm":
        sample_size = min(len(filtered_data), 5000)
        sample_data = np.random.choice(filtered_data, sample_size, replace=False)
        kde = gaussian_kde(sample_data)
        x_vals = np.linspace(threshold1, upper_border, 500)  # Reduce evaluation points
        density = kde(x_vals)
        peak_idx = np.argmax(density)
        peak_value = density[peak_idx]
        half_max = peak_value / 2

        left_idx = np.where(density[:peak_idx] < half_max)[0]
        right_idx = np.where(density[peak_idx:] < half_max)[0]

        if len(left_idx) > 0 and len(right_idx) > 0:
            left_fwhm = x_vals[left_idx[-1]]
            right_fwhm = x_vals[peak_idx + right_idx[0]]
            center = (left_fwhm + right_fwhm) / 2
        else:
            center = x_vals[peak_idx]  # Fallback to peak if FWHM fails

    end_time = time.time()
    # print(f"Method: {method}, Estimated Center: {center:.2f}, Time: {end_time - start_time:.5f} sec")

    return center


def evaluate_methods(image_data, area, n_bootstrap=100):
    """
    Evaluate different estimation methods using bootstrap resampling.

    Parameters:
    - image_data (np.ndarray): Input image data.
    - area (tuple): (threshold1, upper_border) defining the range of interest.
    - n_bootstrap (int): Number of bootstrap samples to estimate variability.

    Returns:
    - dict: Mean estimated center and standard deviation for each method.
    """
    methods = ["hist_peak", "median", "kde_peak", "fwhm"]
    results = {method: [] for method in methods}

    filtered_data = image_data[(image_data > area[0]) & (image_data <= area[1])]
    if filtered_data.size == 0:
        print("No data in specified area.")
        return {}

    sample_size = max(len(filtered_data) // 100, 100)  # Limit subset size

    for _ in tqdm(range(n_bootstrap)):
        bootstrap_sample = np.random.choice(filtered_data, size=sample_size, replace=True)
        for method in methods:
            estimated_center = estimate_distribution_center(bootstrap_sample, area, method)
            if estimated_center is not None:
                results[method].append(estimated_center)

    summary = {}
    print("\nBootstrap Evaluation Results:")
    for method, estimates in results.items():
        mean_center = np.mean(estimates)
        std_dev = np.std(estimates)
        summary[method] = {"Mean Estimated Center": mean_center, "Std Dev": std_dev}
        print(f"{method}: Mean Center = {mean_center:.2f}, Std Dev = {std_dev:.2f}")

    return summary


class GafImage:
    def __init__(self, path_to_image, ppi=9600):
        self.pixel_size = 2.54 * 10 / ppi
        self.path = Path(path_to_image)
        self.name = self.path.name[:-len(self.path.suffix)]
        self.image = np.array([])

    def load_image(self):
        if ((self.path.parent / 'QuickLoads/').exists() and
                os.path.isfile(self.path.parent / ('QuickLoads/' + self.name + '.npy'))):
            self.path = self.path.parent / ('QuickLoads/' + self.name + '.npy')
        print(self.path, self.path.suffix, self.path.suffix=='.bmp')
        if self.path.suffix == '.bmp':
            self.image = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
            print(self.image)
            cv2.destroyAllWindows()
        elif self.path.suffix == '.npy':
            self.image = np.load(self.path)
        else:
            print('The image format is not supported so far!')

    def save_image(self, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        save_path = Path(save_path) / (self.name+'.npy')
        np.save(save_path, self.image)
        self.path = save_path

    def show(self, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black", "red", "yellow"])):
        fig, ax = plt.subplots()
        if np.max(self.image) > 1:
            color_map2 = ax.imshow(self.image, cmap='gray', vmin=0, vmax=255,
                                   extent=(
                                   0, np.shape(self.image)[1] * self.pixel_size, 0, np.shape(self.image)[0] * self.pixel_size))
            norm = matplotlib.colors.Normalize(vmin=0, vmax=255)
        else:
            color_map2 = ax.imshow(self.image, cmap=cmap, vmin=0, vmax=1,
                                   extent=(
                                   0, np.shape(self.image)[1] * self.pixel_size, 0, np.shape(self.image)[0] * self.pixel_size))
            norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map2.cmap)
        sm.set_array([])
        bar = fig.colorbar(sm, ax=ax, extend='max')
        ax.set_xlabel('Position x (mm)')
        ax.set_ylabel('Position y (mm)')
        bar.set_label('Gafchromic response')
        format_save(None, save=False, legend=False, fig=fig, axes=[ax])
        plt.show()
        plt.close('all')

    def transform_to_normed(self, max_n=0):
        start = time.time()

        # The transformation of the data into to 1 normed array
        thresholds = threshold_multiotsu(self.image)

        print('Thresholds: ', time.time()-start)
        start = time.time()

        self.image = self.image * 1.0  # This is a necessary change of the Greyscales np array data format to float!!!

        print('Image floatization: ', time.time() - start)
        start = time.time()

        # mean_GafFilm = bins[(bins > thresholds[0]) & (bins < thresholds[1])][np.argmax(n[(bins > thresholds[0]) & (bins < thresholds[1])])]
        mean_GafFilm = np.median(self.image[self.image > thresholds[0]])
        self.image[self.image > mean_GafFilm] = mean_GafFilm

        print('Gaf Mean: ', time.time() - start)
        start = time.time()

        # Now it is better to estimate the response maximum (greyscale minimum) in a radius around the center
        radius = 10
        image_middle = np.array([np.shape(self.image)[1] // 2, np.shape(self.image)[0] // 2])
        ll_corner, ru_corner = image_middle - radius // self.pixel_size, image_middle + radius // self.pixel_size
        ll_corner, ru_corner = np.array(ll_corner, dtype=np.int64), np.array(ru_corner, dtype=np.int64)
        image_region = self.image[ll_corner[1]:ru_corner[1], ll_corner[0]:ru_corner[0]]

        print('Image middle region: ', time.time() - start)
        start = time.time()

        # print(image_middle, ll_corner, ru_corner)
        rest_max = np.max(self.image)  # This is the right border in the image histogram (response lower border)
        print(rest_max)
        max_n = int(max_n)
        if max_n <= 1:
            min_value = np.min(image_region - rest_max)
            print('MinValue: ', min_value)
        elif max_n < np.multiply(*np.shape(self.image)):
            min_value = np.mean(np.sort(image_region.flatten() - rest_max)[0:max_n])
            print('MinValues: ', np.sort(image_region.flatten() - rest_max)[0:max_n])
        else:
            min_value = np.mean(np.sort(image_region.flatten() - rest_max)[0:int(np.multiply(*np.shape(self.image)))])
            print('2MinValues: ', np.sort(image_region.flatten() - rest_max)[0:int(np.multiply(*np.shape(self.image)))])

        print(min_value)
        print('Max min calc: ', time.time() - start)
        start = time.time()

        # Image left border in the greyscale data (upper border in the response) -> critical for comparison !!!
        self.image = (self.image - rest_max) / min_value
        self.image[self.image > 1] = 1

        print('Image rebase: ', time.time() - start)

        return min_value, rest_max



'''
Test = GafImage('/Users/nico_brosda/Cyrce_Messungen/Gafchromic_221024/gafchromic_matrix211024_001.bmp')
start = time.time()
Test.load_image()
end = time.time()
print(end-start)

start = time.time()
Test.transform_to_normed()
end = time.time()
print(end-start)

Test.show()

# Length scale: 0, np.shape(self.image)[1] * pixel_size, 0, np.shape(self.image)[0] * pixel_size)
# Position corresponding to pixel x=i, y=j: Pos(image[j, i]) = ((i+1/2) * pixel_size, j * pixel_size)

# Position of pixel i, j in image map data: Pos(map[j, i]) = x_pos[i], y_pos[j] or after splitting array into equally
# sized pixel: Pos(map[j, i]) =
# '''
import numpy as np
from scipy.ndimage import rotate, affine_transform
from scipy.optimize import minimize
from skimage.transform import resize


def align_and_compare_images(
    image_low_res,
    image_high_res,
    pixel_size_low,
    pixel_size_high,
    center_position=None,
    rotation=0,
    optimize_alignment=False,
    output_resolution="high"
):
    """
    Align and compare two images, considering spatial extents and resolution differences.

    Parameters:
    -----------
    image_low_res : np.ndarray
        Lower-resolution image (values normalized between 0 and 1).
    image_high_res : np.ndarray
        Higher-resolution image (values normalized between 0 and 1).
    pixel_size_low : float
        Pixel size of the lower-resolution image.
    pixel_size_high : float
        Pixel size of the higher-resolution image.
    center_position : tuple (optional)
        (x, y) position to align the center of the images. Default is None.
    rotation : float (optional)
        Rotation angle (in degrees) to apply to the low-resolution image. Default is 0.
    optimize_alignment : bool (optional)
        Whether to optimize the alignment (minimize difference). Default is False.
    output_resolution : str (optional)
        "high" or "low" to specify the resolution of the output difference image. Default is "high".

    Returns:
    --------
    diff_image : np.ndarray
        Difference image after alignment.
    alignment_score : float
        Metric quantifying how well the images align (lower is better).
    """

    # Step 1: Preprocess inputs (resample images to common coordinate system)
    def resample_image(image, original_pixel_size, target_pixel_size):
        scale_factor = original_pixel_size / target_pixel_size
        output_shape = (np.array(image.shape) * scale_factor).astype(int)
        return resize(image, output_shape, mode='reflect', anti_aliasing=True)

    # Resample low-res image to high-res or vice versa
    if output_resolution == "high":
        image_low_res_resampled = resample_image(image_low_res, pixel_size_low, pixel_size_high)
        image_high_res_resampled = image_high_res
        target_pixel_size = pixel_size_high
    elif output_resolution == "low":
        image_high_res_resampled = resample_image(image_high_res, pixel_size_high, pixel_size_low)
        image_low_res_resampled = image_low_res
        target_pixel_size = pixel_size_low
    else:
        raise ValueError("output_resolution must be 'high' or 'low'.")

    # Step 2: Apply initial alignment (rotation and translation)
    def transform_image(image, rotation, center_shift):
        # Rotate image
        rotated_image = rotate(image, rotation, reshape=False, order=1)
        # Shift image to align center
        shift_matrix = np.eye(3)
        shift_matrix[:2, 2] = center_shift
        transformed_image = affine_transform(
            rotated_image, shift_matrix[:2, :2], offset=shift_matrix[:2, 2], order=1
        )
        return transformed_image

    # Initialize transformation parameters (center and rotation)
    center_shift = np.array([0, 0]) if center_position is None else np.array(center_position)

    # Step 3: Define optimization function if alignment is enabled
    def alignment_metric(params):
        rotation, shift_x, shift_y = params
        transformed_low_res = transform_image(image_low_res_resampled, rotation, [shift_x, shift_y])

        # Find overlapping region dynamically
        overlap = np.minimum(image_high_res_resampled.shape, transformed_low_res.shape)
        high_res_cropped = image_high_res_resampled[:overlap[0], :overlap[1]]
        low_res_cropped = transformed_low_res[:overlap[0], :overlap[1]]

        # Compute alignment score (e.g., MSE)
        diff = high_res_cropped - low_res_cropped
        return np.mean(diff**2)

    # Perform optimization if required
    if optimize_alignment:
        initial_params = [rotation, 0, 0]  # Initial rotation and center shift
        bounds = [(-10, 10), (-50, 50), (-50, 50)]  # Reasonable bounds for optimization
        # bounds = [(-10, 10), (-500, 500), (-500, 500)]  # Reasonable bounds for optimization

        result = minimize(alignment_metric, initial_params, bounds=bounds, method="L-BFGS-B")

        # Update rotation and center_shift with optimized values
        rotation, shift_x, shift_y = result.x
        center_shift = [shift_x, shift_y]

    # Apply final transformation to low-res image
    transformed_low_res = transform_image(image_low_res_resampled, rotation, center_shift)

    # Step 4: Compute the difference image
    overlap = np.minimum(image_high_res_resampled.shape, transformed_low_res.shape)
    high_res_cropped = image_high_res_resampled[:overlap[0], :overlap[1]]
    low_res_cropped = transformed_low_res[:overlap[0], :overlap[1]]
    diff_image = high_res_cropped - low_res_cropped

    # Step 5: Calculate alignment score for the final alignment
    alignment_score = np.mean((high_res_cropped - low_res_cropped) ** 2)

    return diff_image, alignment_score


import numpy as np
from scipy.ndimage import rotate, affine_transform
from scipy.optimize import minimize
from skimage.transform import resize
from tqdm import tqdm
from joblib import Parallel, delayed
import time


def align_and_compare_imagesv2(
    image_low_res,
    image_high_res,
    pixel_size_low,
    pixel_size_high,
    center_position=None,
    rotation=0,
    optimize_alignment=False,
    output_resolution="high",
    force_downscaling=True,
    max_runtime_minutes=10
):
    """
    Align and compare two images, considering spatial extents and resolution differences.

    Parameters:
    -----------
    image_low_res : np.ndarray
        Lower-resolution image (values normalized between 0 and 1).
    image_high_res : np.ndarray
        Higher-resolution image (values normalized between 0 and 1).
    pixel_size_low : float
        Pixel size of the lower-resolution image.
    pixel_size_high : float
        Pixel size of the higher-resolution image.
    center_position : tuple (optional)
        (x, y) position to align the center of the images. Default is None.
    rotation : float (optional)
        Rotation angle (in degrees) to apply to the low-resolution image. Default is 0.
    optimize_alignment : bool (optional)
        Whether to optimize the alignment (minimize difference). Default is False.
    output_resolution : str (optional)
        "high" or "low" to specify the resolution of the output difference image. Default is "high".
    force_downscaling : bool (optional)
        Whether to force downscaling for high-res output if runtime estimation exceeds a threshold. Default is True.
    max_runtime_minutes : int (optional)
        Maximum allowed runtime (in minutes) for the optimization process. Default is 10.

    Returns:
    --------
    diff_image : np.ndarray
        Difference image after alignment.
    alignment_score : float
        Metric quantifying how well the images align (lower is better).
    """

    # Step 1: Preprocess inputs (resample images to common coordinate system)
    def resample_image(image, original_pixel_size, target_pixel_size):
        scale_factor = original_pixel_size / target_pixel_size
        output_shape = (np.array(image.shape) * scale_factor).astype(int)
        return resize(image, output_shape, mode='reflect', anti_aliasing=True)

    # Estimate runtime for optimization
    def estimate_runtime(image_shape, num_iterations):
        num_pixels = np.prod(image_shape)
        estimated_time = num_pixels * num_iterations * 1e-8  # Empirical factor for scaling
        return estimated_time / 60  # Convert to minutes

    # Resample low-res image to high-res or vice versa
    if output_resolution == "high":
        if force_downscaling:
            # Estimate runtime based on image size and optimization iterations
            estimated_runtime = estimate_runtime(image_high_res.shape, 50)
            if estimated_runtime > max_runtime_minutes:
                print(f"Estimated runtime ({estimated_runtime:.2f} minutes) exceeds threshold. Downscaling high-res image for optimization.")
                high_res_downscaled = resample_image(image_high_res, pixel_size_high, pixel_size_low)
                output_resolution = "low"
            else:
                high_res_downscaled = image_high_res
        else:
            high_res_downscaled = image_high_res

        image_low_res_resampled = resample_image(image_low_res, pixel_size_low, pixel_size_high)
        image_high_res_resampled = high_res_downscaled
        target_pixel_size = pixel_size_high

    elif output_resolution == "low":
        start = time.time()
        image_high_res_resampled = resample_image(image_high_res, pixel_size_high, pixel_size_low)
        end  = time.time()
        print(f'Downscaling takes {end-start} seconds')
        image_low_res_resampled = image_low_res
        target_pixel_size = pixel_size_low
    else:
        raise ValueError("output_resolution must be 'high' or 'low'.")

    # Step 2: Apply initial alignment (rotation and translation)
    def transform_image(image, rotation, center_shift):
        # Rotate image
        rotated_image = rotate(image, rotation, reshape=False, order=1)
        # Shift image to align center
        shift_matrix = np.eye(3)
        shift_matrix[:2, 2] = center_shift
        transformed_image = affine_transform(
            rotated_image, shift_matrix[:2, :2], offset=shift_matrix[:2, 2], order=1
        )
        return transformed_image

    # Initialize transformation parameters (center and rotation)
    center_shift = np.array([0, 0]) if center_position is None else np.array(center_position)

    # Step 3: Define optimization function if alignment is enabled
    def alignment_metric(params):
        rotation, shift_x, shift_y = params
        transformed_low_res = transform_image(image_low_res_resampled, rotation, [shift_x, shift_y])

        # Find overlapping region dynamically
        overlap = np.minimum(image_high_res_resampled.shape, transformed_low_res.shape)
        high_res_cropped = image_high_res_resampled[:overlap[0], :overlap[1]]
        low_res_cropped = transformed_low_res[:overlap[0], :overlap[1]]

        # Compute alignment score (e.g., MSE)
        diff = high_res_cropped - low_res_cropped
        return np.mean(diff**2)

    # Parallelize metric evaluations
    def parallel_alignment_metric(params_list):
        results = Parallel(n_jobs=-1)(
            delayed(alignment_metric)(params) for params in params_list
        )
        return results

    # Perform optimization if required
    if optimize_alignment:
        print("Starting optimization...")
        initial_params = [rotation, 0, 0]  # Initial rotation and center shift
        bounds = [(-10, 10), (-50, 50), (-50, 50)]  # Reasonable bounds for optimization
        start_time = time.time()

        with tqdm(total=50, desc="Optimization Progress") as pbar:
            def callback(params):
                pbar.update(1)

            result = minimize(
                alignment_metric,
                initial_params,
                bounds=bounds,
                method="L-BFGS-B",
                callback=callback
            )

        elapsed_time = (time.time() - start_time) / 60
        print(f"Optimization completed in {elapsed_time:.2f} minutes.")

        # Update rotation and center_shift with optimized values
        rotation, shift_x, shift_y = result.x
        center_shift = [shift_x, shift_y]

    # Apply final transformation to low-res image
    transformed_low_res = transform_image(image_low_res_resampled, rotation, center_shift)

    # Step 4: Compute the difference image
    overlap = np.minimum(image_high_res_resampled.shape, transformed_low_res.shape)
    high_res_cropped = image_high_res_resampled[:overlap[0], :overlap[1]]
    low_res_cropped = transformed_low_res[:overlap[0], :overlap[1]]
    diff_image = high_res_cropped - low_res_cropped

    # Step 5: Calculate alignment score for the final alignment
    alignment_score = np.mean((high_res_cropped - low_res_cropped) ** 2)

    return diff_image, alignment_score


def align_and_compare_imagesv3(
    image_low_res,
    image_high_res,
    pixel_size_low,
    pixel_size_high,
    center_position=None,
    rotation=0,
    optimize_alignment=False,
    optimization_method='gradient',
    output_resolution="high",
    force_downscaling=True,
    max_runtime_minutes=10
):
    """
    Align and compare two images, considering spatial extents and resolution differences.

    Parameters:
    -----------
    image_low_res : np.ndarray
        Lower-resolution image (values normalized between 0 and 1).
    image_high_res : np.ndarray
        Higher-resolution image (values normalized between 0 and 1).
    pixel_size_low : float
        Pixel size of the lower-resolution image.
    pixel_size_high : float
        Pixel size of the higher-resolution image.
    center_position : tuple (optional)
        (x, y) position to align the center of the images. Default is None.
    rotation : float (optional)
        Rotation angle (in degrees) to apply to the low-resolution image. Default is 0.
    optimize_alignment : bool (optional)
        Whether to optimize the alignment (minimize difference). Default is False.
    optimization_method : str (optional)
        Method for optimization: 'gradient', 'global_gradient', or 'evolutionary'. Default is 'gradient'.
    output_resolution : str (optional)
        "high" or "low" to specify the resolution of the output difference image. Default is "high".
    force_downscaling : bool (optional)
        Whether to force downscaling for high-res output if runtime estimation exceeds a threshold. Default is True.
    max_runtime_minutes : int (optional)
        Maximum allowed runtime (in minutes) for the optimization process. Default is 10.

    Returns:
    --------
    diff_image : np.ndarray
        Difference image after alignment.
    alignment_score : float
        Metric quantifying how well the images align (lower is better).
    """

    import numpy as np
    from skimage.transform import resize, rotate
    from scipy.optimize import minimize, differential_evolution
    from scipy.ndimage import affine_transform
    import time

    def resample_image(image, original_pixel_size, target_pixel_size):
        scale_factor = original_pixel_size / target_pixel_size
        output_shape = (np.array(image.shape) * scale_factor).astype(int)
        return resize(image, output_shape, mode='reflect', anti_aliasing=True)

    def transform_image(image, rotation, center_shift):
        # Rotate the image without reshape and apply center shift
        rotated_image = rotate(image, rotation, order=1, mode='reflect')
        shift_matrix = np.eye(3)
        shift_matrix[:2, 2] = center_shift
        transformed_image = affine_transform(
            rotated_image, shift_matrix[:2, :2], offset=shift_matrix[:2, 2], order=1
        )
        return transformed_image

    start_time = time.time()

    if output_resolution == "high":
        if force_downscaling:
            high_res_downscaled = resample_image(image_high_res, pixel_size_high, pixel_size_low)
            image_high_res_resampled = high_res_downscaled
            image_low_res_resampled = resample_image(image_low_res, pixel_size_low, pixel_size_high)
            target_pixel_size = pixel_size_high
        else:
            image_high_res_resampled = image_high_res
            image_low_res_resampled = resample_image(image_low_res, pixel_size_low, pixel_size_high)
            target_pixel_size = pixel_size_high

    elif output_resolution == "low":
        image_high_res_resampled = resample_image(image_high_res, pixel_size_high, pixel_size_low)
        image_low_res_resampled = image_low_res
        target_pixel_size = pixel_size_low
    else:
        raise ValueError("output_resolution must be 'high' or 'low'.")

    if center_position is None:
        center_shift = [0, 0]
    else:
        center_shift = list(center_position)

    def alignment_metric(params):
        rotation, shift_x, shift_y = params
        transformed_low_res = transform_image(image_low_res_resampled, rotation, [shift_x, shift_y])

        overlap = np.minimum(image_high_res_resampled.shape, transformed_low_res.shape)
        high_res_cropped = image_high_res_resampled[:overlap[0], :overlap[1]]
        low_res_cropped = transformed_low_res[:overlap[0], :overlap[1]]

        diff = high_res_cropped - low_res_cropped
        mse = np.mean(diff**2)

        # Allow wider shifts but penalize large shifts less aggressively
        shift_penalty = 0.001 * (shift_x**2 + shift_y**2)
        return mse + shift_penalty

    if optimize_alignment:
        if optimization_method == 'gradient':
            result = minimize(
                alignment_metric,
                [rotation, center_shift[0], center_shift[1]],
                bounds=[(-20, 20), (-100, 100), (-100, 100)],
                method="L-BFGS-B"
            )
            rotation, shift_x, shift_y = result.x

        elif optimization_method == 'evolutionary':
            bounds = [(-20, 20), (-100, 100), (-100, 100)]
            result = differential_evolution(alignment_metric, bounds, maxiter=200, popsize=25)
            rotation, shift_x, shift_y = result.x

        else:
            raise ValueError("Unsupported optimization_method. Choose 'gradient' or 'evolutionary'.")

        center_shift = [shift_x, shift_y]

    transformed_low_res = transform_image(image_low_res_resampled, rotation, center_shift)

    overlap = np.minimum(image_high_res_resampled.shape, transformed_low_res.shape)
    high_res_cropped = image_high_res_resampled[:overlap[0], :overlap[1]]
    low_res_cropped = transformed_low_res[:overlap[0], :overlap[1]]
    diff_image = high_res_cropped - low_res_cropped

    alignment_score = np.mean((high_res_cropped - low_res_cropped) ** 2)

    return diff_image, alignment_score


'''
# Example usage (with dummy data)
low_res_image = np.random.rand(100, 100)
low_res_image = np.zeros((100, 100))
low_res_image[45:55, 45:55] = 1.5
high_res_image = np.random.rand(300, 300)
high_res_image = np.zeros((300, 300))
high_res_image[180:210, 130:160] = 1


diff, score = align_and_compare_images(low_res_image, high_res_image, 0.1, 0.03, optimize_alignment=True)
print("Alignment Score:", score)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.imshow(low_res_image)
ax2.imshow(high_res_image)
ax3.imshow(diff)

plt.show()
'''
