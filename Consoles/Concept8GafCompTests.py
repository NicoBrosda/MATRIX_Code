import numpy as np
from skimage.transform import resize, rotate
from scipy.optimize import minimize, differential_evolution
from scipy.ndimage import affine_transform
import time


def resample_image(image, original_pixel_size, target_pixel_size):
    scale_factor = original_pixel_size / target_pixel_size
    output_shape = (np.array(image.shape) * scale_factor).astype(int)
    return resize(image, output_shape, mode='reflect', anti_aliasing=True)


def align_and_compare_images(
    image_low_res,
    image_high_res,
    pixel_size_low,
    pixel_size_high,
    image_down_sampled = None,
    center_position=None,
    rotation=0,
    optimize_alignment=False,
    optimization_method='gradient',
    penalty_factor=0.001,
    bounds = [(-20, 20), (-100, 100), (-100, 100)]
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
    penalty_factor : float - specifies how much image shifting is penalized

    Returns:
    --------
    diff_image : np.ndarray
        Difference image after alignment.
    alignment_score : float
        Metric quantifying how well the images align (lower is better).
    """

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

    if image_down_sampled is None:
        image_high_res_resampled = resample_image(image_high_res, pixel_size_high, pixel_size_low)
    else:
        image_high_res_resampled = image_down_sampled
    image_low_res_resampled = image_low_res
    target_pixel_size = pixel_size_low

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
        shift_penalty = penalty_factor * (shift_x**2 + shift_y**2)
        return mse + shift_penalty

    if optimize_alignment:

        if optimization_method == 'gradient':
            result = minimize(
                alignment_metric,
                [rotation, center_shift[0], center_shift[1]],
                bounds=bounds,
                method="L-BFGS-B"
            )
            rotation, shift_x, shift_y = result.x

        elif optimization_method == 'evolutionary':
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