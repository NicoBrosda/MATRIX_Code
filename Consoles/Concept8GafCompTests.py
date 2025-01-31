import numpy as np
from skimage.transform import resize, rotate
from scipy.optimize import minimize, differential_evolution
from scipy.ndimage import affine_transform
import time

func_calls = 0

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
    penalty_factor=0.,
    bounds = None,
    grid_n = 5,
    ev_pop_size = 25,
    ev_max_iter = 200
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
        Method for optimization: 'gradient' or 'evolutionary'. Default is 'gradient'.
    penalty_factor : float - specifies how much image shifting is penalized

    Returns:
    --------
    diff_image : np.ndarray
        Difference image after alignment.
    alignment_score : float
        Metric quantifying how well the images align (lower is better).
    """
    global func_calls
    func_calls = 0

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

    # assert abs((image_high_res_resampled.shape[0] * target_pixel_size) - (image_high_res.shape[0] * pixel_size_high)) < 1e-6, "Scaling error detected!"
    # assert abs((image_high_res_resampled.shape[1] * target_pixel_size) - (image_high_res.shape[1] * pixel_size_high)) < 1e-6, "Scaling error detected!"
    print(f'Pixel size of lower image: ', target_pixel_size)
    diff_y = abs((image_high_res_resampled.shape[0] * target_pixel_size) - (image_high_res.shape[0] * pixel_size_high))
    diff_x = abs((image_high_res_resampled.shape[1] * target_pixel_size) - (image_high_res.shape[1] * pixel_size_high))
    print(f'{diff_x} difference on image x axis - equal to {diff_x/target_pixel_size} pixels')
    print(f'{diff_y} difference on image y axis - equal to {diff_y/target_pixel_size} pixels')

    max_shape = (
        max(image_high_res_resampled.shape[0], image_low_res_resampled.shape[0]),
        max(image_high_res_resampled.shape[1], image_low_res_resampled.shape[1])
    )

    if not isinstance(bounds, (list, np.ndarray)):
        bounds = [bounds]
    while len(bounds) < 3:
        bounds.append(None)
    for i, bound in enumerate(bounds):
        if bound is None:
            bounds[i] = [(-5, 5), (-max_shape[0] // 10, max_shape[0] // 10), (-max_shape[1] // 10, max_shape[1] // 10)][i]


    def pad_image(image, target_shape):
        pad_y = (target_shape[0] - image.shape[0]) // 2
        pad_x = (target_shape[1] - image.shape[1]) // 2
        return np.pad(image, ((pad_y, target_shape[0] - image.shape[0] - pad_y),
                              (pad_x, target_shape[1] - image.shape[1] - pad_x)),
                      mode='constant', constant_values=0)

    image_high_res_padded = pad_image(image_high_res_resampled, max_shape)
    image_low_res_padded = pad_image(image_low_res_resampled, max_shape)

    if center_position is None:
        center_shift = [0, 0]
    else:
        center_shift = list(center_position)

    def alignment_metric(params):
        global func_calls
        func_calls += 1
        rotation, shift_x, shift_y = params
        transformed_low_res = transform_image(image_low_res_padded, rotation, [shift_x, shift_y])
        valid_mask = (image_high_res_padded > 0) | (transformed_low_res > 0)
        high_res_valid = image_high_res_padded[valid_mask]
        low_res_valid = transformed_low_res[valid_mask]

        if high_res_valid.size == 0 or low_res_valid.size == 0:
            return np.inf

        diff = high_res_valid - low_res_valid
        mse = np.mean(diff ** 2)

        shift_penalty = penalty_factor * (shift_x ** 2 + shift_y ** 2)

        return mse + shift_penalty

    if optimize_alignment:

        if optimization_method == 'gradient':
            grid_search_points = np.linspace(bounds[0][0], bounds[0][1], grid_n)
            best_init = None
            best_score = np.inf

            for r in grid_search_points:
                for sx in np.linspace(bounds[1][0], bounds[1][1], grid_n):
                    for sy in np.linspace(bounds[2][0], bounds[2][1], grid_n):
                        score = alignment_metric([r, sx, sy])
                        if score < best_score:
                            best_score = score
                            best_init = [r, sx, sy]

            result = minimize(
                alignment_metric,
                best_init,
                bounds=bounds,
                method="L-BFGS-B"
            )
            rotation, shift_x, shift_y = result.x

        elif optimization_method == 'evolutionary':
            result = differential_evolution(alignment_metric, bounds, maxiter=ev_max_iter, popsize=ev_pop_size)
            rotation, shift_x, shift_y = result.x

        else:
            raise ValueError("Unsupported optimization_method. Choose 'gradient' or 'evolutionary'.")

        center_shift = [shift_x, shift_y]

    transformed_low_res = transform_image(image_low_res_padded, rotation, center_shift)

    valid_mask = (image_high_res_padded > 0) | (transformed_low_res > 0)
    diff_image = image_high_res_padded - transformed_low_res
    diff_image[~valid_mask] = 0  # Set invalid regions to zero

    alignment_score = np.mean(diff_image[valid_mask] ** 2) if valid_mask.any() else np.inf

    return diff_image, alignment_score, (rotation, center_shift, func_calls)