import cv2
import os

import matplotlib.pyplot as plt
import numpy as np
from EvaluationSoftware.main import *
from skimage.filters import threshold_multiotsu
from skimage.transform import rotate
from scipy.ndimage import affine_transform
import time
from Concept8GafMeasurementComparison import GafImage, estimate_distribution_center, evaluate_methods
from Concept8GafCompTests import resample_image

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black", "red", "yellow"])
color_cycle = sns.color_palette("tab10")

pixel_size = 1/9600 * 2.54 * 10  # The resolution of the scanner was set to 9600 ppi (ppp en Français)
# - resulting in a pixel size of 0.0026458333333333334 mm - this is the real value to use!!!!

# Paths to the Gafchromic Scans
path1 = Path('/Users/nico_brosda/Cyrce_Messungen/Gafchromic_111024/')
gafimages1 = os.listdir(path1)
path2 = Path('/Users/nico_brosda/Cyrce_Messungen/Gafchromic_221024/')
gafimages2 = os.listdir(path2)

gafimages = gafimages1 + gafimages2

for i, gafimage in enumerate(gafimages[0:]):

    print(gafimage, gafimage[-4:])
    if not gafimage[-4:] == '.bmp':
        continue
    if i < len(gafimages1):
        full_path = path1 / gafimage
        results_path = path1
    else:
        full_path = path2 / gafimage
        results_path = path2

    # Load in the image
    start = time.time()
    GafScan = GafImage(full_path, ppi=9600)
    GafScan.load_image()
    if 'matrix211024_006.bmp' in gafimage:
        GafScan.image = GafScan.image[::-1]
        GafScan.image = GafScan.image[:, ::-1]
    image = GafScan.image
    end = time.time()
    print(end - start)

    # The transformation of the data into to 1 normed array
    thresholds = threshold_multiotsu(image)
    response_max_border, mean_GafFilm = GafScan.transform_to_normed(max_n=1e5)
    response_max_border += mean_GafFilm
    image_trans = GafScan.image

    # ------------------------------------------------------------------------------------------------------------
    # Image of Gaf after Trafo vs Dowsampled image after trafo
    # ------------------------------------------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2)
    color_map1 = ax1.imshow(image_trans, cmap=cmap, vmin=0, vmax=1,
                            extent=(0, np.shape(image)[1] * pixel_size, 0, np.shape(image)[0] * pixel_size))
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map1.cmap)
    sm.set_array([])
    bar1 = fig.colorbar(sm, ax=ax1, extend='max')
    ax1.set_xlabel('Position x (mm)')
    ax1.set_ylabel('Position y (mm)')
    ax1.set_title(f'Original Image scale with size {np.shape(image_trans)} pixels')
    bar1.set_label('Gafchromic response (normed to 1)')

    down_samp = resample_image(image_trans, pixel_size, 0.25)
    color_map2 = ax2.imshow(down_samp, cmap=cmap, vmin=0, vmax=1,
                            extent=(0, np.shape(down_samp)[1] * 0.25, 0, np.shape(down_samp)[0] * 0.25))
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map2.cmap)
    sm.set_array([])
    bar2 = fig.colorbar(sm, ax=ax2, extend='max')
    ax2.set_xlabel('Position x (mm)')
    ax2.set_ylabel('Position y (mm)')
    bar2.set_label('Gafchromic response (normed to 1)')
    ax2.set_title(f'Downsampled image with size {np.shape(down_samp)} pixels')
    plotsize = (2*latex_textwidth, latex_textwidth / 1.2419)
    format_save(results_path / 'GafTrafo/', 'DownSampComp_' + gafimage, legend=False, fig=fig, plot_size=plotsize)

    # ------------------------------------------------------------------------------------------------------------
    # Image of downsampled with and without rotation + shift
    # ------------------------------------------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2)
    color_map1 = ax1.imshow(down_samp, cmap=cmap, vmin=0, vmax=1,
                            extent=(0, np.shape(down_samp)[1] * 0.25, 0, np.shape(down_samp)[0] * 0.25))
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map1.cmap)
    sm.set_array([])
    bar1 = fig.colorbar(sm, ax=ax1, extend='max')
    ax1.set_xlabel('Position x (mm)')
    ax1.set_ylabel('Position y (mm)')
    bar1.set_label('Gafchromic response (normed to 1)')
    ax1.set_title(f'Downsampled image with size {np.shape(down_samp)} pixels')

    max_shape = (down_samp.shape[0], down_samp.shape[1])
    def pad_image(image, target_shape):
        pad_y = (target_shape[0] - image.shape[0]) // 2
        pad_x = (target_shape[1] - image.shape[1]) // 2
        return np.pad(image, ((pad_y, target_shape[0] - image.shape[0] - pad_y),
                              (pad_x, target_shape[1] - image.shape[1] - pad_x)),
                      mode='constant', constant_values=0)
    down_samp = pad_image(down_samp, max_shape)
    def transform_image(image, rotation, center_shift):
        # Rotate the image without reshape and apply center shift
        rotated_image = rotate(image, rotation, order=1, mode='reflect')
        shift_matrix = np.eye(3)
        shift_matrix[:2, 2] = center_shift
        transformed_image = affine_transform(
            rotated_image, shift_matrix[:2, :2], offset=shift_matrix[:2, 2], order=1
        )
        return transformed_image
    down_samp = transform_image(down_samp, rotation=10, center_shift=(-down_samp.shape[0]//10, -down_samp.shape[1]//10))
    valid_mask = (down_samp > 0)
    down_samp[~valid_mask] = 0  # Set invalid regions to zero

    color_map2 = ax2.imshow(down_samp, cmap=cmap, vmin=0, vmax=1,
                            extent=(0, np.shape(down_samp)[1] * 0.25, 0, np.shape(down_samp)[0] * 0.25))
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map2.cmap)
    sm.set_array([])
    bar2 = fig.colorbar(sm, ax=ax2, extend='max')
    ax2.set_xlabel('Position x (mm)')
    ax2.set_ylabel('Position y (mm)')
    bar2.set_label('Gafchromic response (normed to 1)')
    ax2.set_title(f'Downsampled image after rotation by 10° and shift by tenth its size')
    format_save(results_path / 'GafTrafo/', 'TrafoComp_' + gafimage, legend=False, fig=fig, plot_size=plotsize)

    cv2.destroyAllWindows()