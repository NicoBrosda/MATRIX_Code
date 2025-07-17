import cv2
import os

import matplotlib.pyplot as plt
import numpy as np
from EvaluationSoftware.main import *
from skimage.filters import threshold_multiotsu
import time
from Concept8GafMeasurementComparison import GafImage, estimate_distribution_center, evaluate_methods

color_cycle = sns.color_palette("tab10")

pixel_size = 1/9600 * 2.54 * 10  # The resolution of the scanner was set to 9600 ppi (ppp en Français)
# - resulting in a pixel size of 0.0026458333333333334 mm - this is the real value to use!!!!

# Paths to the Gafchromic Scans
path1 = Path('/Users/nico_brosda/Cyrce_Messungen/Gafchromic_111024/')
gafimages1 = os.listdir(path1)
path2 = Path('/Users/nico_brosda/Cyrce_Messungen/Gafchromic_221024/')
gafimages2 = os.listdir(path2)

gafimages = gafimages1 + gafimages2

for i, gafimage in enumerate(gafimages):

    if not 'gafchromic_matrix211024_007' in gafimage:
        continue
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

    thresholds = threshold_multiotsu(image)

    # Full image of GafData Hist before trafo
    fig, ax = plt.subplots()
    # Plot 3: Histogram of image data
    n, bins, patches = ax.hist(image.flatten(), bins=200, color='k')
    bins = np.array([bins[k] + (bins[k + 1] - bins[k]) / 2 for k in range(len(bins) - 1)])
    ax.set_xlabel('Greyscale value')

    # Some additions to plot 3
    [ax.axvline(th, color='m', label='GafImage Greycale \n thresholds (Multi Otsu)', ls='--') if k == 0
     else ax.axvline(th, color='m', ls='--')
     for k, th in enumerate(thresholds)]

    summary = evaluate_methods(image, (thresholds[0], 255))
    for j, method in enumerate(['hist_peak', 'median', 'kde_peak', 'fwhm']):
        start = time.time()
        center = estimate_distribution_center(image, (thresholds[0], 255), method=method)
        time_consumed = time.time() - start
        ax.axvline(center, label=f'{method} - time: {time_consumed: .4f} s \n result: {center: .2f} \n Bootstrap Std: {summary[method]["Std Dev"]: .2f}', color=color_cycle[j], alpha=0.8)

    ax.legend()
    format_save(results_path / 'GafTrafo/', 'DistributionTest_' + gafimage, legend=False, fig=fig)

    # Generate the plot
    fig, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2)

    # Plot 1: Original Greyscale scan of Gafchromic
    color_map = ax1.imshow(image, cmap='gray', vmin=0, vmax=255, extent=(0, np.shape(image)[1]*pixel_size, 0, np.shape(image)[0]*pixel_size))
    norm = matplotlib.colors.Normalize(vmin=0, vmax=255)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map.cmap)
    sm.set_array([])
    bar = fig.colorbar(sm, ax=ax1, extend='max')
    ax1.set_xlabel('Position x (mm)')
    ax1.set_ylabel('Position y (mm)')

    # Plot 3: Histogram of image data
    n, bins, patches = ax3.hist(image.flatten(), bins=200, color='k')
    bins = np.array([bins[k]+(bins[k+1] - bins[k])/2 for k in range(len(bins)-1)])
    ax3.set_xlabel('Greyscale value')


    # The transformation of the data into to 1 normed array
    thresholds = threshold_multiotsu(image)

    image_trans = deepcopy(image) * 1.0  # This is a necessary change of the Greyscales np array data format to float!!!
    mean_GafFilm = bins[(bins > thresholds[0]) & (bins < thresholds[1])][np.argmax(n[(bins > thresholds[0]) & (bins < thresholds[1])])]
    image_trans[image_trans > mean_GafFilm] = mean_GafFilm
    # Now it is better to estimate the images maximum in a radius around the center to exclude the notes
    radius = 10
    image_middle = np.array([np.shape(image_trans)[1] // 2, np.shape(image_trans)[0] // 2])
    ll_corner, ru_corner = image_middle - radius//pixel_size, image_middle + radius//pixel_size
    ll_corner, ru_corner = np.array(ll_corner, dtype=np.int64), np.array(ru_corner, dtype=np.int64)

    print(image_middle, ll_corner, ru_corner)
    rest_max = np.max(image_trans)
    '''
    image_trans = (image_trans - rest_max) / np.min(image_trans[ll_corner[1]:ru_corner[1], ll_corner[0]:ru_corner[0]] - rest_max)
    image_trans[image_trans > 1] = 1
    # '''
    response_max_border, mean_GafFilm = GafScan.transform_to_normed(max_n=1e5)
    response_max_border += mean_GafFilm
    image_trans = GafScan.image

    # Some additions to plot 3
    # Some additions to plot 3
    [ax3.axvline(th, color='k', label='Thresholds \n (Multi Otsu)', ls='--') if k == 0
     else ax3.axvline(th, color='k', ls='--')
     for k, th in enumerate(thresholds)]

    ax3.axvline(mean_GafFilm, color='b', ls='-', label='Trafo 0')
    ax3.axvline(response_max_border, color='r', ls='-', label='Trafo 1')
    ax3.legend()

    # Plot 2: Image with conventional colour scale after trafo
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black", "red", "yellow"])
    color_map2 = ax2.imshow(image_trans, cmap=cmap, vmin=0, vmax=1, extent=(0, np.shape(image)[1]*pixel_size, 0, np.shape(image)[0]*pixel_size))
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map2.cmap)
    sm.set_array([])
    bar2 = fig.colorbar(sm, ax=ax2, extend='max')
    ax2.set_xlabel('Position x (mm)')
    ax2.set_ylabel('Position y (mm)')

    # Some additional plots for the function usage
    ax1.scatter(*(image_middle * pixel_size), marker='x', c='orange')
    box = Rectangle(ll_corner * pixel_size, radius * 2, radius * 2, ls="--", lw=0.8, ec="orange", fc="none")
    ax1.add_artist(box)
    ax2.scatter(*(image_middle*pixel_size), marker='x', c='k', alpha=0.2)
    box = Rectangle(ll_corner*pixel_size, radius*2, radius*2, ls="--", lw=0.8, ec="k", fc="none", alpha=0.2)
    ax2.add_artist(box)
    # ax2.axvline(image_middle[0]*pixel_size, ls=':', c='deepskyblue')
    # ax2.axhline(image_middle[1]*pixel_size, ls=':', c='lime')

    # """
    # Plot ax 4: Gaf Hist after Trafo
    n, bins, patches = ax4.hist(image_trans.flatten(), bins=100, color='k')
    bins = np.array([bins[k] + (bins[k + 1] - bins[k]) / 2 for k in range(len(bins) - 1)])
    ax4.set_xlabel('Gafchromic response (normed to 1)')

    ax4.axvline(0, color='b', ls='-', label='Mean of \n Gafchromic \n = Image 0')
    ax4.axvline(1, color='r', ls='-', label='Max of \n response \n = Image 1')
    ax4.legend()
    '''
    # Plot 4: Cross sections
    ax4.plot([k*pixel_size for k in range(np.shape(image_trans)[1])], image_trans[image_middle[1], :], c='lime', label='X cross section')
    ax4.plot([k*pixel_size for k in range(np.shape(image_trans)[0])], image_trans[:, image_middle[0]], c='deepskyblue', label='Y cross section')
    ax4.set_xlabel('Pixel')
    ax4.set_ylabel('Gafchromic signal normed to one')
    ax4.legend()
    # ax4.hist(image_trans.flatten(), bins=100)
    # ax4.set_xlabel('Normed to one')
    # ax4.axvline(0, color='lime', ls='--')
    # ax4.set_ylabel('Hist data cropped image')
    '''

    # Save all
    plot_size = (latex_textwidth * 1.5, latex_textwidth * 1.5 / 1.3)
    format_save(results_path / 'GafTrafo/', gafimage, legend=False, fig=fig, plot_size=plot_size)
    # '''

    # Full image of Gafchromic Scan after trafo
    fig, ax = plt.subplots()
    color_map2 = ax.imshow(image_trans, cmap=cmap, vmin=0, vmax=1,
                            extent=(0, np.shape(image)[1] * pixel_size, 0, np.shape(image)[0] * pixel_size))
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map2.cmap)
    sm.set_array([])
    bar = fig.colorbar(sm, ax=ax, extend='max')
    ax.set_xlabel('Position x (mm)')
    ax.set_ylabel('Position y (mm)')
    bar.set_label('Gafchromic response (normed to 1)')
    format_save(results_path / 'GafTrafo/', 'FullImage_'+gafimage, legend=False, fig=fig)

    # Full image of GafData Hist before trafo
    fig, ax = plt.subplots()
    # Plot 3: Histogram of image data
    n, bins, patches = ax.hist(image.flatten(), bins=200, color='k')
    bins = np.array([bins[k] + (bins[k + 1] - bins[k]) / 2 for k in range(len(bins) - 1)])
    ax.set_xlabel('Greyscale value')

    # Some additions to plot 3
    [ax.axvline(th, color='k', label='GafImage Greycale \n thresholds (Multi Otsu)', ls='--') if k == 0
     else ax.axvline(th, color='k', ls='--')
     for k, th in enumerate(thresholds)]

    ax.axvline(mean_GafFilm, color='b', ls='-', label='Mean of Gafchromic \n  = Image 0')
    ax.axvline(response_max_border, color='r', ls='-', label='Max of Gafchromic \n (Mean from 1e5 points) \n = Image 1')
    ax.legend()
    format_save(results_path / 'GafTrafo/', 'HistBefore_' + gafimage, legend=False, fig=fig)

    # Full image of GafData Hist before trafo
    fig, ax = plt.subplots()
    # Plot 3: Histogram of image data
    n, bins, patches = ax.hist(image.flatten(), bins=200, color='k')
    bins = np.array([bins[k] + (bins[k + 1] - bins[k]) / 2 for k in range(len(bins) - 1)])
    ax.set_xlabel('Greyscale value')

    format_save(results_path / 'GafTrafo/', 'HistBeforePlain_' + gafimage, legend=False, fig=fig)
    # """

    # Image of Gaf after Trafo vs Dowsampled image after trafo
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
    ax1.set_title(f'Downsampled image with size {np.shape(down_samp)} pixels')
    format_save(results_path / 'GafTrafo/', 'DownSampComp_' + gafimage, legend=False, fig=fig)

    cv2.destroyAllWindows()