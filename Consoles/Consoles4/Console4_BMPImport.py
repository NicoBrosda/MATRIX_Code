import cv2
import os

import matplotlib.pyplot as plt
import numpy as np
from EvaluationSoftware.main import *
from skimage.filters import threshold_multiotsu

# Some calculations handmade to get a rough impression of the Grayscale Scan pixel size
# Square aperture:
xL = np.array([3986, 3943, 3898])
xR = np.array([13567, 13524, 13474])
width = np.mean(xR-xL)
print(width)
# Square aperture has 25 mm - meaning the pixel size is
print('Square aperture pixel size (mm): ', 25 / width)

# Round aperture (Gaf 005)
dLR = 14618 - 875
dUO = 14725 - 918
print(dLR, dUO)
print((dLR + dUO) / 2)
# Round aperture has 36.51 mm - meaning the pixel size is
print('Round aperture pixel size (mm): ', 36.51/((dLR + dUO) / 2))

# This calculated pixel size aligns with a resolution of 150 dpi - pixel size = 0.0026246733333333337 - I use this value
pixel_size = 0.0026246733333333337

# Paths to the Gafchromic Scans
path1 = Path('/Users/nico_brosda/Cyrce_Messungen/Gafchromic_111024/')
gafimages1 = os.listdir(path1)
path2 = Path('/Users/nico_brosda/Cyrce_Messungen/Gafchromic_221024/')
gafimages2 = os.listdir(path2)

gafimages = gafimages1 + gafimages2

for i, gafimage in enumerate(gafimages[0:]):
    if not 'matrix211024_006.bmp' in gafimage:
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
    image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    print(np.shape(image))

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
    n, bins, patches = ax3.hist(image.flatten(), bins=500)
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
    image_trans = (image_trans - rest_max) / np.min(image_trans[ll_corner[1]:ru_corner[1], ll_corner[0]:ru_corner[0]] - rest_max)
    image_trans[image_trans > 1] = 1

    # Some additions to plot 3
    [ax3.axvline(th, color='r') for k, th in enumerate(thresholds)]
    ax3.axvline(mean_GafFilm, color='k', ls='--', label='Mean of \n Gafchromic \n = Image 0')
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
    ax2.axvline(image_middle[0]*pixel_size, ls=':', c='deepskyblue')
    ax2.axhline(image_middle[1]*pixel_size, ls=':', c='lime')

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

    # Save all
    plot_size = (latex_textwidth * 1.5, latex_textwidth * 1.5 / 1.3)
    format_save(results_path / 'GafTrafo/', gafimage, legend=False, fig=fig, axes=[ax1, ax2, ax3, ax4], plot_size=plot_size)
    # '''

    # Full image of Gafchromic Scan after trafo
    fig, ax = plt.subplots()
    if 'matrix211024_006.bmp' in gafimage:
        image_trans = image_trans[::-1]
        image_trans = image_trans[:, ::-1]
    color_map2 = ax.imshow(image_trans, cmap=cmap, vmin=0, vmax=1,
                            extent=(0, np.shape(image)[1] * pixel_size, 0, np.shape(image)[0] * pixel_size))
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map2.cmap)
    sm.set_array([])
    bar = fig.colorbar(sm, ax=ax, extend='max')
    ax.set_xlabel('Position x (mm)')
    ax.set_ylabel('Position y (mm)')
    bar.set_label('Gafchromic response (normed to 1)')
    format_save(results_path / 'GafTrafo/', 'FullImage_'+gafimage, legend=False, fig=fig, axes=[ax])

    cv2.destroyAllWindows()