from Consoles.StyleConsoles.Utils_ImageLoad import *
from PIL import Image
from Consoles.Consoles8Gafchromic.Concept8GafCompTests import (align_and_compare_images, resample_image,
                                                               transform_image, align_and_compare_images2)

# Save path and options for the map
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Style/Presentations/')
background_subtraction = True
normalization = True

zero_scale = True
intensity_limitz = [0, 1]

pixel = 'fill'

dpi = 300
format = '.svg'
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black", "red", "yellow"])
array_color = sns.color_palette("hls", 2)

image_list = []
# image_list.append((Path('/Users/nico_brosda/Cyrce_Messungen/matrix_230924/'), 'Logo', '85.0', np.array([2, 34]), np.array([50-16, 50+16])))
# image_list.append((Path('/Users/nico_brosda/Cyrce_Messungen/matrix_230924/'), 'BeamScan', '85.0', np.array([4, 36]), np.array([48-16, 48+16])))
# image_list.append((Path('/Users/nico_brosda/Cyrce_Messungen/matrix_230924/'), 'Array3_BeamShape', None, np.array([4, 36]), np.array([52-16, 52+16])))
# image_list.append((Path('/Users/nico_brosda/Cyrce_Messungen/matrix_230924/'), 'round_aperture_2_3scans', '70.0', np.array([0, 40]), np.array([32-20, 32+20])))
image_list.append((Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/'), 'Misc', None, np.array([4, 40]), np.array([16-18, 16+18])))

for obj in image_list:
    fig, ax = plt.subplots(figsize=fullsize_plot)

    # Image params
    folder_path = obj[0]
    image = obj[1]
    position = obj[2]
    x_limits = obj[3]
    y_limits = obj[4]

    A = load_image(folder_path, image, background_subtraction=background_subtraction, normalization=normalization,
                   position=position)

    if zero_scale:
        A.maps[0]['x'] = A.maps[0]['x'] - np.min(x_limits)
        A.maps[0]['y'] = A.maps[0]['y'] - np.min(y_limits)

        ext_x = 0 - np.min(x_limits)
        ext_y = 0 - np.min(y_limits)

        x_limits = x_limits - np.min(x_limits)
        y_limits = y_limits - np.min(y_limits)


    intensity_limits = [np.min(A.maps[0]['z']) * intensity_limitz[0], np.max(A.maps[0]['z']) * intensity_limitz[1]]

    mapx, mapy, mapz = homogenize_pixel_size(A.maps[0])
    pixel_size = mapx[1] - mapx[0]

    extent = (ext_x, np.shape(mapz)[1] * pixel_size + ext_x, ext_y,
        np.shape(mapz)[0] * pixel_size + ext_y)
    print(extent)
    ax.imshow(mapz[::-1], cmap=cmap, extent=extent, vmin=intensity_limits[0], vmax=intensity_limits[1])

    # A.plot_map(None, pixel=pixel, intensity_limits=intensity_limits, ax_in=ax, fig_in=fig, cmap=cmap, imshow=True)

    norm = matplotlib.colors.Normalize(vmin=intensity_limits[0], vmax=intensity_limits[1])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    if x_limits is None:
        x_limits = ax.get_xlim()
    if y_limits is None:
        y_limits = ax.get_ylim()

    print(x_limits, y_limits)
    bar = fig.colorbar(sm, ax=ax, extend='max')
    ax.set_xlabel('Position x (mm)')
    ax.set_ylabel('Position y (mm)')
    bar.set_label(f'Signal Current ({scale_dict[A.scale][1]}A)')

    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)

    '''
    x_scale = ax.get_xlim()
    y_scale = ax.get_ylim()
    if x_scale[1] - x_scale[0] < y_scale[1] - y_scale[0]:
        ax.set_xlim(x_scale[0] - (y_scale[1] - y_scale[0] - x_scale[1] + x_scale[0]) / 2,
                    x_scale[1] + (y_scale[1] - y_scale[0] - x_scale[1] + x_scale[0]) / 2)
    elif x_scale[1] - x_scale[0] > y_scale[1] - y_scale[0]:
        ax.set_ylim(y_scale[0] - (x_scale[1] - x_scale[0] - y_scale[1] + y_scale[0]) / 2,
                    y_scale[1] + (x_scale[1] - x_scale[0] - y_scale[1] + y_scale[0]) / 2)
    '''
    # ax.axhline(profile, c=array_color[0], ls='-')
    add_png_icon(ax, A, 'top left', translation='x', zoom=0.2)

    if not background_subtraction:
        image += '_RAW'
    format_save(save_path=results_path, save_name=f"Sandbox_{image}", dpi=dpi, plot_size=fullsize_plot,
                save_format=save_format, fig=fig, legend=False)
