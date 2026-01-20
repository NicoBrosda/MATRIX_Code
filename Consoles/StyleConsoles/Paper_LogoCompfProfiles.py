from Consoles.StyleConsoles.Utils_ImageLoad import *
from PIL import Image
from Consoles.Consoles8Gafchromic.Concept8GafCompTests import (align_and_compare_images, resample_image,
                                                               transform_image, align_and_compare_images2)

# Save path and options for the map
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Style/Paper/')
background_subtraction = True
normalization = True

plot_size = (18 * cm, 2 / 2 * 18 / 1.2419 * cm)
# Setup of the final figure
# Structure: Ax1 Logo Line Super Res - Ax2 Logo Gaf - Ax3 BeamShape 2-Line - Ax4 BeamShape Gaf -
# Ax5 Logo Overlay - Ax6 Beam Overlay
fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=plot_size)
# Adjust spacing: left, right, bottom, top, wspace, hspace
fig.subplots_adjust(wspace=0.3, hspace=0.25)
# Axis limits
y_limits = []
x_limits = []
zero_scale = True
intensity_limits = None
intensity_limitsg = [0, 1]

pixel = 'fill'

dpi = 300
format = '.png'
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black", "red", "yellow"])
array_color = sns.color_palette("hls", 2)

# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

# Axis limits
x_limit1 = np.array([22 - 18, 22 + 14])
y_limit1 = np.array([23.5 - 15, 23.5 + 17])

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_230924/')
image = 'Logo'
position = '85.0'
A1 = load_image(folder_path, image, background_subtraction=background_subtraction, normalization=normalization,
               position=position)
A1.maps[0] = overlap_treatment(A1.maps[0], A1, super_res=True)
# Correction to make measurements comparable | I = 2.385
A1.maps[0]['z'] = A1.maps[0]['z']*0.95* (2/2.385) / 0.82

print(np.shape(A1.maps[0]['z']))

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_230924/')
image = 'Array3_Logo'
position = '85.0'

A2 = load_image(folder_path, image, background_subtraction=background_subtraction, normalization=normalization,
                position=position)
A2.maps[0]['y'] = A2.maps[0]['y'] - 110 + 66.5
mid = np.argsort(np.abs(A2.maps[0]['y'] - 66.5))[0]
dist = len(A2.maps[0]['y']) - mid
A2.maps[0]['y'] = A2.maps[0]['y'][-2 * dist:]
A2.maps[0]['z'] = A2.maps[0]['z'][-2 * dist:]
# '''

A1.maps[0]['y'] = A1.maps[0]['y'][-55:]
A1.maps[0]['z'] = A1.maps[0]['z'][-55:]
A1.maps[0]['y'], A1.maps[0]['z'] = np.append(A1.maps[0]['y'], A1.maps[0]['y'][-1]+0.25), np.append(A1.maps[0]['z'], [np.zeros_like(A1.maps[0]['z'][-1])], axis=0)

for i, image_map in enumerate(A2.maps):
    A2.maps[i]['z'] = simple_zero_replace(image_map['z'])

A2.maps[0] = overlap_treatment(A2.maps[0], A2, True)

print(np.shape(A2.maps[0]['z']))

# Selection of the image (automatic assigning of the Analyzer)
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/Gafchromic_221024/')
image = 'gafchromic_matrix211024_006.bmp'
position = None
gaf1 = load_image(folder_path, image, background_subtraction=background_subtraction, normalization=normalization,
                 position=position)
DownSampGaf = GafImage(folder_path / image)
quick_load = ((folder_path / image).parent / 'QuickLoads') / (image[:-4] + '.npy')
low_pixel_size = 0.25

if os.path.isfile(quick_load):
    print('Quick Load')
    DownSampGaf.load_image(quick=True)
    down_samp1 = DownSampGaf.image
else:
    down_samp1 = resample_image(gaf1.image, gaf1.pixel_size, low_pixel_size)
    DownSampGaf.image = down_samp1
    DownSampGaf.save_image(quick_load.parent)


_x, _y, _z = homogenize_pixel_size(
    [A2.maps[0]['x'], A2.maps[0]['y'], np.abs(A2.maps[0]['z']) / np.max(A2.maps[0]['z'])])
diff2, score2, addition2 = align_and_compare_images2(_z[::-1], gaf1.image, low_pixel_size, gaf1.pixel_size,
                                                     optimize_alignment=True, bounds=(-3, 3), ev_max_iter=1000,
                                                     ev_pop_size=10, optimization_method='evolutionary',
                                                     image_down_sampled=down_samp1)

gaf1.image = addition2[-2]
gaf1.pixel_size = low_pixel_size

_x, _y, _z = homogenize_pixel_size(
    [A1.maps[0]['x'], A1.maps[0]['y'], np.abs(A1.maps[0]['z']) / np.max(A1.maps[0]['z'])])
diff1, score1, addition1 = align_and_compare_images(_z[::-1], gaf1.image, low_pixel_size, gaf1.pixel_size,
                                                     optimize_alignment=True, bounds=(-5, 5), ev_max_iter=10000,
                                                     ev_pop_size=10, optimization_method='evolutionary',
                                                     image_down_sampled=gaf1.image)

fig2, [axx1, axx2] = plt.subplots(1, 2, figsize=(10, 5))

axx1.imshow(diff1, cmap=cmap, vmin=-1, vmax=1)

axx2.imshow(diff2, cmap=cmap, vmin=-1, vmax=1)

plt.show()

print(np.shape(diff1), np.min(diff1), np.max(diff1))
print(score1)

if zero_scale:
    ext_x = 0 - np.min(x_limit1)
    ext_y = 0 - np.min(y_limit1)

else:
    ext_x = 0
    ext_y = 0

exp_image2 = addition2[-1]
exp_image1 = addition1[-1]
print(np.shape(gaf1.image))
print(np.shape(exp_image1))
print(np.shape(exp_image2))

# Profile location
profile = 15
px = (profile - ext_y)/low_pixel_size
signal3 = gaf1.image[-int(px), :]
signal2 = exp_image2[-int(px), :]
signal1 = exp_image1[-int(px), :]

y_pos1 = np.arange(ext_x, ext_x + np.shape(signal3)[0] * low_pixel_size, low_pixel_size)

# ------------------------------------------------------------------------------------------------------------------
# Ax1: Logo 128x0.5x0.5 image
# ------------------------------------------------------------------------------------------------------------------
ax = ax1

x_limit = x_limit1
y_limit = y_limit1

if zero_scale:
    ext_x = 0 - np.min(x_limit)
    ext_y = 0 - np.min(y_limit)

    x_limit = x_limit - np.min(x_limit)
    y_limit = y_limit - np.min(y_limit)
else:
    ext_x = 0
    ext_y = 0

color_map = ax.imshow(exp_image1*np.max(A1.maps[0]['z']), cmap=cmap, vmin=0, vmax=np.max(A1.maps[0]['z']), extent=(
    ext_x, np.shape(gaf1.image)[1] * gaf1.pixel_size + ext_x, ext_y, np.shape(gaf1.image)[0] * gaf1.pixel_size + ext_y))

norm = matplotlib.colors.Normalize(vmin=0, vmax=np.max(A1.maps[0]['z']))
sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map.cmap)
sm.set_array([])
bar = fig.colorbar(sm, ax=ax, extend='max')
ax.set_xlabel('Position x (mm)')
ax.set_ylabel('Position y (mm)')
bar.set_label(f'Signal Current ({scale_dict[A1.scale][1]}A)')

x_limits.append(x_limit)
y_limits.append(y_limit)

ax.axhline(profile, c=array_color[0], ls='-')

# ------------------------------------------------------------------------------------------------------------------
# Ax2: Logo 128x0.25x0.5 image
# ------------------------------------------------------------------------------------------------------------------
ax = ax2

x_limit = x_limit1
y_limit = y_limit1

if zero_scale:
    ext_x = 0 - np.min(x_limit)
    ext_y = 0 - np.min(y_limit)

    x_limit = x_limit - np.min(x_limit)
    y_limit = y_limit - np.min(y_limit)
else:
    ext_x = 0
    ext_y = 0

color_map = ax.imshow(exp_image2*np.max(A2.maps[0]['z']), cmap=cmap, vmin=0, vmax=np.max(A2.maps[0]['z']), extent=(
    ext_x, np.shape(gaf1.image)[1] * gaf1.pixel_size + ext_x, ext_y, np.shape(gaf1.image)[0] * gaf1.pixel_size + ext_y))

norm = matplotlib.colors.Normalize(vmin=0, vmax=np.max(A2.maps[0]['z']))
sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map.cmap)
sm.set_array([])
bar = fig.colorbar(sm, ax=ax, extend='max')
ax.set_xlabel('Position x (mm)')
ax.set_ylabel('Position y (mm)')
bar.set_label(f'Signal Current ({scale_dict[A2.scale][1]}A)')

x_limits.append(x_limit)
y_limits.append(y_limit)

ax.axhline(profile, c=array_color[1], ls='-')

# ------------------------------------------------------------------------------------------------------------------
# Ax3: Logo Gafchromic
# ------------------------------------------------------------------------------------------------------------------
ax = ax3

# Axis limits
x_limit = x_limit1
y_limit = y_limit1

if zero_scale:
    ext_x = 0 - np.min(x_limit)
    ext_y = 0 - np.min(y_limit)

    x_limit = x_limit - np.min(x_limit)
    y_limit = y_limit - np.min(y_limit)
else:
    ext_x = 0
    ext_y = 0

color_map = ax.imshow(gaf1.image, cmap=cmap, vmin=0, vmax=1, extent=(
    ext_x, np.shape(gaf1.image)[1] * gaf1.pixel_size + ext_x, ext_y, np.shape(gaf1.image)[0] * gaf1.pixel_size + ext_y))
norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map.cmap)
sm.set_array([])
bar = fig.colorbar(sm, ax=ax, extend='max')
ax.set_xlabel('Position x (mm)')
ax.set_ylabel('Position y (mm)')
bar.set_label('Normalized Response')

x_limits.append(x_limit)
y_limits.append(y_limit)

ax.axhline(profile, c='k', ls='--')

# ------------------------------------------------------------------------------------------------------------------
# Ax4: Profile Logo
# ------------------------------------------------------------------------------------------------------------------
ax = ax4

ax.plot(y_pos1, signal1, c=array_color[0], ls='-')
ax.plot(y_pos1, signal2, c=array_color[1], ls='-')

ax.plot(y_pos1, signal3, c='k', ls='--')

ax.set_xlabel('Position x (mm)')
ax.set_ylabel('Normalized Response')
ax.set_xlim(2, 30)

# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

for i, ax in enumerate([ax1, ax2, ax3]):
    if x_limits[i] is None:
        x_limits[i] = ax.get_xlim()
    if y_limits[i] is None:
        y_limits[i] = ax.get_ylim()

    ax.set_xlim(x_limits[i])
    ax.set_ylim(y_limits[i])
    x_scale = ax.get_xlim()
    y_scale = ax.get_ylim()
    if x_scale[1] - x_scale[0] < y_scale[1] - y_scale[0]:
        ax.set_xlim(x_scale[0] - (y_scale[1] - y_scale[0] - x_scale[1] + x_scale[0]) / 2,
                    x_scale[1] + (y_scale[1] - y_scale[0] - x_scale[1] + x_scale[0]) / 2)
    elif x_scale[1] - x_scale[0] > y_scale[1] - y_scale[0]:
        ax.set_ylim(y_scale[0] - (x_scale[1] - x_scale[0] - y_scale[1] + y_scale[0]) / 2,
                    y_scale[1] + (x_scale[1] - x_scale[0] - y_scale[1] + y_scale[0]) / 2)

for ax in fig.axes:
    if is_colorbar(ax):
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    else:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

'''
add_diode_geometry_indicator(ax2, A2, position='upper right', fig=fig)
add_diode_geometry_indicator(ax3, A3, position='upper right', fig=fig)
add_diode_geometry_indicator(ax4, A4, position='upper right', fig=fig)
'''

add_png_icon(ax1, A1, 'top left', translation='x', zoom=0.2)
add_png_icon(ax2, A2, 'top left', translation='x', zoom=0.2)

ax3.text(*transform_axis_to_data_coordinates(ax3, [0.03, 0.97]), r'\textbf{Gafchromic}', fontsize=9, ha='left',
         va='top', color='k')

ax1.text(*transform_axis_to_data_coordinates(ax1, [0.97, 0.97]), r'\textbf{(a)}', fontsize=9, ha='right',
         va='top', color='k')
ax2.text(*transform_axis_to_data_coordinates(ax2, [0.97, 0.97]), r'\textbf{(b)}', fontsize=9, ha='right',
         va='top', color='k')
ax3.text(*transform_axis_to_data_coordinates(ax3, [0.97, 0.97]), r'\textbf{(c)}', fontsize=9, ha='right',
         va='top', color='k')
ax4.text(*transform_axis_to_data_coordinates(ax4, [0.97, 0.97]), r'\textbf{(d)}', fontsize=9, ha='right',
         va='top', color='k')

'''
ax1.text(*transform_axis_to_data_coordinates(ax1, [0.5, 0.97]), 'Phantom', fontsize=11, ha='center',
        va='top', color='k')
ax2.text(*transform_axis_to_data_coordinates(ax2, [0.5, 0.97]), '1x128 array', fontsize=11, ha='center',
        va='top', color='k')
ax3.text(*transform_axis_to_data_coordinates(ax3, [0.5, 0.97]), '1x128 array', fontsize=11, ha='center',
        va='top', color='k')
ax4.text(*transform_axis_to_data_coordinates(ax4, [0.5, 0.97]), '2x64 array \n 0.25$\\,$mm offset', fontsize=11, ha='center',
        va='top', color='k')
'''

# plt.show()
format_save(save_path=results_path, save_name=f"Graph4_GafchromicCompProfile{'LogoComp'}", dpi=dpi, plot_size=plot_size, save_format=format, fig=fig)