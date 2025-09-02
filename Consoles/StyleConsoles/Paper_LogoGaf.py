import matplotlib.pyplot as plt

from Consoles.StyleConsoles.Utils_ImageLoad import *
from PIL import Image
from Consoles.Consoles8Gafchromic.Concept8GafCompTests import align_and_compare_images, resample_image, transform_image

# Save path and options for the map
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Style/Paper/')
background_subtraction = True
normalization = True

plot_size = (18 * cm, 3/2  * 18 / 1.2419 * cm)
# Setup of the final figure
# Structure: Ax1 Logo Line Super Res - Ax2 Logo Gaf - Ax3 BeamShape 2-Line - Ax4 BeamShape Gaf -
# Ax5 Logo Overlay - Ax6 Beam Overlay
fig, [[ax1, ax2], [ax3, ax4], [ax5, ax6]] = plt.subplots(3, 2, figsize=plot_size)
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
format = '.svg'
cmap=matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black", "red", "yellow"])
# ------------------------------------------------------------------------------------------------------------------
# Ax1: Logo image
# ------------------------------------------------------------------------------------------------------------------
ax = ax1
x_limit = np.array([20-18, 20+14])
y_limit = np.array([66.5-15, 66.5+17])
# Selection of the image (automatic assigning of the Analyzer)
'''
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
image = '_GafCompLogo_'
position = None
A1 = load_image(folder_path, image, background_subtraction=background_subtraction, normalization=normalization,
               position=position)
'''
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_230924/')
image = 'Array3_Logo'
position = '85.0'

A1 = load_image(folder_path, image, background_subtraction=background_subtraction, normalization=normalization,
               position=position)
A1.maps[0]['y'] = A1.maps[0]['y'] - 110 + 66.5
mid = np.argsort(np.abs(A1.maps[0]['y'] - 66.5))[0]
dist = len(A1.maps[0]['y']) - mid
A1.maps[0]['y'] = A1.maps[0]['y'][-2*dist:]
A1.maps[0]['z'] = A1.maps[0]['z'][-2*dist:]
# '''
if '221024' in str(folder_path):
    version = 'v1'
else:
    version = 'v2'
    
A = A1

for i, image_map in enumerate(A.maps):
    A.maps[i]['z'] = simple_zero_replace(image_map['z'])

A.maps[0] = overlap_treatment(A.maps[0], A, True)

# Correction of signal current | I = 2.02 nA
A.maps[0]['z'] = A.maps[0]['z'] * (2/2.02) / 0.82
A1 = deepcopy(A)

if zero_scale:
    A.maps[0]['x'] = A.maps[0]['x'] - np.min(x_limit)
    A.maps[0]['y'] = A.maps[0]['y'] - np.min(y_limit)
    x_limit = x_limit - np.min(x_limit)
    y_limit = y_limit - np.min(y_limit)
A.plot_map(None, pixel=pixel, intensity_limits=intensity_limits, ax_in=ax, fig_in=fig, cmap=cmap, imshow=True)
x_limits.append(x_limit)
y_limits.append(y_limit)

# ------------------------------------------------------------------------------------------------------------------
# Ax2: Beam 2x64x0,5x0,5 image
# ------------------------------------------------------------------------------------------------------------------
ax = ax2
x_limit = np.array([22-14, 22+14])
y_limit = np.array([66.5-11, 66.5+17])
# Selection of the image (automatic assigning of the Analyzer)
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
image = '2Line_Beam_'
position = None
A2 = load_image(folder_path, image, background_subtraction=background_subtraction, normalization=normalization,
               position=position)
A = A2
for i, image_map in enumerate(A.maps):
    A.maps[i]['z'] = simple_zero_replace(image_map['z'])

A.maps[0] = overlap_treatment(A.maps[0], A, True)

# Correction of signal current | I = 2.02 nA
A.maps[0]['z'] = A.maps[0]['z'] * (2/2.015) / 0.9
A2 = deepcopy(A)

if zero_scale:
    A.maps[0]['x'] = A.maps[0]['x'] - np.min(x_limit)
    A.maps[0]['y'] = A.maps[0]['y'] - np.min(y_limit)
    x_limit = x_limit - np.min(x_limit)
    y_limit = y_limit - np.min(y_limit)
A.plot_map(None, pixel=pixel, intensity_limits=intensity_limits, ax_in=ax, fig_in=fig, cmap=cmap, imshow=True)
x_limits.append(x_limit)
y_limits.append(y_limit)

# ------------------------------------------------------------------------------------------------------------------
# Ax3: Logo Gafchromic
# ------------------------------------------------------------------------------------------------------------------
ax = ax3
# Axis limits
x_limit = np.array([22-18, 22+14])
y_limit = np.array([28-15, 28+17])
# Selection of the image (automatic assigning of the Analyzer)
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/Gafchromic_221024/')
image = 'gafchromic_matrix211024_006.bmp'
position = None
gaf = load_image(folder_path, image, background_subtraction=background_subtraction, normalization=normalization,
               position=position)
DownSampGaf = GafImage(folder_path / image)
quick_load = ((folder_path / image).parent / 'QuickLoads') / (image[:-4] + '.npy')
low_pixel_size = 0.25

if os.path.isfile(quick_load):
    print('Quick Load')
    DownSampGaf.load_image(quick=True)
    down_samp = DownSampGaf.image
else:
    down_samp = resample_image(gaf.image, gaf.pixel_size, low_pixel_size)
    DownSampGaf.image = down_samp
    DownSampGaf.save_image(quick_load.parent)

_x, _y, _z = homogenize_pixel_size([A1.maps[0]['x'], A1.maps[0]['y'], np.abs(A1.maps[0]['z']) / np.max(A1.maps[0]['z'])])

diff3, score3, addition3 = align_and_compare_images(_z[::-1], gaf.image, low_pixel_size, gaf.pixel_size,
                                                    optimize_alignment=True, bounds=(-3, 3), ev_max_iter=1000,
                                                    ev_pop_size=10, optimization_method='evolutionary',
                                                    image_down_sampled=down_samp)
print(np.shape(diff3), np.min(diff3), np.max(diff3))
print(score3)
gaf.image = transform_image(down_samp, rotation=np.array(-2.5), center_shift=[0, 0])
gaf.pixel_size = low_pixel_size

if zero_scale:
    ext_x = 0 - np.min(x_limit)
    ext_y = 0 - np.min(y_limit)

    x_limit = x_limit - np.min(x_limit)
    y_limit = y_limit - np.min(y_limit)
else:
    ext_x = 0
    ext_y = 0

color_map = ax.imshow(gaf.image, cmap=cmap, vmin=0, vmax=1, extent=(
    ext_x, np.shape(gaf.image)[1] * gaf.pixel_size + ext_x, ext_y, np.shape(gaf.image)[0] * gaf.pixel_size + ext_y))
norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map.cmap)
sm.set_array([])
bar = fig.colorbar(sm, ax=ax, extend='max')
ax.set_xlabel('Position x (mm)')
ax.set_ylabel('Position y (mm)')
bar.set_label('Normalized Response')

x_limits.append(x_limit)
y_limits.append(y_limit)

# ------------------------------------------------------------------------------------------------------------------
# Ax4: Beam 2x64x0,5x0,5 image
# ------------------------------------------------------------------------------------------------------------------
ax = ax4
# Axis limits
x_limit = np.array([22-14, 22+14])
y_limit = np.array([28-14, 28+14])
# Selection of the image (automatic assigning of the Analyzer)
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/Gafchromic_221024/')
image = 'gafchromic_matrix211024_010.bmp'
position = None
gaf = load_image(folder_path, image, background_subtraction=background_subtraction, normalization=normalization,
               position=position)
DownSampGaf = GafImage(folder_path / image)
quick_load = ((folder_path / image).parent / 'QuickLoads') / (image[:-4] + '.npy')
low_pixel_size = 0.25
if os.path.isfile(quick_load):
    print('Quick Load')
    DownSampGaf.load_image(quick=True)
    down_samp = DownSampGaf.image
else:
    down_samp = resample_image(gaf.image, gaf.pixel_size, low_pixel_size)
    DownSampGaf.image = down_samp
    DownSampGaf.save_image(quick_load.parent)

_x, _y, _z = homogenize_pixel_size([A2.maps[0]['x'], A2.maps[0]['y'], np.abs(A2.maps[0]['z']) / np.max(A2.maps[0]['z'])])
diff4, score4, addition4 = align_and_compare_images(_z[::-1], gaf.image, low_pixel_size, gaf.pixel_size,
                                                    optimize_alignment=True, bounds=(-3, 3), ev_max_iter=1000,
                                                    ev_pop_size=10, optimization_method='evolutionary',
                                                    image_down_sampled=down_samp)

gaf.image = transform_image(down_samp, rotation=np.array(-2.5), center_shift=[0, 0])
gaf.pixel_size = low_pixel_size

if zero_scale:
    ext_x = 0 - np.min(x_limit)
    ext_y = 0 - np.min(y_limit)

    x_limit = x_limit - np.min(x_limit)
    y_limit = y_limit - np.min(y_limit)
else:
    ext_x = 0
    ext_y = 0

color_map = ax.imshow(gaf.image, cmap=cmap, vmin=0.0, vmax=1, extent=(
    ext_x, np.shape(gaf.image)[1] * gaf.pixel_size + ext_x, ext_y, np.shape(gaf.image)[0] * gaf.pixel_size + ext_y))
norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1)
sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map.cmap)
sm.set_array([])
bar = fig.colorbar(sm, ax=ax, extend='max')
ax.set_xlabel('Position x (mm)')
ax.set_ylabel('Position y (mm)')
bar.set_label('Normalized Response')

x_limits.append(x_limit)
y_limits.append(y_limit)

# ------------------------------------------------------------------------------------------------------------------
# Ax5: Overlay Logo
# ------------------------------------------------------------------------------------------------------------------
cmap = sns.color_palette('coolwarm', as_cmap=True)

ax = ax5
# Axis limits
x_limit = np.array([24-18, 24+14])
y_limit = np.array([28-15, 28+17])
if zero_scale:
    ext_x = 0 - np.min(x_limit)
    ext_y = 0 - np.min(y_limit)

    x_limit = x_limit - np.min(x_limit)
    y_limit = y_limit - np.min(y_limit)
else:
    ext_x = 0
    ext_y = 0

diff3 = transform_image(-diff3, rotation=np.array(-2.5), center_shift=[0, 0])

color_map = ax.imshow(diff3, cmap=cmap, vmin=-1, vmax=1, extent=(
    ext_x, np.shape(diff3)[1] * low_pixel_size + ext_x, ext_y, np.shape(diff3)[0] * low_pixel_size + ext_y))
norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map.cmap)
sm.set_array([])
bar = fig.colorbar(sm, ax=ax, extend='max')
ax.set_xlabel('Position x (mm)')
ax.set_ylabel('Position y (mm)')
bar.set_label('Normed Response Difference')

x_limits.append(x_limit)
y_limits.append(y_limit)
# ------------------------------------------------------------------------------------------------------------------
# Ax6: Overlay Beamshape
# ------------------------------------------------------------------------------------------------------------------
ax = ax6
# Axis limits
x_limit = np.array([24-14, 24+14])
y_limit = np.array([28-14, 28+14])
if zero_scale:
    ext_x = 0 - np.min(x_limit)
    ext_y = 0 - np.min(y_limit)

    x_limit = x_limit - np.min(x_limit)
    y_limit = y_limit - np.min(y_limit)
else:
    ext_x = 0
    ext_y = 0

diff4 = transform_image(-diff4, rotation=np.array(-2.5), center_shift=[0, 0])

color_map = ax.imshow(diff4, cmap=cmap, vmin=-1, vmax=1, extent=(
    ext_x, np.shape(diff4)[1] * low_pixel_size + ext_x, ext_y, np.shape(diff4)[0] * low_pixel_size + ext_y))
norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map.cmap)
sm.set_array([])
bar = fig.colorbar(sm, ax=ax, extend='max')
ax.set_xlabel('Position x (mm)')
ax.set_ylabel('Position y (mm)')
bar.set_label('Normed Response Difference')

x_limits.append(x_limit)
y_limits.append(y_limit)

# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6]):
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
ax3.text(*transform_axis_to_data_coordinates(ax3, [0.03, 0.97]), r'\textbf{Gafchromic}', fontsize=7, ha='left',
        va='top', color='k')
ax4.text(*transform_axis_to_data_coordinates(ax4, [0.03, 0.97]), r'\textbf{Gafchromic}', fontsize=7, ha='left',
        va='top', color='k')
ax5.text(*transform_axis_to_data_coordinates(ax5, [0.03, 0.97]), r'\textbf{Difference (a)-(c)}', fontsize=7, ha='left',
        va='top', color='k')
ax6.text(*transform_axis_to_data_coordinates(ax6, [0.03, 0.97]), r'\textbf{Difference (b)-(d)}', fontsize=7, ha='left',
        va='top', color='k')

ax1.text(*transform_axis_to_data_coordinates(ax1, [0.97, 0.97]), r'\textbf{(a)}', fontsize=8, ha='right',
        va='top', color='k')
ax2.text(*transform_axis_to_data_coordinates(ax2, [0.97, 0.97]), r'\textbf{(b)}', fontsize=8, ha='right',
        va='top', color='k')
ax3.text(*transform_axis_to_data_coordinates(ax3, [0.97, 0.97]), r'\textbf{(c)}', fontsize=8, ha='right',
        va='top', color='k')
ax4.text(*transform_axis_to_data_coordinates(ax4, [0.97, 0.97]), r'\textbf{(d)}', fontsize=8, ha='right',
        va='top', color='k')
ax5.text(*transform_axis_to_data_coordinates(ax5, [0.97, 0.97]), r'\textbf{(e)}', fontsize=8, ha='right',
        va='top', color='k')
ax6.text(*transform_axis_to_data_coordinates(ax6, [0.97, 0.97]), r'\textbf{(f)}', fontsize=8, ha='right',
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

just_save(save_path=results_path, save_name=f"GafchromicComp{version}", dpi=dpi, plot_size=plot_size, save_format=format, fig=fig)