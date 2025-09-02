import matplotlib.pyplot as plt

from Consoles.StyleConsoles.Utils_ImageLoad import *
from PIL import Image

# Save path and options for the map
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Style/Paper/')
background_subtraction = True
normalization = True

plot_size = (18 * cm, 18 / 1.2419 * cm)
# Setup of the final figure
fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=plot_size)
# Adjust spacing: left, right, bottom, top, wspace, hspace
fig.subplots_adjust(wspace=0.25, hspace=0.25)
# Axis limits
y_limits = []
x_limits = []
zero_scale = True
intensity_limits = None
pixel = 'fill'

dpi = 300
format = '.pdf'
cmap=matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black", "red", "yellow"])
# ------------------------------------------------------------------------------------------------------------------
# Ax1: Logo image or schematic?
# ------------------------------------------------------------------------------------------------------------------
x_limits.append(None)
y_limits.append(None)

with Image.open('/Users/nico_brosda/Cyrce_Messungen/3D_Files/Logo.jpg') as img:
    # img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # img = img.transpose(Image.FLIP_TOP_BOTTOM)
    # img = img.rotate(180)
    ax1.imshow(img)
ax1.axis('off')
ax1.set_xlim(ax1.get_xlim()), ax1.set_ylim(ax1.get_ylim())
ax1.annotate('', xy=(80.6, 280), xytext=(303.1, 280),
             arrowprops=dict(arrowstyle='<->', color='red'))
ax1.axvline(80.6, ymin=0.3, ymax=0.7, ls='--', c='r', alpha=0.6)
ax1.axvline(303.1, ymin=0.3, ymax=0.7, ls='--', c='r', alpha=0.6)

ax1.text((ax1.get_xlim()[1]-ax1.get_xlim()[0]) / 2, 284, 'Text within x mm', ha='center', va='top',
         fontsize=7, color='red')
# ------------------------------------------------------------------------------------------------------------------
# Ax2: Logo 128x0.5x0.5 image
# ------------------------------------------------------------------------------------------------------------------
x_limits2 = np.array([2, 34])
y_limits2 = np.array([100-13, 100+19])
# Selection of the image (automatic assigning of the Analyzer)
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_230924/')
image = 'Logo'
position = '85.0'
A2 = load_image(folder_path, image, background_subtraction=background_subtraction, normalization=normalization,
               position=position)
A2.maps[0] = overlap_treatment(A2.maps[0], A2, super_res=True)
# Correction to make measurements comparable | I = 2.385
A2.maps[0]['z'] = A2.maps[0]['z']*0.95* (2/2.385) / 0.82
if zero_scale:
    A2.maps[0]['x'] = A2.maps[0]['x'] - np.min(x_limits2)
    A2.maps[0]['y'] = A2.maps[0]['y'] - np.min(y_limits2)
    x_limits2 = x_limits2 - np.min(x_limits2)
    y_limits2 = y_limits2 - np.min(y_limits2)
A2.plot_map(None, pixel=pixel, intensity_limits=intensity_limits, ax_in=ax2, fig_in=fig, cmap=cmap, imshow=True)
x_limits.append(x_limits2)
y_limits.append(y_limits2)
# ------------------------------------------------------------------------------------------------------------------
# Ax3: Logo 128x0.25x0.5 image
# ------------------------------------------------------------------------------------------------------------------
x_limits3 = np.array([2, 34])
y_limits3 = np.array([110-13, 110+19])
# Selection of the image (automatic assigning of the Analyzer)
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_230924/')
image = 'Array3_Logo'
position = '85.0'
A3 = load_image(folder_path, image, background_subtraction=background_subtraction, normalization=normalization,
               position=position)
print(A3.diode_size)
A3.maps[0] = overlap_treatment(A3.maps[0], A3, True)

# Correction to make measurements comparable | I = 2.435 nA
A3.maps[0]['z'] = A3.maps[0]['z']*0.95* (2/2.435) / 0.82

if zero_scale:
    A3.maps[0]['x'] = A3.maps[0]['x'] - np.min(x_limits3)
    A3.maps[0]['y'] = A3.maps[0]['y'] - np.min(y_limits3)
    x_limits3 = x_limits3 - np.min(x_limits3)
    y_limits3 = y_limits3 - np.min(y_limits3)
A3.plot_map(None, pixel=pixel, intensity_limits=intensity_limits, ax_in=ax3, fig_in=fig, cmap=cmap, imshow=True)
x_limits.append(x_limits3)
y_limits.append(y_limits3)

# ------------------------------------------------------------------------------------------------------------------
# Ax4: Logo 2x64x0.5x0.5 image
# ------------------------------------------------------------------------------------------------------------------
x_limits4 = np.array([20-16, 20+16])
y_limits4 = [66.5-13, 66.5+19]
# Selection of the image (automatic assigning of the Analyzer)
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
image = '_GafCompLogo_'
position = None
A4 = load_image(folder_path, image, background_subtraction=background_subtraction, normalization=normalization,
               position=position)
for i, image_map in enumerate(A4.maps):
    A4.maps[i]['z'] = simple_zero_replace(image_map['z'])

A4.maps[0] = overlap_treatment(A4.maps[0], A4, True)

# Correction of signal current | I = 2.02 nA
A4.maps[0]['z'] = A4.maps[0]['z'] * (2/2.02) / 0.82

if zero_scale:
    A4.maps[0]['x'] = A4.maps[0]['x'] - np.min(x_limits4)
    A4.maps[0]['y'] = A4.maps[0]['y'] - np.min(y_limits4)
    x_limits4 = x_limits4 - np.min(x_limits4)
    y_limits4 = y_limits4 - np.min(y_limits4)
A4.plot_map(None, pixel=pixel, intensity_limits=intensity_limits, ax_in=ax4, fig_in=fig, cmap=cmap, imshow=True)
x_limits.append(x_limits4)
y_limits.append(y_limits4)
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

for i, ax in enumerate([ax1, ax2, ax3, ax4]):
    if i == 0:
        continue
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

add_png_icon(ax2, A2, 'top left', translation='x', zoom=0.15)
add_png_icon(ax3, A3, 'top left', translation='x', zoom=0.15)
add_png_icon(ax4, A4, 'top left', translation='x', zoom=0.15)

ax1.text(*transform_axis_to_data_coordinates(ax1, [0.97, 0.97]), r'\textbf{(a)}', fontsize=8, ha='right',
        va='top', color='k')
ax2.text(*transform_axis_to_data_coordinates(ax2, [0.97, 0.97]), r'\textbf{(b)}', fontsize=8, ha='right',
        va='top', color='k')
ax3.text(*transform_axis_to_data_coordinates(ax3, [0.97, 0.97]), r'\textbf{(c)}', fontsize=8, ha='right',
        va='top', color='k')
ax4.text(*transform_axis_to_data_coordinates(ax4, [0.97, 0.97]), r'\textbf{(d)}', fontsize=8, ha='right',
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
just_save(save_path=results_path, save_name=f"LogoComp", dpi=dpi, plot_size=plot_size, save_format=format, fig=fig)