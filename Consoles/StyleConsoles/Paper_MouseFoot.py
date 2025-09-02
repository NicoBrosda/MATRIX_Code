from Consoles.StyleConsoles.Utils_ImageLoad import *

# Save path and options for the map
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Style/Paper/')
background_subtraction = True
normalization = True

# Axis limits
x_limits = np.array([22.5-11, 22.5+11])
y_limits = [66.5-11, 66.5+11]
# y_limits = np.array([94, 126])
zero_scale = True

intensity_limits = np.array([75, 105]) * (2/1.995) / 0.82
pixel = 'fill'

plot_size = fullsize_plot
plot_size = (8.9 * cm, 8.9 / 1.2419 * cm)

dpi = 300
format = '.pdf'

cmap = sns.color_palette("Greys_r", as_cmap=True)

# Selection of the image (automatic assigning of the Analyzer)
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_211124/')
image = '17_2DSmall_foot_xyscan_'
# image = 'Array3_Logo'
position = None


A = load_image(folder_path, image, background_subtraction=background_subtraction, normalization=normalization,
               position=position)
for i, image_map in enumerate(A.maps):
    A.maps[i]['z'] = simple_zero_replace(image_map['z'])

A.maps[0]['z'] = A.maps[0]['z'] * (2/1.995) / 0.82

fig, ax = plt.subplots(figsize=plot_size)

if zero_scale:
    A.maps[0]['x'] = A.maps[0]['x'] - np.min(x_limits)
    A.maps[0]['y'] = A.maps[0]['y'] - np.min(y_limits)

    x_limits = x_limits - np.min(x_limits)
    y_limits = y_limits - np.min(y_limits)

A.plot_map(None, pixel=pixel, intensity_limits=intensity_limits, ax_in=ax, fig_in=fig, cmap=cmap, imshow=True)

if x_limits is None:
    x_limits = ax.get_xlim()
if y_limits is None:
    y_limits = ax.get_ylim()

ax.set_xlim(x_limits)
ax.set_ylim(y_limits)
# ax.set_aspect('equal')

# '''
# Scale the axis true to scale
print(ax.get_xlim())
print(ax.get_ylim())
x_scale = ax.get_xlim()
y_scale = ax.get_ylim()
if x_scale[1] - x_scale[0] < y_scale[1] - y_scale[0]:
    ax.set_xlim(x_scale[0] - (y_scale[1] - y_scale[0] - x_scale[1] + x_scale[0]) / 2,
                x_scale[1] + (y_scale[1] - y_scale[0] - x_scale[1] + x_scale[0]) / 2)
elif x_scale[1] - x_scale[0] > y_scale[1] - y_scale[0]:
    ax.set_ylim(y_scale[0] - (x_scale[1] - x_scale[0] - y_scale[1] + y_scale[0]) / 2,
                y_scale[1] + (x_scale[1] - x_scale[0] - y_scale[1] + y_scale[0]) / 2)
# '''

# Insert after ax.set_aspect('equal'):
# add_diode_geometry_indicator(ax, A, position='upper right', fig=fig)
print(ax.get_xlim())
print(ax.get_ylim())

add_png_icon(ax, A, 'top left', zoom=0.1, translation=['x', 'y'], background=True)
x0, y0, x1, y1 = add_image(ax, Path('/Users/nico_brosda/Cyrce_Messungen/3D_Files/Foot_Setup.jpg'), location=(0.02, 0.02), zoom=0.04, align_corner=(0, 0))
ax.text(*transform_axis_to_data_coordinates(ax, (x0, y0)), 'Measurement Setup', fontsize=5, ha='left', va='bottom', color='red', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.1'))
x0, y0, x1, y1 = add_image(ax, Path('/Users/nico_brosda/Cyrce_Messungen/3D_Files/Mouse_Phantom.jpg'), location=(0.02, y1 + 0.02), zoom=0.05, align_corner=(0, 0))
ax.text(*transform_axis_to_data_coordinates(ax, (x0, y0)), 'Phantom', fontsize=5, ha='left', va='bottom', color='red', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.1'))

for ax in fig.axes:
    if is_colorbar(ax):
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    else:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

just_save(save_path=results_path, save_name=f"{image}", dpi=dpi, plot_size=plot_size, save_format=format, fig=fig)

# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=plot_size)

A.maps[0] = overlap_treatment(A.maps[0], A, True)

if zero_scale:
    A.maps[0]['x'] = A.maps[0]['x'] - np.min(x_limits)
    A.maps[0]['y'] = A.maps[0]['y'] - np.min(y_limits)

    x_limits = x_limits - np.min(x_limits)
    y_limits = y_limits - np.min(y_limits)

A.plot_map(None, pixel=pixel, intensity_limits=intensity_limits, ax_in=ax, fig_in=fig, cmap=cmap, imshow=True)

if x_limits is None:
    x_limits = ax.get_xlim()
if y_limits is None:
    y_limits = ax.get_ylim()

ax.set_xlim(x_limits)
ax.set_ylim(y_limits)
# ax.set_aspect('equal')

# '''
# Scale the axis true to scale
print(ax.get_xlim())
print(ax.get_ylim())
x_scale = ax.get_xlim()
y_scale = ax.get_ylim()
if x_scale[1] - x_scale[0] < y_scale[1] - y_scale[0]:
    ax.set_xlim(x_scale[0] - (y_scale[1] - y_scale[0] - x_scale[1] + x_scale[0]) / 2,
                x_scale[1] + (y_scale[1] - y_scale[0] - x_scale[1] + x_scale[0]) / 2)
elif x_scale[1] - x_scale[0] > y_scale[1] - y_scale[0]:
    ax.set_ylim(y_scale[0] - (x_scale[1] - x_scale[0] - y_scale[1] + y_scale[0]) / 2,
                y_scale[1] + (x_scale[1] - x_scale[0] - y_scale[1] + y_scale[0]) / 2)
# '''

# Insert after ax.set_aspect('equal'):
# add_diode_geometry_indicator(ax, A, position='upper right', fig=fig)
print(ax.get_xlim())
print(ax.get_ylim())

add_png_icon(ax, A, 'top left', zoom=0.1, translation=['x', 'y'], background=True)
x0, y0, x1, y1 = add_image(ax, Path('/Users/nico_brosda/Cyrce_Messungen/3D_Files/Foot_Setup.jpg'), location=(0.02, 0.02), zoom=0.04, align_corner=(0, 0))
add_image(ax, Path('/Users/nico_brosda/Cyrce_Messungen/3D_Files/Mouse_Phantom.jpg'), location=(0.02, y1 + 0.02), zoom=0.05, align_corner=(0, 0))

for ax in fig.axes:
    if is_colorbar(ax):
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    else:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

just_save(save_path=results_path, save_name=f"{image}superRes", dpi=dpi, plot_size=plot_size, save_format=format, fig=fig)