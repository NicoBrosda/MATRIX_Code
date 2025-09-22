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
plot_size = (2 * 8.9 * cm, 8.0 / 1.2419 * cm)

dpi = 300
format = '.svg'

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
# SuperRes ????????????????
A.maps[0]['z'] = A.maps[0]['z'] * (2/1.995) / 0.82

fig, [ax1, ax2]= plt.subplots(1, 2, figsize=plot_size)
fig.subplots_adjust(wspace=0.0, hspace=0.0)

# ------------------------------------------------------------------------------------------------------------------
# Ax1: Image of phantom
# ------------------------------------------------------------------------------------------------------------------
with Image.open('/Users/nico_brosda/Cyrce_Messungen/3D_Files/Mouse_Phantom.jpg') as img:
    # img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # img = img.transpose(Image.FLIP_TOP_BOTTOM)
    # img = img.rotate(180)
    ax1.imshow(img)
ax1.axis('off')
ax1.set_xlim(ax1.get_xlim()), ax1.set_ylim(ax1.get_ylim())

'''
x0, y0, x1, y1 = add_image(ax1, Path('/Users/nico_brosda/Cyrce_Messungen/3D_Files/Foot_Setup.jpg'), location=(0.02, 0.02), zoom=0.04, align_corner=(0, 0))
ax1.text(*transform_axis_to_data_coordinates(ax1, (x0, y0)), 'Measurement Setup', fontsize=5, ha='left', va='bottom', color='red', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.1'))
'''

# ------------------------------------------------------------------------------------------------------------------
# Ax2: Measurement mouse Foot
# ------------------------------------------------------------------------------------------------------------------
if zero_scale:
    A.maps[0]['x'] = A.maps[0]['x'] - np.min(x_limits)
    A.maps[0]['y'] = A.maps[0]['y'] - np.min(y_limits)

    x_limits = x_limits - np.min(x_limits)
    y_limits = y_limits - np.min(y_limits)

A.plot_map(None, pixel=pixel, intensity_limits=intensity_limits, ax_in=ax2, fig_in=fig, cmap=cmap, imshow=True)

if x_limits is None:
    x_limits = ax2.get_xlim()
if y_limits is None:
    y_limits = ax2.get_ylim()

ax2.set_xlim(x_limits)
ax2.set_ylim(y_limits)
# ax.set_aspect('equal')

# '''
# Scale the axis true to scale
print(ax2.get_xlim())
print(ax2.get_ylim())
x_scale = ax2.get_xlim()
y_scale = ax2.get_ylim()
if x_scale[1] - x_scale[0] < y_scale[1] - y_scale[0]:
    ax2.set_xlim(x_scale[0] - (y_scale[1] - y_scale[0] - x_scale[1] + x_scale[0]) / 2,
                x_scale[1] + (y_scale[1] - y_scale[0] - x_scale[1] + x_scale[0]) / 2)
elif x_scale[1] - x_scale[0] > y_scale[1] - y_scale[0]:
    ax2.set_ylim(y_scale[0] - (x_scale[1] - x_scale[0] - y_scale[1] + y_scale[0]) / 2,
                y_scale[1] + (x_scale[1] - x_scale[0] - y_scale[1] + y_scale[0]) / 2)
# '''

# Insert after ax.set_aspect('equal'):
# add_diode_geometry_indicator(ax, A, position='upper right', fig=fig)
print(ax2.get_xlim())
print(ax2.get_ylim())

add_png_icon(ax2, A, 'top left', zoom=0.1, translation=['x', 'y'], background=True)

ax1.text(*transform_axis_to_data_coordinates(ax1, [0.97, 0.97]), r'\textbf{(a)}', fontsize=10, ha='right',
        va='top', color='k', bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0.1'))
ax2.text(*transform_axis_to_data_coordinates(ax2, [0.97, 0.97]), r'\textbf{(b)}', fontsize=10, ha='right',
        va='top', color='k', bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0.1'))

format_save(save_path=results_path, save_name=f"Graph5_MouseFoot", dpi=dpi, plot_size=plot_size, save_format=format, fig=fig)
