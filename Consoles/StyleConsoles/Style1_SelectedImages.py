from Consoles.StyleConsoles.Utils_ImageLoad import *

# Save path and options for the map
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Style/ImageSandbox/')
background_subtraction = True
normalization = True
x_limits = [2, 34]
y_limits = [85, 115]
# y_limits = [95, 125]
intensity_limits = None
pixel = 'fill'

plot_size = fullsize_plot
dpi = 300
format = '.svg'

cmap=matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black", "red", "yellow"])

# Selection of the image (automatic assigning of the Analyzer)
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_230924/')
image = 'Logo'
# image = 'Array3_Logo'
position = '85.0'

A = load_image(folder_path, image, background_subtraction=background_subtraction, normalization=normalization,
               position=position)

fig, ax = plt.subplots()

A.plot_map(None, pixel=pixel, intensity_limits=intensity_limits, ax_in=ax, fig_in=fig, cmap=cmap)

if x_limits is None:
    x_limits = ax.get_xlim()
if y_limits is None:
    y_limits = ax.get_ylim()

ax.set_xlim(x_limits)
ax.set_ylim(y_limits)
# ax.set_aspect('equal')

# '''
# Scale the axis true to scale
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
add_diode_geometry_indicator(ax, A, position='upper right', fig=fig)

format_save(save_path=results_path, save_name=f"{image}", dpi=dpi, plot_size=plot_size, save_format=format, fig=fig)