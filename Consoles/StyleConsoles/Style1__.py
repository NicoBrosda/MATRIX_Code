from Consoles.StyleConsoles.Utils_ImageLoad import *

# Save path and options for the map
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Style/ImageSandbox/Matilde')
background_subtraction = True
normalization = True

# Axis limits
x_limits = np.array([2, 34])
y_limits = [69, 101]
# y_limits = np.array([94, 126])
zero_scale = True

intensity_limits = None
pixel = 'fill'

plot_size = fullsize_plot
plot_size = (6 * cm, 6 / 1.2419 * cm)

dpi = 300
format = '.svg'

cmap=matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black", "red", "yellow"])

# Selection of the image (automatic assigning of the Analyzer)
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_19062024/')
image = '10s_iphcmatrixcrhea_'
# image = 'Array3_Logo'
position = None

A = load_image(folder_path, image, background_subtraction=background_subtraction, normalization=normalization,
               position=position)

fig, ax = plt.subplots()

if zero_scale:
    A.maps[0]['x'] = A.maps[0]['x'] - np.min(x_limits)
    A.maps[0]['y'] = A.maps[0]['y'] - np.min(y_limits)

    x_limits = x_limits - np.min(x_limits)
    y_limits = y_limits - np.min(y_limits)

print(np.shape(A.maps[0]['z']))
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