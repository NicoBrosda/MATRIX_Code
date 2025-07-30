from Consoles.StyleConsoles.Utils_ImageLoad import *
from Consoles.Consoles8Gafchromic.Concept8GafCompTests import align_and_compare_images, resample_image, transform_image

# Save path and options for the map
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Style/ImageSandbox/Matilde')
background_subtraction = True
normalization = True

# Axis limits
x_limits = np.array([22-16, 22+16])
y_limits = np.array([12, 44])

# x_limits = np.array([0, 50])
# y_limits = np.array([0, 50])

zero_scale = True

intensity_limits = [0, 1]
pixel = 'fill'

plot_size = fullsize_plot
plot_size = (6 * cm, 6 / 1.2419 * cm)

dpi = 300
format = '.svg'

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black", "red", "yellow"])

# Selection of the image (automatic assigning of the Analyzer)
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/Gafchromic_221024/')
image = 'gafchromic_matrix211024_006.bmp'
position = None


gaf = load_image(folder_path, image, background_subtraction=background_subtraction, normalization=normalization,
               position=position)

A = load_image(Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/'), '_GafCompLogo_',
               background_subtraction=background_subtraction, normalization=normalization, position=position)

for i, image_map in enumerate(A.maps):
        A.maps[i]['z'] = simple_zero_replace(image_map['z'])
A.maps[0]['x'], A.maps[0]['y'], A.maps[0]['z'] = homogenize_pixel_size(
        [A.maps[0]['x'], A.maps[0]['y'], np.abs(A.maps[0]['z']) / np.max(A.maps[0]['z'])])

low_pixel_size = A.maps[0]['x'][1] - A.maps[0]['x'][0]

DownSampGaf = GafImage(folder_path / image)
quick_load = ((folder_path / image).parent / 'QuickLoads') / (image[:-4] + '.npy')
print(quick_load)
if os.path.isfile(quick_load):
    DownSampGaf.load_image(quick=True)
    down_samp = DownSampGaf.image
else:
    down_samp = resample_image(gaf.image, gaf.pixel_size, low_pixel_size)
    DownSampGaf.image = down_samp
    DownSampGaf.save_image(quick_load.parent)

diff, score, addition = align_and_compare_images(A.maps[0]['z'], gaf.image, low_pixel_size, gaf.pixel_size,
                                               optimize_alignment=True, bounds=(-3, 3), ev_max_iter=500, ev_pop_size=10,
                                               optimization_method='evolutionary', image_down_sampled=down_samp)

print(addition[0], addition[1])

gaf.image = transform_image(down_samp, rotation=np.array(-2.5), center_shift=[0, 0])
gaf.pixel_size = low_pixel_size

fig, ax = plt.subplots()

if zero_scale:
    ext_x = 0 - np.min(x_limits)
    ext_y = 0 - np.min(y_limits)

    x_limits = x_limits - np.min(x_limits)
    y_limits = y_limits - np.min(y_limits)
else:
    ext_x = 0
    ext_y = 0

color_map2 = ax.imshow(gaf.image, cmap=cmap, vmin=0, vmax=1, extent=(
    ext_x, np.shape(gaf.image)[1] * gaf.pixel_size + ext_x, ext_y, np.shape(gaf.image)[0] * gaf.pixel_size + ext_y))
norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map2.cmap)
sm.set_array([])
bar = fig.colorbar(sm, ax=ax, extend='max')
ax.set_xlabel('Position x (mm)')
ax.set_ylabel('Position y (mm)')
bar.set_label('Normalized Response')

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