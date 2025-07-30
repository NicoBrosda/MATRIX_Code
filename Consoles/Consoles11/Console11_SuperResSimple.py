import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from Consoles.StyleConsoles.Utils_ImageLoad import *

# Save path and options for the map
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Style/ImageSandbox/Matilde')
background_subtraction = True
normalization = True

print(super_resolution_matrix(4))

print(super_resolution_matrix2(4))
# Axis limits
x_limits = np.array([20-16, 20+16])
y_limits = [66.5-16, 66.5+16]
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
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
image = '_GafComp200_'
# image = 'Array3_Logo'
position = None


A = load_image(folder_path, image, background_subtraction=background_subtraction, normalization=normalization,
               position=position)
for i, image_map in enumerate(A.maps):
    A.maps[i]['z'] = simple_zero_replace(image_map['z'])

# Parameters
N = np.shape(A.maps[i]['z'])[0] // 2 - 1
box_size = 1
row = 16

# Generate Gaussian data centered in array of length N
x = np.arange(N)
center = N / 2
sigma = N / 6
array1 = A.maps[i]['z'].T[row, 0::2]
array2 = A.maps[i]['z'].T[row, 1::2]
input_array = A.maps[i]['z'].T[row]

N = len(input_array) // 2
super_res = np.zeros(2 * N - 1)

for n in range(N):
    if n == 0:
        super_res[2*n] = array1[n] + array2[n]
    else:
        super_res[2*n] = array1[n] + array2[n]
        super_res[2*n-1] = array1[n] + array2[n-1]

super_res = super_res / 2
# Middle row: 2N+2 elements, simulate same Gaussian sampled at higher resolution

middle_array = A.maps[i]['z'].T[row]

# Normalize data to [0,1] for coloring
norm = mcolors.Normalize(vmin=np.min(A.maps[i]['z']), vmax=np.max(A.maps[i]['z']))

fig, ax = plt.subplots(figsize=(2*N/4, 3))  # wider figure

# Top row
for i in range(N):
    color = cmap(norm(array1[i]))
    rect = plt.Rectangle((i*box_size, 2), box_size, box_size,
                         facecolor=color, edgecolor='black')
    ax.add_patch(rect)


# Super Res
num_small_boxes = 2*N-1
for i in range(num_small_boxes):
    color = cmap(norm(super_res[i]))
    rect = plt.Rectangle(((i+1)*(box_size/2), 1), box_size/2, box_size,
                         facecolor=color, edgecolor='black')
    ax.add_patch(rect)


# Bottom row shifted by half box
for i in range(N):
    color = cmap(norm(array2[i]))
    x_shifted = i*box_size + box_size/2
    rect = plt.Rectangle((x_shifted, 0), box_size, box_size,
                         facecolor=color, edgecolor='black')
    ax.add_patch(rect)

# Colorbar under the plot
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # only needed for older mpl versions
cbar = plt.colorbar(sm, orientation='horizontal', ax=ax, pad=0.1, fraction=0.04)
cbar.set_label('Measured signal current (pA)')

# Adjust axes
total_width = N*box_size + box_size/2
ax.set_xlim(-1, total_width+1)
ax.set_ylim(-0.5, 5.2)
ax.axis('off')

center = (ax.get_xlim()[1] - ax.get_xlim()[0]) // 2
# '''
# Add row labels on the left
ax.text(center, 2.5, 'Array 1', fontsize=13, va='center', ha='center', bbox={'facecolor': 'white', 'edgecolor': 'white', 'alpha': 1.0, 'pad': 2, 'zorder': 10})
ax.text(center, 1.5, 'Simple Super Res', fontsize=13, va='center', ha='center', bbox={'facecolor': 'white', 'edgecolor': 'white', 'alpha': 1.0, 'pad': 2, 'zorder': 10})
ax.text(center, 0.5, 'Array 2', fontsize=13, va='center', ha='center', bbox={'facecolor': 'white', 'edgecolor': 'white', 'alpha': 1.0, 'pad': 2, 'zorder': 10})
# '''

plot_size = [fullsize_plot[0] * 3, fullsize_plot[1]]
just_save(Path('/Users/nico_brosda/Cyrce_Messungen/Style') / 'SuperRes', 'SuperResSimple', plot_size=plot_size)
