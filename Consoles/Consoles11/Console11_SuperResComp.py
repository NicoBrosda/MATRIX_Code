import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from Consoles.StyleConsoles.Utils_ImageLoad import *

# Save path and options for the map
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Style/ImageSandbox/Matilde')
background_subtraction = True
normalization = True

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
for crit in ['_GafComp200_', '_GafComp400_', '_GafComp40_', '_GafCompLogo_', '_GafCompMisc_', '_GafCompPEEK_',
                    '_MouseFoot_', '_MouseFoot2_', '2Line_Beam_']:

    image = crit
    # image = 'Array3_Logo'
    position = None


    A = load_image(folder_path, image, background_subtraction=background_subtraction, normalization=normalization,
                   position=position)
    for i, image_map in enumerate(A.maps):
        A.maps[i]['z'] = simple_zero_replace(image_map['z'])

    # Parameters
    N = np.shape(A.maps[0]['z'])[0] // 2 - 1

    box_size = 1
    for row in [16, 30, 50]:
        x = np.arange(N)
        center = N / 2
        sigma = N / 6
        array1 = A.maps[0]['z'].T[row, 0::2]
        array2 = A.maps[0]['z'].T[row, 1::2]
        input_array = A.maps[0]['z'].T[row]
        super_res_profile1, res1, rank1, s1 = np.linalg.lstsq(super_resolution_matrix_correct_red(np.shape(input_array)[0])*1/2, input_array[:-1], rcond=None)
        super_res_profile2, res2, rank2, s2 = np.linalg.lstsq(super_resolution_matrix_correct_red(np.shape(input_array)[0])*1/2, input_array[1:], rcond=None)

        M_left = super_resolution_matrix_correct_red(np.shape(input_array)[0])
        M_right = super_resolution_matrix_correct_red(np.shape(input_array)[0])
        FG_left = input_array[:-1]
        FG_right = input_array[1:]
        M_combined = np.vstack([M_left, M_right]) * 1/2
        FG_combined = np.hstack([FG_left, FG_right])
        super_res, res, rank, s = np.linalg.lstsq(M_combined, FG_combined, rcond=None)

        simple = np.zeros(2 * N - 1)

        for n in range(N):
            if n == 0:
                simple[2*n] = array1[n] + array2[n]
            else:
                simple[2*n] = array1[n] + array2[n]
                simple[2*n-1] = array1[n] + array2[n-1]

        simple = simple / 2

        '''
        N = len(input_array) // 2
        super_res = np.zeros(2 * N - 1)
        
        for n in range(2*N-1):
            super_res[n] = ((n+1)/(2*N-1) * super_res_profile1[n] + (2*N-1-n)/(2*N-1) * super_res_profile2[n])
            # super_res[n] = (super_res_profile1[n] + super_res_profile2[n])/2
        '''

        # Middle row: 2N+2 elements, simulate same Gaussian sampled at higher resolution

        middle_array = A.maps[0]['z'].T[row]

        # Normalize data to [0,1] for coloring
        norm = mcolors.Normalize(vmin=np.min(A.maps[0]['z']), vmax=np.max(A.maps[0]['z']))

        fig, ax = plt.subplots(figsize=(2*N/4, 5))  # wider figure

        # Top row
        for i in range(N):
            color = cmap(norm(array1[i]))
            rect = plt.Rectangle((i*box_size, 4), box_size, box_size,
                                 facecolor=color, edgecolor='black')
            ax.add_patch(rect)

        # Super Res
        num_small_boxes = 2*N
        for i in range(num_small_boxes):
            color = cmap(norm(super_res[i]))
            rect = plt.Rectangle(((i+0)*(box_size/2), 3), box_size/2, box_size,
                                 facecolor=color, edgecolor='black')
            ax.add_patch(rect)

        # SuperRes Simple
        num_small_boxes = 2*N-1
        for i in range(num_small_boxes):
            color = cmap(norm(simple[i]))
            rect = plt.Rectangle(((i+1)*(box_size/2), 2), box_size/2, box_size,
                                 facecolor=color, edgecolor='black')
            ax.add_patch(rect)

        # SuperRes Simple
        num_small_boxes = 2 * N
        for i in range(num_small_boxes):
            color = cmap(norm(input_array[i]))
            rect = plt.Rectangle((i * (box_size / 2), 1), box_size / 2, box_size,
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
        ax.text(center, 4.5, 'Array 1', fontsize=13, va='center', ha='center', bbox={'facecolor': 'white', 'edgecolor': 'white', 'alpha': 1.0, 'pad': 2, 'zorder': 10})
        ax.text(center, 3.5, 'Super-res', fontsize=13, va='center', ha='center', bbox={'facecolor': 'white', 'edgecolor': 'white', 'alpha': 1.0, 'pad': 2, 'zorder': 10})
        ax.text(center, 2.5, 'Super-res Simple', fontsize=13, va='center', ha='center', bbox={'facecolor': 'white', 'edgecolor': 'white', 'alpha': 1.0, 'pad': 2, 'zorder': 10})
        ax.text(center, 1.5, 'Ignoring overlay', fontsize=13, va='center', ha='center', bbox={'facecolor': 'white', 'edgecolor': 'white', 'alpha': 1.0, 'pad': 2, 'zorder': 10})
        ax.text(center, 0.5, 'Array 2', fontsize=13, va='center', ha='center', bbox={'facecolor': 'white', 'edgecolor': 'white', 'alpha': 1.0, 'pad': 2, 'zorder': 10})
        # '''

        plot_size = [fullsize_plot[0] * 3, fullsize_plot[1]]
        just_save(Path('/Users/nico_brosda/Cyrce_Messungen/Style') / f'SuperRes/{crit}/', f'{crit}_{row}_SuperResFormulaCorrect', plot_size=plot_size)
