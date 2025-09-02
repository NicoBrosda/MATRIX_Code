import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from Consoles.StyleConsoles.Utils_ImageLoad import *

# Save path and options for the map
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Style/ImageSandbox/Matilde')
background_subtraction = True
normalization = True

# Axis limits
x_limits = np.array([20 - 16, 20 + 16])
y_limits = [66.5 - 16, 66.5 + 16]
# y_limits = np.array([94, 126])
zero_scale = True

intensity_limits = None
pixel = 'fill'

plot_size = fullsize_plot
plot_size = (6 * cm, 6 / 1.2419 * cm)

dpi = 300
format = '.svg'

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black", "red", "yellow"])

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

    print(np.shape(A.maps[0]['z']))
    fig, ax = plt.subplots()
    A.plot_map(None, pixel=pixel, intensity_limits=intensity_limits, ax_in=ax, fig_in=fig, cmap=cmap, imshow=True)
    format_save(Path('/Users/nico_brosda/Cyrce_Messungen/Style') / f'SuperRes/{crit}/', f'{crit}_NoSuperRes',
              fig=fig)

    fig, ax = plt.subplots()
    A.maps[0] = overlap_treatment(A.maps[0], A, super_res=True)
    A.plot_map(None, pixel=pixel, intensity_limits=intensity_limits, ax_in=ax, fig_in=fig, cmap=cmap, imshow=True)
    format_save(Path('/Users/nico_brosda/Cyrce_Messungen/Style') / f'SuperRes/{crit}/', f'{crit}_SuperRes',
              fig=fig)
