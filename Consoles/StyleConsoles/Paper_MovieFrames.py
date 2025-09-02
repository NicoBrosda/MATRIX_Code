from EvaluationSoftware.main import *
from Consoles.StyleConsoles.Utils_ImageLoad import *

mapping = Path('../../Files/mapping.xlsx')
direction1 = pd.read_excel(mapping, header=1)
direction1 = np.array([int(k[-3:]) for k in direction1['direction_1']])
direction2 = pd.read_excel(mapping, header=1)
direction2 = np.array([int(k[-3:]) for k in direction2['direction_2']])
mapping = Path('../../Files/Mapping_MatrixArray.xlsx')
data2 = pd.read_excel(mapping, header=None)
mapping_map = data2.to_numpy().flatten()
translated_mapping = np.array([direction2[np.argwhere(direction1 == i)[0][0]]-1 for i in mapping_map])
readout, position_parser = lambda x, y: ams_2D_assignment_readout(x, y, channel_assignment=translated_mapping), standard_position
A = Analyzer((11, 11), 0.8, 0.2, readout=readout)
A.scale = 'nano'

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_111024/')
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Style/Paper/')

dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_111024/')
matrix_dark = ['2DLarge_dark_200_um_0_nA__nA_1.9_x_21.0_y_70.35.csv']

frame_start = 1312
frame_space = 2
frames = 8
frames = [frame_start + i*frame_space for i in range(frames)]

plot_size = (18*cm, 9.3/1.2419*cm)
save_format = '.pdf'
dpi = 300
cmap=matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black", "red", "yellow"])

# '''
crit = '2DLarge_movieScan_'
files = os.listdir(folder_path)
map_storage = []
names = []
for i in frames:
    A.set_measurement(folder_path, '_'+str(i)+'_'+crit)
    A.set_dark_measurement(dark_path, matrix_dark)
    norm_func = lambda list_of_files, instance, method='least_squares': normalization_from_translated_array_v3(
        list_of_files, instance, method, align_lines=True)
    A.normalization(folder_path, ['2DLarge_YScan_'], normalization_module=norm_func)
    A.load_measurement()
    A.create_map(inverse=[True, False])
    map_storage.append(A.maps)
    names.append(A.name)

fig, axs = plt.subplots(nrows=2, ncols=len(map_storage)//2, figsize=plot_size)
fig.subplots_adjust(wspace=0, hspace=0)
intensity_limits = [1, np.max([np.max(i[0]['z']) for i in map_storage])*1.0]
for i, image_map in tqdm(enumerate(map_storage)):
    cbar = False
    y_ticks = False
    x_ticks = False
    if i == 0 or i == len(frames)//2:
        y_ticks = True
    if i >= len(frames)//2:
        x_ticks = True

    ax = axs.flatten()[i]
    A.name = names[i]
    A.maps = map_storage[i]
    A.maps[0]['z'] = zero_pixel_replace(A.maps[0]['z'])
    A.plot_map(None, pixel='fill', save_format='png', colorbar=cbar, cmap=cmap,
               intensity_limits=intensity_limits, imshow=True, ax_in=ax, fig_in=fig)

    ax.text(*transform_axis_to_data_coordinates(ax, [0.97, 0.97]), r'\textbf{'+f'{i*frame_space}$\\,$ms'+r'}', fontsize=8, ha='right',
            va='top', color='k')
    if i == 0:
        add_png_icon(ax, A, 'top left', translation=None, zoom=0.1)

    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    if not y_ticks:
        ax.set_yticklabels([])
        ax.set_ylabel(None)
    if not x_ticks:
        ax.set_xticklabels([])
        ax.set_xlabel(None)



norm = matplotlib.colors.Normalize(vmin=intensity_limits[0], vmax=intensity_limits[1])
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
if intensity_limits[0] > 0:
    extend='both'
else:
    extend='max'

# Add a global one
bar = fig.colorbar(sm,
    ax=axs,
    orientation="vertical",
    extend=extend,
    pad = 0.02
)
bar.set_label(f'Signal Current ({scale_dict[A.scale][1]}A)')

bar.ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
bar.ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

just_save(save_path=results_path, save_name=f"MovieFramesWire", dpi=dpi, plot_size=plot_size, save_format=save_format, fig=fig)