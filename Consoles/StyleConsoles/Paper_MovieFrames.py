from copy import deepcopy

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
readout, position_parser = lambda x, y: ams_2D_assignment_readout_WPE_choice(x, y, channel_assignment=translated_mapping, sample_size=[0, 100], version='mean'), standard_position
A = Analyzer((11, 11), 0.8, 0.2, readout=readout)
A.scale = 'nano'

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_111024/')
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Style/Paper/')

dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_111024/')
matrix_dark = ['2DLarge_dark_200_um_0_nA__nA_1.9_x_21.0_y_70.35.csv']

zero_frame = 1333
frame_start = 1309
frame_space = 2
fps = 0.7
frames = 8
frames = [frame_start + i*frame_space for i in range(frames)]

plot_size = (18*cm, 9.3/1.2419*cm)
save_format = '.pdf'
dpi = 300
cmap=matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black", "red", "yellow"])

# '''
crit = '2DLarge_movieScan_'
files = os.listdir(folder_path)

if zero_frame is not None:
    A.set_measurement(folder_path, '_'+str(zero_frame)+'_'+crit)
    A.set_dark_measurement(dark_path, matrix_dark)
    norm_func = lambda list_of_files, instance, method='least_squares': normalization_from_translated_array_v3(
        list_of_files, instance, method, align_lines=True)
    A.normalization(folder_path, ['2DLarge_YScan_'], normalization_module=norm_func)
    A.load_measurement()
    A.create_map(inverse=[True, False])
    zero_map = zero_pixel_replace(A.maps[0]['z'])
else:
    zero_map = 0

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
    A.maps[0]['z'] = np.abs(zero_pixel_replace(A.maps[0]['z']) - zero_map)
    map_storage.append(A.maps)
    names.append(A.name)

fig, axs = plt.subplots(nrows=2, ncols=len(map_storage)//2, figsize=plot_size)
fig.subplots_adjust(wspace=0, hspace=0)
intensity_limits = [0, np.max([np.max(i[0]['z']) for i in map_storage])*1.0]
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
    A.maps[0]['z'] = map_storage[i][0]['z']
    A.plot_map(None, pixel='fill', save_format='png', colorbar=cbar, cmap=cmap,
               intensity_limits=intensity_limits, imshow=True, ax_in=ax, fig_in=fig, alpha=1)

    if i == 0:
        cmap2 = sns.color_palette("Greys", as_cmap=True)
        A.maps[0]['z'] = zero_map
        A.plot_map(None, pixel='fill', save_format='png', colorbar=False, cmap=cmap2,
                   intensity_limits=[1, np.max(zero_map)], imshow=True, ax_in=ax, fig_in=fig, alpha=1, zorder=-1)

    ax.text(*transform_axis_to_data_coordinates(ax, [0.97, 0.97]), r'\textbf{'+f'{i*frame_space/fps:.1f}$\\,$s'+r'}', fontsize=8, ha='right',
            va='top', color='k')

    ax.set_xlim(ax.get_xlim()), ax.set_ylim(ax.get_ylim())

    # 1 mm spacing for minor ticks
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1.0))

    # hide major ticks completely
    ax.xaxis.set_major_locator(ticker.NullLocator())
    ax.yaxis.set_major_locator(ticker.NullLocator())

    ax.set_yticklabels([])
    ax.set_ylabel(None)

    ax.set_xticklabels([])
    ax.set_xlabel(None)

    if i == 0:
        add_png_icon(ax, A, 'top left', translation=None, zoom=0.1)

        # 1 mm square positioned wherever you like:
        x0, y0 = 18, int(ax.get_ylim()[0]) + 2   # lower-left corner of the L
        span = 2.0  # mm

        # center the cross at some point in your data coordinates
        center = (x0, y0)  # x=18 mm, y=5 mm
        span = 2.0  # 1 mm in both directions
        span_arrow_cross(ax, center=center, span=span, tick_size=0.3, lw=1)
        # span_arrow(ax, (x0, y0), (x0, y0+span), lw=1)
        ax.text(x0+0.5, y0 + span / 2, fr"{span:.0f}$\,$mm", ha='left', va='center', fontsize=9)


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
bar.set_label(f'Signal Current difference ({scale_dict[A.scale][1]}A)')

bar.ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
bar.ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

# plt.show()
format_save(save_path=results_path, save_name=f"Graph7_MovieFramesWire", dpi=dpi, plot_size=plot_size,
            major_ticks=[False, False], minor_xticks=False, minor_yticks=False, save_format=save_format, fig=fig)