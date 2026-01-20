from copy import deepcopy
from datetime import datetime

from EvaluationSoftware.main import *
from EvaluationSoftware.readout_modules import ams_channel_assignment_readout
from EvaluationSoftware.normalization_modules import normalization_from_translated_array
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# -------------------------
# Parameters (scalable!)
# -------------------------
L = 40          # wedge length
h1 = 2.14          # start thickness
h2 = 5.64         # end thickness
h3 = 5  # desired wedge height
x0 = 0

L2 = 32.25         # color box length
x_box = 5       # where color box starts
h_box = 6
y_box = 6      # vertical placement

diode_colour = sns.color_palette("icefire", as_cmap=True)

def draw_wedge(ax, x0, L, h1, h2, y0=0, **kwargs):
    points = [
        (x0, y0),
        (x0 + L, y0),
        (x0 + L, y0 + h2),
        (x0, y0 + h1),
    ]
    ax.add_patch(Polygon(points, closed=True, **kwargs))


def x_at_height(h3, h1, h2, L, x0=0):
    """
    Return x-position where a linear wedge reaches height h3.
    """
    if not (min(h1, h2) <= h3 <= max(h1, h2)):
        raise ValueError("h3 is outside the wedge height range")

    return x0 + L * (h3 - h1) / (h2 - h1)


def draw_color_box(ax, x_start, L2, y0, h_box, n_steps=128, cmap="viridis"):
    dx = L2 / n_steps
    patches = []
    colors = np.linspace(0, 1, n_steps)

    for i, c in enumerate(colors):
        patches.append(
            Rectangle((x_start + i * dx, y0), dx, h_box)
        )

    collection = PatchCollection(patches, cmap=cmap)
    collection.set_array(colors)
    ax.add_collection(collection)


def draw_height_line(ax, h3, h1, h2, L, x0=0, **kwargs):
    x_h3 = x_at_height(h3, h1, h2, L, x0)

    ax.vlines(
        x=x_h3,
        ymin=ax.get_ylim()[0],
        ymax=ax.get_ylim()[1],
        linestyles="dashed",
        **kwargs
    )

    return x_h3


def add_schematic(ax, size=(35, 35), loc='upper right', diode_colour=diode_colour):
    ax_inset = inset_axes(
        ax,
        width=f"{size[0]}%",     # relative to parent axis
        height=f"{size[1]}%",
        loc="upper right",
        borderpad=0
    )

    # Inset cosmetics
    ax_inset.set_aspect("equal")
    ax_inset.set_xlim(-2, 45)
    ax_inset.set_ylim(-2, (h2-h1+h_box)*1.2)
    ax_inset.axis("off")   # schematic only

    # Draw schematic inside inset
    draw_wedge(
        ax_inset,
        x0=x0,
        L=L,
        h1=h1,
        h2=h2,
        facecolor="lightgray",
        edgecolor="black"
    )

    draw_color_box(
        ax_inset,
        x_start=x_box,
        L2=L2,
        y0=y_box,
        h_box=h_box,
        n_steps=128,
        cmap=diode_colour
    )

    x_h3 = draw_height_line(
        ax_inset,
        h3=h3,
        h1=h1,
        h2=h2,
        L=L,
        color="red",
        linewidth=1.5
    )


mapping = Path('../../Files/mapping.xlsx')
data = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:])-1 for k in data['direction_2']]
readout, position_parser, voltage_parser, current_parser = (
    lambda x, y: ams_2line_fast_avg(x, y, channel_assignment=channel_assignment),
    standard_position,
    standard_voltage,
    current5
)

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_171225/')
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/ResultsPEEK_161225/')

A = Analyzer((2, 64), (0.4, 0.4), (0.1, 0.1), readout=readout,
                 diode_offset=[[0, - 0.25], np.zeros(64)], position_parser=position_parser,
                 voltage_parser=voltage_parser, current_parser=current_parser)
dark_path = folder_path
dark = ['dark_current.csv']
A.set_dark_measurement(dark_path, dark)
norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_260325/')
norm_array1 = ['exp7_norm1,9V_']
norm = norm_array1
norm_func = lambda list_of_files, instance, method='least_squares': normalization_from_translated_array_v3(
    list_of_files, instance, method, align_lines=True)
A.normalization(norm_path, norm, normalization_module=norm_func)
A.scale = 'nano'

measurements = array_txt_file_search(os.listdir(folder_path), searchlist=['exp2_', 'exp4_'], file_suffix='.csv', txt_file=False, blacklist=['.png'])

def time_parser(input_string, parser=None):
    time_string = input_string[input_string.rindex('_')+1:input_string.rindex('.csv')]
    if parser is None:
        return int(time_string)
    else:
        return parser(int(time_string))

measurements = sorted(
    measurements,
    key=lambda s: int(s.rsplit("_", 1)[1].replace(".csv", ""))
)
time_0 = time_parser(measurements[0], parser=datetime.fromtimestamp)
times = np.array([(time_parser(i, parser=datetime.fromtimestamp)-time_0).total_seconds() / 60 for i in measurements])

try:
    cache = np.load(results_path / 'long_term_cache.npy')
except FileNotFoundError:
    cache = []
    for measurement in tqdm(measurements):
        cache.append(A.readout(folder_path / measurement, A)['signal'])
    cache = np.array(cache)
    np.save(results_path / 'long_term_cache.npy', cache)

print(np.shape(cache))
# cache = cache.transpose(0, 2, 1).reshape(cache.shape[0], -1)
cache2 = np.empty((cache.shape[0], 128), dtype=cache.dtype)

cache2[:, 0::2] = cache[:, 1, :]
cache2[:, 1::2] = cache[:, 0, :]
cache = cache2

print(np.shape(cache))
cache = A.signal_conversion(cache)

# Adjust the time structure and homogenize the current structure of the response
end_8nA = np.argsort(np.abs(times-39))[0]
cache = np.concatenate((cache[:end_8nA] * 25/8, cache[end_8nA:]))

time_steps = [times[i+1]-times[i] for i in range(len(times)-1)]
for i, step in enumerate(time_steps):
    if step > 3:
        times = np.concatenate((times[:i+1], times[i+1:]-step))

print(np.shape(cache))
print(np.shape(times))

np.savetxt(results_path / 'long_term_data.txt', cache, delimiter=',', fmt='%.6f')
np.savetxt(results_path / 'long_term_times.txt', times, delimiter=',', fmt='%.3f')

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

fig, ax = plt.subplots()

time_colour = sns.color_palette("flare_r", as_cmap=True)

for i in range(np.shape(cache)[1]):
    ax.plot(i, cache[0, i], marker='x', color=diode_colour(i/np.shape(cache)[1]))
    ax.plot(i, cache[100, i], marker='x', color=diode_colour(i/np.shape(cache)[1]))
    ax.plot(i, cache[300, i], marker='x', color=diode_colour(i/np.shape(cache)[1]))
    ax.plot(i, cache[500, i], marker='x', color=diode_colour(i/np.shape(cache)[1]))
    ax.plot(i, cache[1000, i], marker='x', color=diode_colour(i/np.shape(cache)[1]))
    ax.plot(i, cache[2600, i], marker='x', color=diode_colour(i/np.shape(cache)[1]))
    ax.plot(i, cache[-1, i], marker='x', color=diode_colour(i/np.shape(cache)[1]))

print(times[100], times[300], times[500], times[1000], times[2600])
ax.set_xlabel('Diode ()')
ax.set_ylabel(f'Measurement signal ({scale_dict[A.scale][-1]}A)')

format_save(results_path, f'LongTermIrradiationBragg_ArrayAssignment', save_format='.png',)

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

fig, ax = plt.subplots()

for i in range(np.shape(cache)[1]):
    ax.plot(times, cache[:, 127-i], marker='x', ls='', color=diode_colour(1-i/128))

ax.set_xlabel('Irradiation Time (min)')
ax.set_ylabel(f'Measurement signal ({scale_dict[A.scale][-1]}A)')

add_schematic(ax)

format_save(results_path, f'LongTermIrradiationBragg_Signal', save_format='.png',)

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

fig, ax = plt.subplots()

time_colour = sns.color_palette("flare_r", as_cmap=True)

for i in range(np.shape(cache)[0]):
    ax.plot(cache[i], marker='', ls='-', color=time_colour(i/np.shape(cache)[0]))
    ax.plot(np.argmax(cache[i]), np.max(cache[i]), marker='|', color='k', zorder=3)

ax.set_xlabel('Diode ()')
ax.set_ylabel(f'Measurement signal ({scale_dict[A.scale][-1]}A)')

ax.set_xlim(ax.get_xlim()), ax.set_ylim(ax.get_ylim())

improved_gradient_scale(times, time_colour, ax, 'min')
format_save(results_path, f'LongTermIrradiationBragg_ArrayCurves', save_format='.png',)

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

print(times[-1])