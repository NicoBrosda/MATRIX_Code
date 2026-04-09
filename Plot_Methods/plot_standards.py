# Some imports of necessary packages and definitions of standard name spaces
import matplotlib.pyplot as plt  # Used in this script is matplotlib 3.7.2 - newer versions should also be no problem
import locale
from matplotlib import ticker
import time
from matplotlib.patches import Rectangle
import warnings
import seaborn as sns  # v. 0.12.2 (seaborn isn't necessarily required, but offers nice colour schemes)
from Plot_Methods.helper_functions import *
from Plot_Methods.label_standard import *
import matplotlib
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm

# Standard save format for plots
save_format = '.svg'

# Tex or no Tex?
use_LaTeX = True

# Different local settings for plot formatting - e.g. we tell matplotlib to use german language standards if this is our
# defined language. Note that changes defined before the call of this script part are ignored with plt.defaults()
if not language_english:
    locale.setlocale(locale.LC_ALL, "de_DE")
    plt.rcdefaults()
    # plt.rc('axes.formatter', use_locale=True)
    plt.rcParams['axes.formatter.use_locale'] = True

# Use Latex for plot formatting and define font family: - Warning! This requires you to have a valid Latex distribution
# installed on your system -> For further information consider: https://matplotlib.org/stable/tutorials/text/usetex.html
plt.rcParams.update({
    "text.usetex": use_LaTeX,
})
# Here we define the font family we want to use -> This can and should be equal to the font used in the Text Editor were
# the graphics are implemented
if use_LaTeX:
    plt.rcParams["font.family"] = ["Latin Modern Roman"]

# It is reasonable to save graphs directly in the size that's later used in your text document! This way, all font sizes
# keep their defined size and match to the font of the text.
# Since matplotlib only speaks in inch, we need some conversions from inch to cm, which are directly applied below
cm = 1 / 2.54  # centimeters in inches
# Here I inserted the Textwidth of my LaTeX document, that also defines the size of my graphs. In LaTeX you can get this
# information with "\printinunitsof{cm}\prntlen{\textwidth}" (\usepackage{layouts} in header). Nevertheless, the value
# should normally not deviate if you write a DIN-A4 document
latex_textwidth = 14.6979*cm

# This defines the size of plots used in the text. E.g. I defined a standard fullsize_plot where the ratio of width to
# height is 1.3. This can easily be adapted for other formats.
fullsize_plot = (latex_textwidth, latex_textwidth/1.3)
fullsize_plot = (latex_textwidth, latex_textwidth / 1.2419)
halfsize_plot = (latex_textwidth/2, latex_textwidth/1.3/2)
totalpage_plot = (14.6979*cm, 20*cm)
totalpage_plot_side = (20*cm, 14.6979*cm)

# This is e.g. used for plots that take 45% of the whole documents side width of ca. 21cm. Though, letting plots be
# wider than the text_width is not advised if you're not 100% sure, that there are no complications with printing the
# document.
biggerboxes_plot = (20.99701*cm*0.45, 20.99701*cm*0.45/1.3)

# Now one can also define standard text sizes for the plots and assign where to use which size. The standards below are
# reasonable choices for a DIN A4 document - you should nevertheless check if these values really fit your document
SMALL_SIZE = 10  # 8
MEDIUM_SIZE = 11  # 9
BIGGER_SIZE = 11

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

poster = False
if poster:
    posterfont = 20
    plt.rc('font', size=posterfont)  # controls default text sizes
    plt.rc('axes', titlesize=posterfont)  # fontsize of the axes title
    plt.rc('axes', labelsize=posterfont)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=posterfont-4)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=posterfont-4)  # fontsize of the tick labels
    plt.rc('legend', fontsize=posterfont)  # legend fontsize
    plt.rc('figure', titlesize=posterfont+4)  # fontsize of the figure title
    fullsize_plot = (28 * cm, 28 * cm / 1.2419)

presentation = True
if presentation:
    posterfont = 16
    plt.rc('font', size=posterfont)  # controls default text sizes
    plt.rc('axes', titlesize=posterfont)  # fontsize of the axes title
    plt.rc('axes', labelsize=posterfont)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=posterfont-4)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=posterfont-4)  # fontsize of the tick labels
    plt.rc('legend', fontsize=posterfont)  # legend fontsize
    plt.rc('figure', titlesize=posterfont+4)  # fontsize of the figure title
    fullsize_plot = (16 * cm, 16 * cm / 1.2419)

paper = False
if paper:
    if use_LaTeX:
        plt.rcParams["font.family"] = ["sans-serif"]
        plt.rcParams["font.sans-serif"] = ["Helvetica"]

    paperfont = 7
    plt.rc('font', size=paperfont)  # controls default text sizes
    plt.rc('axes', titlesize=paperfont)  # fontsize of the axes title
    plt.rc('axes', labelsize=paperfont)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=paperfont)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=paperfont)  # fontsize of the tick labels
    plt.rc('legend', fontsize=paperfont-2)  # legend fontsize
    plt.rc('figure', titlesize=paperfont)  # fontsize of the figure title
    fullsize_plot = (18 * cm, 18 * cm / 1.2419)

if use_LaTeX:
    plt.rcParams.update({'mathtext.default':  'regular'})  # No cursive text in math environment!

# This defines a simple standard layout, all plots at our chair (and in general) should fulfill:
def afp_layout(axes=None, second_axis=False, *arg, **kwarg):
    def share_x_axis(axis1, axis2):
        # Check if both axes share the same x-axis limits
        same_xlim = axis1.get_xlim() == axis2.get_xlim()
        if not same_xlim:
            return False
        # Check if both axes share the same position in figure coordinates
        same_position = axis1.transAxes == axis2.transAxes
        return same_position

    def share_y_axis(axis1, axis2):
        # Check if both axes share the same x-axis limits
        same_ylim = axis1.get_ylim() == axis2.get_ylim()
        if not same_ylim:
            return False
        # Check if both axes share the same position in figure coordinates
        same_position = axis1.transAxes == axis2.transAxes
        return same_position

    # Here it is important to hand axes if they deviate from the currently used standard axes of a plot
    if axes is None:
        axes = plt.gcf().get_axes()
    # Boxed, ticks inside the axes, no grid lines in plot + automatic detection if x- or y-axis is shared
    formatted = []
    for i, ax in enumerate(axes):
        if i in formatted:
            continue
        for j, ax2 in enumerate(axes):
            if i == j:
                continue
            if j in formatted:
                continue
            if share_x_axis(ax, ax2):
                ax.grid(which='both', visible=False)
                ax.tick_params(which='both', direction='in', top=True, right=False, left=True, bottom=True)
                ax2.grid(which='both', visible=False)
                ax2.tick_params(which='both', direction='in', top=True, right=True, left=False, bottom=True)
                formatted.append(i)
                formatted.append(j)
            elif share_y_axis(ax, ax2):
                ax.grid(which='both', visible=False)
                ax.tick_params(which='both', direction='in', top=False, right=True, left=True, bottom=True)
                ax2.grid(which='both', visible=False)
                ax2.tick_params(which='both', direction='in', top=True, right=True, left=True, bottom=False)
                formatted.append(i)
                formatted.append(j)
        if i not in formatted:
            ax.grid(which='both', visible=False)
            ax.tick_params(which='both', direction='in', top=True, right=True, left=True, bottom=True)
            formatted.append(i)
        '''
        if i < len(axes)-1 and share_x_axis(ax, axes[i+1]):
            print('Yes')
            ax.grid(which='both', visible=False)
            ax.tick_params(which='both', direction='in', top=True, right=False, left=True, bottom=True)
            axes[i+1].grid(which='both', visible=False)
            axes[i+1].tick_params(which='both', direction='in', top=True, right=True, left=False, bottom=True)
            formatted.append(i)
            formatted.append(i+1)
        elif i < len(axes)-1 and share_y_axis(ax, axes[i+1]):
            ax.grid(which='both', visible=False)
            ax.tick_params(which='both', direction='in', top=False, right=True, left=True, bottom=True)
            axes[i+1].grid(which='both', visible=False)
            axes[i+1].tick_params(which='both', direction='in', top=True, right=True, left=True, bottom=False)
            formatted.append(i)
            formatted.append(i+1)
        else:
            ax.grid(which='both', visible=False)
            ax.tick_params(which='both', direction='in', top=True, right=True, left=True, bottom=True)
            formatted.append(i)
        '''

# I wrapped the afp_layout in another functions that allows deviations from the layout without touching the standards
# from our chair
def standard_layout(axes=None, second_axis=False, *args, **kwargs):
    if axes is None:
        axes = plt.gcf().get_axes()

    afp_layout(axes, *args, second_axis=second_axis, **kwargs)


# Python consists of a standard colour cycle for plots on the same axes. With the code below it is possible to change
# the colour cycle "tab10" that is normally used.
# Compare: https://matplotlib.org/stable/tutorials/colors/colormaps.html and especially
# https://seaborn.pydata.org/tutorial/color_palettes.html
# Currently this code is commented because I cannot really recommend changing the standard cycle "tab10". The standard
# cycle is already offering a good distinctive difference between colours. Worth considering is how colorblindness can
# impact your plots, a nice example can be found in: https://gist.github.com/mwaskom/b35f6ebc2d4b340b4f64a4e28e778486
# mpl.rcParams['axes.prop_cycle'] = cycler('color', sns.color_palette("bright"))


# If you want a nice representation for parameters in a certain range, this example code below might be helpful for you.
# The code maps the parameter space of e.g. RTA temperatures from a certain range (400-900)°C to a colour palette (blue
# to red). Fort this I simply map the parameter range to the interval 0-1, since all colour palettes are accessed with
# numbers from 0-1.
def color_mapper(x: float, minx: float, maxx: float, start: float = 0.0, end: float = 1.0):
    if x < minx:
        x = minx
    if x > maxx:
        x = maxx
    return ((x * (end-start)) / (maxx-minx)) - ((minx * (end-start)) / (maxx - minx)) + start


rta_colourmap = sns.color_palette("icefire", as_cmap=True)
# rta_colourmapper = lambda x: color_mapper(x, 500, 900, end=1)  # Simple definition with lambda expression


def rta_colourmapper(x):
    if x <= 700:
        return color_mapper(x, 500, 700, start=0.1, end=0.4)
    elif 700 < x:
        return color_mapper(x, 700, 900, start=0.6, end=1)


fluence_colourmap = sns.color_palette("flare", as_cmap=True)


def fluence_colourmapper(x):
    if x <= 5:
        return color_mapper(x, 0.8, 4, start=0, end=1)
    elif 5 < x <= 50:
        return color_mapper(x, 10, 50, start=0.5, end=0.9)
    elif 50 < x:
        return color_mapper(x, 495, 505, start=0.9, end=1.0)


# This defines a colour range for measurement temperatures between 5 and 40 K
temperature_colourmap = sns.color_palette("crest_r", as_cmap=True)
temperature_colourmapper = lambda x: color_mapper(x, 5, 40)
temp_colour = lambda x: temperature_colourmap(temperature_colourmapper(x))


# I use some workarounds to ensure that the switch between the German and English plot norms is working: mainly this
# replaces '.' with ',' or the other way around. This is helpful if Python automatically writes values in the plots.

def dot_replace(string):
    if use_LaTeX:
        return string.replace('.', '{,}')
    else:
        return string.replace('.', ',')


def comma_replace(string):
    if use_LaTeX:
        return string.replace(',', '{.}')
    else:
        return string.replace(',', '.')


def format_func(c, pos):  # formatter function takes tick label and tick position
    s = str(c)
    ind = s.index('.')
    if use_LaTeX:
        return s[:ind]+'{,}'+s[ind+1:]  # change dot to comma
    else:
        return s[:ind] + ',' + s[ind + 1:]  # change dot to comma


if language_english:
    def format_func(c, pos, after=1):  # formatter function takes tick label and tick position
        if isinstance(c, str):
            try:
                c = comma_replace(c)
                c = float(c)
            except ValueError:
                return c
        s = '{c:.'+str(after)+'f}'
        s = s.format(c=c)
        return s  # change dot to comma
else:
    def format_func(c, pos, after=1):  # formatter function takes tick label and tick position
        if isinstance(c, str):
            try:
                c = comma_replace(c)
                c = float(c)
            except ValueError:
                return dot_replace(c)
        s = '{c:.'+str(after)+'f}'
        s = s.format(c=c)
        return dot_replace(s)  # change dot to comma

# A ticker is an object from matplotlib to define the tick positions and labels. Here a costume formatter is defined
# that automatically adapts , or . depending on the language setting
fmt = ticker.FuncFormatter(format_func)


def gradient_arrow(ax, start, end, cmap="viridis", n=50, lw=3, *args, **kwargs):
    # Zuerst die Punkte und Segmente erzeugen
    x = np.linspace(start[0], end[0], n)
    y = np.linspace(start[1], end[1], n)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Überprüfen, ob cmap eine Funktion oder ein String ist
    if callable(cmap):
        # Wenn cmap eine Funktion ist, erstellen wir ein benutzerdefiniertes Farbschema
        colors = [cmap(i / n) for i in range(n)]
        cmap_obj = matplotlib.colors.LinearSegmentedColormap.from_list('custom_cmap', colors, N=n)
        lc = LineCollection(segments, cmap=cmap_obj, linewidth=lw, *args, **kwargs)
        # Auch für den Pfeilkopf verwenden
        arrow_cmap = cmap_obj
    else:
        # Wenn cmap ein String ist, rufen wir get_cmap auf
        cmap_func = plt.get_cmap(cmap, n)
        lc = LineCollection(segments, cmap=cmap_func, linewidth=lw, *args, **kwargs)
        # Für den Pfeilkopf verwenden
        arrow_cmap = cmap_func

    lc.set_array(np.linspace(0, 1, n))
    ax.add_collection(lc)

    # Arrow head: Triangle
    tricoords = [(0, -0.4), (0.5, 0), (0, 0.4), (0, -0.4)]
    angle = np.arctan2(end[1] - start[1], end[0] - start[0])
    rot = matplotlib.transforms.Affine2D().rotate(angle)
    tricoords2 = rot.transform(tricoords)
    tri = matplotlib.path.Path(tricoords2, closed=True)
    ax.scatter(end[0], end[1], c=1, s=(2 * lw) ** 2, marker=tri, cmap=arrow_cmap, vmin=0, *args, **kwargs)
    ax.autoscale_view()


def gradient_scale(param_list, cmap_param, ax_in=None, param_unit='MeV', point=[0.1, 0.94], ha='left', va='top'):
    if ax_in is None:
        ax_in = plt.gca()

    gradient_arrow(ax_in, transform_axis_to_data_coordinates(ax_in, [0.1, point[1]-0.015]),
                           transform_axis_to_data_coordinates(ax_in, [0.1, point[1]-0.145]),
                       cmap=cmap_param, lw=10, zorder=5)
    ax_in.text(*transform_axis_to_data_coordinates(ax_in, [0.035, point[1]]),
            f'{np.min(param_list): .2f}$\\,${param_unit}', fontsize=13, c=cmap_param(0), ha=ha, va=va,
               zorder=3, bbox={'facecolor': 'w', 'alpha': 0.8, 'pad': 2, 'edgecolor': 'w'})
    ax_in.text(*transform_axis_to_data_coordinates(ax_in, [0.025, point[1]-0.23]),
            f'{np.max(param_list): .2f}$\\,${param_unit}', fontsize=13, c=cmap_param(0.99), ha=ha, va=va,
               zorder=3, bbox={'facecolor': 'w', 'alpha': 0.8, 'pad': 2, 'edgecolor': 'w'})


def improved_gradient_scale(param_list, cmap, ax_in=None, param_unit='MeV',
                            point=[0.1, 0.94], param_mapper=None, arrow_width=10, text_fontsize=13, param_format='.2f'):
    """
    Creates a gradient scale with arrow and text labels with dynamic color mapping.
    """
    if ax_in is None:
        ax_in = plt.gca()

    # Get min/max values
    param_min = np.min(param_list)
    param_max = np.max(param_list)

    # Default param_mapper - linear mapping from min to max of param_list
    if param_mapper is None:
        param_mapper = lambda p: (p - param_min) / (param_max - param_min) if param_max > param_min else 0.5

    # Get max and min of params in param_mapper
    param_min_mapper = param_mapper(param_min)
    param_max_mapper = param_mapper(param_max)

    colors = [cmap(i) for i in np.linspace(param_min_mapper, param_max_mapper, 100)]
    cmap_handed = matplotlib.colors.LinearSegmentedColormap.from_list('custom_cmap', colors, N=50)

    # Create a one-line function to map parameters to colors
    param_color = lambda p: cmap(param_mapper(p))

    # Create text strings for minimum and maximum
    min_text = f'{param_min:{param_format}}$\\,${param_unit}'
    max_text = f'{param_max:{param_format}}$\\,${param_unit}'

    x_base, y_base = point

    arrow_x = x_base
    arrow_top_y = y_base - 0.02
    arrow_bottom_y = arrow_top_y - 0.13

    # Transformation zu Datenkoordinaten
    arrow_start_data = transform_axis_to_data_coordinates(ax_in, [arrow_x, arrow_top_y])
    arrow_end_data = transform_axis_to_data_coordinates(ax_in, [arrow_x, arrow_bottom_y])

    # Draw the arrow
    gradient_arrow(ax_in, arrow_start_data, arrow_end_data, cmap=cmap_handed, lw=arrow_width, zorder=5)

    # Texte für Min/Max
    min_text_pos = transform_axis_to_data_coordinates(ax_in, [arrow_x, arrow_top_y])
    max_text_pos = transform_axis_to_data_coordinates(ax_in, [arrow_x, arrow_bottom_y - 0.045])

    ax_in.text(*min_text_pos, min_text, fontsize=text_fontsize,
               c=param_color(param_min), ha='center', va='bottom', zorder=3,
               bbox={'facecolor': 'w', 'alpha': 0.8, 'pad': 2, 'edgecolor': 'w'})

    ax_in.text(*max_text_pos, max_text, fontsize=text_fontsize,
               c=param_color(param_max), ha='center', va='top', zorder=3,
               bbox={'facecolor': 'w', 'alpha': 0.8, 'pad': 2, 'edgecolor': 'w'})


# Now to the interesting part: This function represents an example how to apply settings to a generated plot, without
# the necessity to call the settings during plotting.
def format_save(save_path=Path('./Plots/'), save_name='', save=True, legend=True, legend_separator=None, x_after=0,
                y_after=0, x_fmt=format_func, y_fmt=format_func, second_axis=False, major_ticks=[True, True],
                minor_xticks=True, minor_yticks=True, english=language_english, legend_position=0,
                plot_size=fullsize_plot, x_rotation=0, plot_variant='PL_spectrum', save_format=save_format, dpi=300,
                fig=None, axes=None, bbox=None, transparent=False, *args) -> None:
    """
    :param save_path: The path where the plot is saved. The standard path just uses (or creates) a folder in the folder
    of this python script ("./Plots/")
    :param save_name: The name under which the plot is saved. If no save_name is given, the plot name is automatically
    generated out of the current time. In case of a given name, an existing plot at the specified location with the same
    name will be overwritten.
    :param save: True to save the plot.
    :param legend: True if legend shall be shown. (Grabs and merges the legends from several axes of the plot.
    :param legend_separator: This allows adding characters between the legend sections from the different axes.
    :param x_after: How many numbers shall the ticks of x-axis have behind the separator.
    :param y_after: How many numbers shall the ticks of y-axis have behind the separator.
    :param x_fmt: Format for the x-ticks -> As standard the dot/comma selector is used
    :param y_fmt: Format for the y-ticks -> As standard the dot/comma selector is used
    Note that these formatter functions also allow stuff like coloured labels
    :param second_axis: Should a second x-axis with conversions be added at the top of the plot?
    :param minor_xticks: True if minor x-ticks shall be visible.
    :param minor_yticks: True if minor y-ticks shall be visible.
    :param english: Handing over of language switch parameter.
    :param legend_position: Lets you specify the position of the legend in the plots. The standard "0" - meaning the
    best position is automatically selected by matplotlib is especially for long legend often not feasible. The position
    is given with a number or string (compare https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html)
    :param plot_size: This specifies the size the plot is saved in. As already discussed, you should avoid scaling the
    plots after saving, since this shifts also the font size.
    :param x_rotation: Rotation of x_labels. Might be necessary if labels are so long that they overlap.
    :param plot_variant: This specifies the variant of the plot and allows setting the corresponding labels.
    :param args: Further args we want to hand to the .savefig function that finally saves the plot.
    :return: None (The plot is saved and closed afterward - not doing this will eventually clog your RAM.
    """
    # get current plot
    if axes is None:
        axes = plt.gcf().get_axes()
        ax = plt.gca()
    if fig is None:
        fig = plt.gcf()

    # If x_after is not a list, make it one -> This allows x formatting of every axis alone
    if not isinstance(x_after, (list, tuple, np.ndarray)):
        x_after = np.zeros_like(axes) + x_after

    # If y_after is not a list, make it one -> This allows y formatting of every axis alone
    if not isinstance(y_after, (list, tuple, np.ndarray)):
        y_after = np.zeros_like(axes) + y_after

    # Check if there is a colorbar in the plot:
    no_colorbar = []
    for i, ax in enumerate(axes):
        if is_colorbar(ax):
            no_colorbar.append(False)
        else:
            no_colorbar.append(True)


    # string formatting
    cache = [[], []]

    for k in range(len(axes)):
        ax = axes[k]
        # set ticks automatic:

        if ax.get_xscale() == 'log' or ax.get_xscale() == 'symlog':
            pass
        else:
            if no_colorbar[k]:
                if major_ticks[0]:
                    ax.xaxis.set_major_locator(ticker.AutoLocator())
                else:
                    ax.xaxis.set_major_locator(ticker.NullLocator())
                if minor_xticks:
                    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

        if ax.get_yscale() == 'log' or ax.get_yscale() == 'symlog':
            pass
        else:
            if no_colorbar[k]:
                if major_ticks[0]:
                    ax.yaxis.set_major_locator(ticker.AutoLocator())
                else:
                    ax.yaxis.set_major_locator(ticker.NullLocator())
                if minor_yticks:
                    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            else:
                ax.yaxis.set_major_locator(ticker.AutoLocator())
                ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

        if not english:
            # Get pick labels:
            warnings.simplefilter('ignore')
            ylabels = [item.get_text() for item in ax.get_yticklabels()]
            xlabels = [item.get_text() for item in ax.get_xticklabels()]
            for i in range(len(ylabels)):
                ylabels[i] = y_fmt(ylabels[i], 0, y_after[k])
            ax.yaxis.set_ticklabels(ylabels)
            for i in range(len(xlabels)):
                xlabels[i] = x_fmt(xlabels[i], 0, x_after[k])
            if x_rotation != 0:
                ax.xaxis.set_ticklabels(xlabels, rotation=x_rotation, ha='right')
            else:
                ax.xaxis.set_ticklabels(xlabels)
            warnings.resetwarnings()

        # replace dots in tick labels:
        # ax.yaxis.set_major_formatter(lambda c, pos: y_fmt(c, pos, y_after[k]))

        # replace dots in tick labels:
        # ax.xaxis.set_major_formatter(lambda c, pos: x_fmt(c, pos, x_after[k]))

        current_handles, current_labels = ax.get_legend_handles_labels()
        if not english:
            for i in range(len(current_labels)):
                current_labels[i] = dot_replace(current_labels[i])
        cache[0] += current_handles
        cache[1] += current_labels

        if legend_separator is not None:
            extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
            cache[0] += [extra]
            separator = '----------------------------'
            cache[1] += [separator]
    if len(cache[1]) > 0 and legend:
        leg = axes[0].legend(cache[0], cache[1], loc=legend_position)

    if second_axis:
        ax2 = ax.twiny()
        ax2.set_title(ax_labels[plot_variant]['x2'])
        ax2.set_xlim(ax.get_xlim())
        ax2.xaxis.set_major_locator(ticker.AutoLocator())
        ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax2.xaxis.set_major_formatter(lambda c, pos: format_func(x_conversion(c), pos, after=1))
        ax2.tick_params(which='both', direction='in', top=True, right=False, left=False, bottom=False)

    # Set the size of the figure
    fig.set_size_inches(plot_size)

    # standard layout of the plot
    standard_layout(np.array(axes)[no_colorbar])  # Here the above defined standard layout is applied.

    fig.canvas.draw()  # If we want to save the figure we first need to draw it (without showing it)
    # plt.pause(0.0001)  # This is an easy bug-fix, where matplotlib is not updating the current figure after drawing

    # Here post-processing is possible, like adding in a legend - position dependent text etc. The plot positions are
    # now fixed. A relative positioning would shift else.
    if bbox is None:
        bb = 'tight'
    else:
        bb = bbox
    # Saving of the figure and showing if necessary
    if save_path is not None and not os.path.exists(save_path):
        os.makedirs(save_path)
    if save:
        if save_name == '':
            fig.savefig(Path(save_path) / (str(time.asctime()) + save_format), dpi=dpi, bbox_inches=bb,
                        transparent=transparent, *args)
        else:
            fig.savefig(Path(save_path) / (save_name + save_format), dpi=dpi, bbox_inches=bb,
                        transparent=transparent, *args)

        plt.close(fig=fig)


def just_save(save_path=Path('./Plots/'), save_name='', save=True, legend=True, legend_separator=None, x_after=0,
                y_after=0, x_fmt=format_func, y_fmt=format_func, second_axis=False, minor_xticks=True,
                minor_yticks=True, english=language_english, legend_position=0, plot_size=fullsize_plot, x_rotation=0,
                plot_variant='PL_spectrum', save_format=save_format, dpi=300, fig=None, axes=None, bbox=None, *args) -> None:
    """
    :param save_path: The path where the plot is saved. The standard path just uses (or creates) a folder in the folder
    of this python script ("./Plots/")
    :param save_name: The name under which the plot is saved. If no save_name is given, the plot name is automatically
    generated out of the current time. In case of a given name, an existing plot at the specified location with the same
    name will be overwritten.
    :param save: True to save the plot.
    :param legend: True if legend shall be shown. (Grabs and merges the legends from several axes of the plot.
    :param legend_separator: This allows adding characters between the legend sections from the different axes.
    :param x_after: How many numbers shall the ticks of x-axis have behind the separator.
    :param y_after: How many numbers shall the ticks of y-axis have behind the separator.
    :param x_fmt: Format for the x-ticks -> As standard the dot/comma selector is used
    :param y_fmt: Format for the y-ticks -> As standard the dot/comma selector is used
    Note that these formatter functions also allow stuff like coloured labels
    :param second_axis: Should a second x-axis with conversions be added at the top of the plot?
    :param minor_xticks: True if minor x-ticks shall be visible.
    :param minor_yticks: True if minor y-ticks shall be visible.
    :param english: Handing over of language switch parameter.
    :param legend_position: Lets you specify the position of the legend in the plots. The standard "0" - meaning the
    best position is automatically selected by matplotlib is especially for long legend often not feasible. The position
    is given with a number or string (compare https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html)
    :param plot_size: This specifies the size the plot is saved in. As already discussed, you should avoid scaling the
    plots after saving, since this shifts also the font size.
    :param x_rotation: Rotation of x_labels. Might be necessary if labels are so long that they overlap.
    :param plot_variant: This specifies the variant of the plot and allows setting the corresponding labels.
    :param args: Further args we want to hand to the .savefig function that finally saves the plot.
    :return: None (The plot is saved and closed afterward - not doing this will eventually clog your RAM.
    """

    # get current plot
    if axes is None:
        axes = plt.gcf().get_axes()
        ax = plt.gca()
    if fig is None:
        fig = plt.gcf()

    # If x_after is not a list, make it one -> This allows x formatting of every axis alone
    if not isinstance(x_after, (list, tuple, np.ndarray)):
        x_after = np.zeros_like(axes) + x_after

    # If y_after is not a list, make it one -> This allows y formatting of every axis alone
    if not isinstance(y_after, (list, tuple, np.ndarray)):
        y_after = np.zeros_like(axes) + y_after

    # Check if there is a colorbar in the plot:
    no_colorbar = []
    for i, ax in enumerate(axes):
        if is_colorbar(ax):
            no_colorbar.append(False)
        else:
            no_colorbar.append(True)

    # string formatting
    cache = [[], []]

    for k in range(len(axes)):
        ax = axes[k]

        if no_colorbar[k]:
            # ax.xaxis.set_major_locator(ticker.AutoLocator())
            if minor_xticks:
                ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

        if minor_yticks:
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

        if not english:
            # Get pick labels:
            warnings.simplefilter('ignore')
            ylabels = [item.get_text() for item in ax.get_yticklabels()]
            xlabels = [item.get_text() for item in ax.get_xticklabels()]
            for i in range(len(ylabels)):
                ylabels[i] = y_fmt(ylabels[i], 0, y_after[k])
            ax.yaxis.set_ticklabels(ylabels)
            for i in range(len(xlabels)):
                xlabels[i] = x_fmt(xlabels[i], 0, x_after[k])
            if x_rotation != 0:
                ax.xaxis.set_ticklabels(xlabels, rotation=x_rotation, ha='right')
            else:
                ax.xaxis.set_ticklabels(xlabels)
            warnings.resetwarnings()

        # replace dots in tick labels:
        # ax.yaxis.set_major_formatter(lambda c, pos: y_fmt(c, pos, y_after[k]))

        # replace dots in tick labels:
        # ax.xaxis.set_major_formatter(lambda c, pos: x_fmt(c, pos, x_after[k]))

        current_handles, current_labels = ax.get_legend_handles_labels()
        if not english:
            for i in range(len(current_labels)):
                current_labels[i] = dot_replace(current_labels[i])
        cache[0] += current_handles
        cache[1] += current_labels
        if legend_separator is not None:
            extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
            cache[0] += [extra]
            separator = '----------------------------'
            cache[1] += [separator]
    if len(cache[1]) > 0 and legend:
        leg = axes[0].legend(cache[0], cache[1], loc=legend_position)

    if second_axis:
        ax2 = ax.twiny()
        ax2.set_title(ax_labels[plot_variant]['x2'])
        ax2.set_xlim(ax.get_xlim())
        ax2.xaxis.set_major_locator(ticker.AutoLocator())
        ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax2.xaxis.set_major_formatter(lambda c, pos: format_func(x_conversion(c), pos, after=1))
        ax2.tick_params(which='both', direction='in', top=True, right=False, left=False, bottom=False)

    # Set the size of the figure
    fig.set_size_inches(plot_size)

    # standard layout of the plot
    standard_layout(np.array(axes)[no_colorbar])  # Here the above defined standard layout is applied.

    fig.canvas.draw()  # If we want to save the figure we first need to draw it (without showing it)
    # plt.pause(0.0001)  # This is an easy bug-fix, where matplotlib is not updating the current figure after drawing

    # Here post-processing is possible, like adding in a legend - position dependent text etc. The plot positions are
    # now fixed. A relative positioning would shift else.

    if bbox is None:
        bb = 'tight'
    else:
        bb = bbox

    # Saving of the figure and showing if necessary
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if save:
        if save_name == '':
            fig.savefig(Path(save_path) / (str(time.asctime()) + save_format), *args, dpi=dpi, bbox_inches=bb)
        else:
            fig.savefig(Path(save_path) / (save_name + save_format), *args, dpi=dpi, bbox_inches=bb)

        plt.close(fig=fig)


from matplotlib.patches import Arc
from matplotlib.transforms import Bbox, IdentityTransform, TransformedBbox

class AngleAnnotation(Arc):
    """
    Draws an arc between two vectors which appears circular in display space.
    """
    def __init__(self, xy, p1, p2, size=75, unit="points", ax=None,
                 text="", textposition="inside", text_kw=None, **kwargs):
        """
        Parameters
        ----------
        xy, p1, p2 : tuple or array of two floats
            Center position and two points. Angle annotation is drawn between
            the two vectors connecting *p1* and *p2* with *xy*, respectively.
            Units are data coordinates.

        size : float
            Diameter of the angle annotation in units specified by *unit*.

        unit : str
            One of the following strings to specify the unit of *size*:

            * "pixels": pixels
            * "points": points, use points instead of pixels to not have a
              dependence on the DPI
            * "axes width", "axes height": relative units of Axes width, height
            * "axes min", "axes max": minimum or maximum of relative Axes
              width, height

        ax : `matplotlib.axes.Axes`
            The Axes to add the angle annotation to.

        text : str
            The text to mark the angle with.

        textposition : {"inside", "outside", "edge"}
            Whether to show the text in- or outside the arc. "edge" can be used
            for custom positions anchored at the arc's edge.

        text_kw : dict
            Dictionary of arguments passed to the Annotation.

        **kwargs
            Further parameters are passed to `matplotlib.patches.Arc`. Use this
            to specify, color, linewidth etc. of the arc.

        """
        self.ax = ax or plt.gca()
        self._xydata = xy  # in data coordinates
        self.vec1 = p1
        self.vec2 = p2
        self.size = size
        self.unit = unit
        self.textposition = textposition

        super().__init__(self._xydata, size, size, angle=0.0,
                         theta1=self.theta1, theta2=self.theta2, **kwargs)

        self.set_transform(IdentityTransform())
        self.ax.add_patch(self)

        self.kw = dict(ha="center", va="center",
                       xycoords=IdentityTransform(),
                       xytext=(0, 0), textcoords="offset points",
                       annotation_clip=True)
        self.kw.update(text_kw or {})
        self.text = ax.annotate(text, xy=self._center, **self.kw)

    def get_size(self):
        factor = 1.
        if self.unit == "points":
            factor = self.ax.figure.dpi / 72.
        elif self.unit[:4] == "axes":
            b = TransformedBbox(Bbox.unit(), self.ax.transAxes)
            dic = {"max": max(b.width, b.height),
                   "min": min(b.width, b.height),
                   "width": b.width, "height": b.height}
            factor = dic[self.unit[5:]]
        return self.size * factor

    def set_size(self, size):
        self.size = size

    def get_center_in_pixels(self):
        """return center in pixels"""
        return self.ax.transData.transform(self._xydata)

    def set_center(self, xy):
        """set center in data coordinates"""
        self._xydata = xy

    def get_theta(self, vec):
        vec_in_pixels = self.ax.transData.transform(vec) - self._center
        return np.rad2deg(np.arctan2(vec_in_pixels[1], vec_in_pixels[0]))

    def get_theta1(self):
        return self.get_theta(self.vec1)

    def get_theta2(self):
        return self.get_theta(self.vec2)

    def set_theta(self, angle):
        pass

    # Redefine attributes of the Arc to always give values in pixel space
    _center = property(get_center_in_pixels, set_center)
    theta1 = property(get_theta1, set_theta)
    theta2 = property(get_theta2, set_theta)
    width = property(get_size, set_size)
    height = property(get_size, set_size)

    # The following two methods are needed to update the text position.
    def draw(self, renderer):
        self.update_text()
        super().draw(renderer)

    def update_text(self):
        c = self._center
        s = self.get_size()
        angle_span = (self.theta2 - self.theta1) % 360
        angle = np.deg2rad(self.theta1 + angle_span / 2)
        r = s / 2
        if self.textposition == "inside":
            r = s / np.interp(angle_span, [60, 90, 135, 180],
                                          [3.3, 3.5, 3.8, 4])
        self.text.xy = c + r * np.array([np.cos(angle), np.sin(angle)])
        if self.textposition == "outside":
            def R90(a, r, w, h):
                if a < np.arctan(h/2/(r+w/2)):
                    return np.sqrt((r+w/2)**2 + (np.tan(a)*(r+w/2))**2)
                else:
                    c = np.sqrt((w/2)**2+(h/2)**2)
                    T = np.arcsin(c * np.cos(np.pi/2 - a + np.arcsin(h/2/c))/r)
                    xy = r * np.array([np.cos(a + T), np.sin(a + T)])
                    xy += np.array([w/2, h/2])
                    return np.sqrt(np.sum(xy**2))

            def R(a, r, w, h):
                aa = (a % (np.pi/4))*((a % (np.pi/2)) <= np.pi/4) + \
                     (np.pi/4 - (a % (np.pi/4)))*((a % (np.pi/2)) >= np.pi/4)
                return R90(aa, r, *[w, h][::int(np.sign(np.cos(2*a)))])

            bbox = self.text.get_window_extent()
            X = R(angle, r, bbox.width, bbox.height)
            trans = self.ax.figure.dpi_scale_trans.inverted()
            offs = trans.transform(((X-s/2), 0))[0] * 72
            self.text.set_position([offs*np.cos(angle), offs*np.sin(angle)])


def arc_between_points(p1, p2, sagitta=0.05, ax=None, **kwargs):
    """
    Draw a circular arc between p1 and p2 with a given sagitta (arc height).

    sagitta > 0  → arc bulges left
    sagitta < 0  → arc bulges right
    """
    if ax is None:
        ax = plt.gca()

    x1, y1 = p1
    x2, y2 = p2

    dx, dy = x2 - x1, y2 - y1
    d = np.hypot(dx, dy)

    if abs(sagitta) < 1e-6:
        ax.plot([x1, x2], [y1, y2], **kwargs)
        return

    # radius from sagitta
    R = (d**2 / (8 * abs(sagitta))) + abs(sagitta) / 2

    # midpoint
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2

    # perpendicular unit vector
    ux, uy = -dy / d, dx / d

    # center of circle
    cx = mx + np.sign(sagitta) * (R - abs(sagitta)) * ux
    cy = my + np.sign(sagitta) * (R - abs(sagitta)) * uy

    # angles
    theta1 = np.degrees(np.arctan2(y1 - cy, x1 - cx))
    theta2 = np.degrees(np.arctan2(y2 - cy, x2 - cx))

    arc = Arc(
        (cx, cy),
        2 * R,
        2 * R,
        theta1=theta1,
        theta2=theta2,
        **kwargs
    )
    ax.add_patch(arc)

    return arc


def curved_line_through_points(p0, p1, p2, ax=None, n=100, **kwargs):
    """
    Draw a smooth curve through three points using a quadratic Bézier curve.

    p0, p1, p2 : (x, y)
    p1 is the control point (defines curvature)
    """
    if ax is None:
        ax = plt.gca()

    t = np.linspace(0, 1, n)[:, None]

    p0 = np.array(p0)
    p1 = np.array(p1)
    p2 = np.array(p2)

    curve = (
            (1 - t) ** 2 * p0 +
            2 * (1 - t) * t * p1 +
            t ** 2 * p2
    )

    ax.plot(curve[:, 0], curve[:, 1], **kwargs)
