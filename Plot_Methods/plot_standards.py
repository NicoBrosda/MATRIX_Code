# Some imports of necessary packages and definitions of standard name spaces
import numpy as np  # v. 1.25.1
import matplotlib.pyplot as plt  # Used in this script is matplotlib 3.7.2 - newer versions should also be no problem
import matplotlib as mpl
from cycler import cycler
import locale
from matplotlib import ticker, cm
import os
import time
from matplotlib.patches import Rectangle
import warnings
import seaborn as sns  # v. 0.12.2 (seaborn isn't necessarily required, but offers nice colour schemes)
from pathlib import Path
from Plot_Methods.helper_functions import *
from Plot_Methods.label_standard import *
import matplotlib
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm

# Standard save format for plots
save_format = '.png'

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
halfsize_plot = (latex_textwidth/2, latex_textwidth/1.3/2)
totalpage_plot = (14.6979*cm, 20*cm)
totalpage_plot_side = (20*cm, 14.6979*cm)

# This is e.g. used for plots that take 45% of the whole documents side width of ca. 21cm. Though, letting plots be
# wider than the text_width is not advised if you're not 100% sure, that there are no complications with printing the
# document.
biggerboxes_plot = (20.99701*cm*0.45, 20.99701*cm*0.45/1.3)

# Now one can also define standard text sizes for the plots and assign where to use which size. The standards below are
# reasonable choices for a DIN A4 document - you should nevertheless check if these values really fit your document
SMALL_SIZE = 8
MEDIUM_SIZE = 9
BIGGER_SIZE = 11

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
if use_LaTeX:
    plt.rcParams.update({'mathtext.default':  'regular'})  # No cursive text in math environment!


# This defines a simple standard layout, all plots at our chair (and in general) should fulfill:
def afp_layout(axes=None, *arg, **kwarg):
    # Here it is important to hand axes if they deviate from the currently used standard axes of a plot
    if axes is None:
        axes = plt.gcf().get_axes()
    # Boxed, ticks inside the axes, no grid lines in plot
    for ax in axes:
        ax.grid(which='both', visible=False)
        ax.tick_params(which='both', direction='in')
    if len(axes) == 1:
        axes[0].tick_params(which='both', direction='in', top=True, right=True, left=True, bottom=True)
    elif len(axes) == 2:
        axes[0].tick_params(which='both', direction='in', top=True, right=False, left=True, bottom=True)
        axes[1].tick_params(which='both', direction='in', top=True, right=True, left=False, bottom=True)


# I wrapped the afp_layout in another functions that allows deviations from the layout without touching the standards
# from our chair
def standard_layout(axes=None, *args, **kwargs):
    if axes is None:
        axes = plt.gcf().get_axes()

    afp_layout(axes, *args, **kwargs)


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

# '''
def gradient_arrow(ax, start, end, cmap="viridis", n=50, lw=3):
    cmap = plt.get_cmap(cmap, n)
    # Arrow shaft: LineCollection
    x = np.linspace(start[0], end[0], n)
    y = np.linspace(start[1], end[1], n)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, linewidth=lw)
    lc.set_array(np.linspace(0, 1, n))
    ax.add_collection(lc)
    # Arrow head: Triangle
    tricoords = [(0, -0.4), (0.5, 0), (0, 0.4), (0, -0.4)]
    angle = np.arctan2(end[1]-start[1], end[0]-start[0])
    rot = matplotlib.transforms.Affine2D().rotate(angle)
    tricoords2 = rot.transform(tricoords)
    tri = matplotlib.path.Path(tricoords2, closed=True)
    ax.scatter(end[0], end[1], c=1, s=(2*lw)**2, marker=tri, cmap=cmap,vmin=0)
    ax.autoscale_view()
# '''

# Now to the interesting part: This function represents an example how to apply settings to a generated plot, without
# the necessity to call the settings during plotting.
def format_save(save_path=Path('./Plots/'), save_name='', save=True, legend=True, legend_separator=None, x_after=0,
                y_after=0, x_fmt=format_func, y_fmt=format_func, second_axis=False, minor_xticks=True,
                minor_yticks=True, english=language_english, legend_position=0, plot_size=fullsize_plot, x_rotation=0,
                plot_variant='PL_spectrum', format=save_format, *args) -> None:
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
    axes = plt.gcf().get_axes()
    ax = plt.gca()
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
        ax.yaxis.set_major_locator(ticker.AutoLocator())
        if minor_yticks:
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        if no_colorbar[k]:
            ax.xaxis.set_major_locator(ticker.AutoLocator())
            if minor_xticks:
                ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

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

    plt.draw()  # If we want to save the figure we first need to draw it (without showing it)
    plt.pause(0.0001)  # This is an easy bug-fix, where matplotlib is not updating the current figure after drawing

    # Here post-processing is possible, like adding in a legend - position dependent text etc. The plot positions are
    # now fixed. A relative positioning would shift else.

    # Saving of the figure and showing if necessary
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if save:
        if save_name == '':
            fig.savefig(Path(save_path) / (str(time.asctime()) + format), dpi=300, bbox_inches="tight", *args)
        else:
            fig.savefig(Path(save_path) / (save_name + format), dpi=300, bbox_inches="tight", *args)

        plt.close()


def just_save(save_path=Path('./Plots/'), save_name='', save=True, legend=True, legend_separator=None, x_after=0,
                y_after=0, x_fmt=format_func, y_fmt=format_func, second_axis=False, minor_xticks=True,
                minor_yticks=True, english=language_english, legend_position=0, plot_size=fullsize_plot, x_rotation=0,
                plot_variant='PL_spectrum', format=save_format, *args) -> None:
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
    axes = plt.gcf().get_axes()
    ax = plt.gca()
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

    plt.draw()  # If we want to save the figure we first need to draw it (without showing it)
    plt.pause(0.0001)  # This is an easy bug-fix, where matplotlib is not updating the current figure after drawing

    # Here post-processing is possible, like adding in a legend - position dependent text etc. The plot positions are
    # now fixed. A relative positioning would shift else.

    # Saving of the figure and showing if necessary
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if save:
        if save_name == '':
            fig.savefig(Path(save_path) / (str(time.asctime()) + format), dpi=300, bbox_inches="tight", *args)
        else:
            fig.savefig(Path(save_path) / (save_name + format), dpi=300, bbox_inches="tight", *args)

        plt.close()