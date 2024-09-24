import os

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from EvaluationSoftware.main import Analyzer
from EvaluationSoftware.readout_modules import design_readout, ams_channel_assignment_readout
from EvaluationSoftware.position_parsing_modules import standard_position
from pathlib import Path
from copy import deepcopy
import time
from matplotlib import ticker
import numpy as np


channel_assignment = {}
j = 0
for i in range(1, 129):
    if i <= 32:
        channel_assignment[str(i)] = 2*j+1
    elif i <= 64:
        channel_assignment[str(i)] = 64 - 2*j
    elif i <= 96:
        channel_assignment[str(i)] = 65 + 2*j
    elif i <= 128:
        channel_assignment[str(i)] = 128 - 2*j
    j += 1
    if i == 32 or i == 64 or i == 96:
        j = 0
    channel_assignment[str(i+1)] = i+1

readout, position_parser = lambda x, y, z: ams_channel_assignment_readout(x, y, z, channel_assignment=channel_assignment), standard_position
analyzer = Analyzer((1, 128), 0.42, 0.08, readout=readout)

folder_path = Path('./TestMap/')
save_name = 'Test'

# creating the figure and axes object
fig, ax = plt.subplots()

analyzer.set_measurement(folder_path, save_name)
if len(analyzer.measurement_files) > 0:
    analyzer.load_measurement(readout_module=readout)
    analyzer.create_map()
    analyzer.plot_map(ax=ax, fig=fig, colorbar=False)
else:
    ax.text(0.1, 0.5,
            'No data files found in folder:\n\n' + str(folder_path) + '\n\n under search criterion: ' + str(save_name),
            fontsize=20)
files_before_update = deepcopy(analyzer.measurement_files)


# update function to update data and plot
def update(frame):
    global files_before_update

    # start = time.time()
    ax.clear()  # clearing the axes

    analyzer.set_measurement(folder_path, save_name)

    if len(analyzer.measurement_files) > 1:
        if len(files_before_update) > len(analyzer.measurement_files):
            analyzer.load_measurement(readout_module=readout)
        elif len(files_before_update) < len(analyzer.measurement_files):
            for file in analyzer.measurement_files:
                if file not in files_before_update:
                    pos = position_parser(file)
                    cache = readout(file, analyzer)
                    cache['signal'] = cache['signal'] * analyzer.norm_factor
                    cache.update({'position': pos})
                    analyzer.measurement_data.append(cache)
        analyzer.create_map()
        if len(fig.axes) > 1:
            analyzer.plot_map(ax=ax, fig=fig, colorbar=False)
        else:
            analyzer.plot_map(ax=ax, fig=fig, colorbar=True)
        intensity_limits = [0, np.abs(np.max(analyzer.map['z']) * 0.9)]
        ticklabels = np.linspace(intensity_limits[0], intensity_limits[1], 7)
        ticks = np.linspace(*fig.axes[1].get_ylim(), 7)

        fig.axes[1].set_yticks(ticks)
        fig.axes[1].set_yticklabels([str(round(i, 1)) for i in ticklabels])
        fig.axes[1].yaxis.set_minor_locator(ticker.AutoMinorLocator())
        files_before_update = deepcopy(analyzer.measurement_files)
    else:
        ax.text(0.1, 0.5, 'No data files found in folder:\n\n'+str(folder_path)+'\n\n under search criterion: '
                + str(save_name), fontsize=20)

    fig.canvas.draw()  # forcing the artist to redraw itself
    # plt.pause(1)
    # end = time.time()
    # print(end - start)


anim = FuncAnimation(fig, update, cache_frame_data=False)

plt.show()

