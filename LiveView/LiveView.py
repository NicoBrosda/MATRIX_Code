import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from EvaluationSoftware.main import Analyzer
from EvaluationSoftware.readout_modules import design_readout
from pathlib import Path

analyzer = Analyzer((1, 128), 0.42, 0.08, readout=design_readout)

# creating the figure and axes object
fig, ax = plt.subplots()

folder_path = Path('/Users/nico_brosda/Desktop/NewMaps/')
save_name = 'Test'


# update function to update data and plot
def update(frame):
    ax.clear()  # clearing the axes
    analyzer.set_measurement(folder_path, save_name)
    analyzer.load_measurement(readout_module=design_readout)
    analyzer.create_map()
    if len(fig.axes) > 1:
        analyzer.plot_map(ax=ax, fig=fig, colorbar=False)
    else:
        analyzer.plot_map(ax=ax, fig=fig, colorbar=True)

    fig.canvas.draw()  # forcing the artist to redraw itself
    # plt.pause(1)


anim = FuncAnimation(fig, update)
plt.show()