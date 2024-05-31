import matplotlib.pyplot as plt
import scipy as sp

from Plot_Methods.plot_standards import *

length = 2  # length of one periode of the signal
level_base = 0  # noise
delta = 1  # s/n ration
phase = 0  # shift of the signal [0, 2*pi] or [0°, 360°]

t = np.linspace(0, 10, 1000)
fig, ax = plt.subplots()


def rectangle_wave(T, delta, base=0, phase=0):
    return base + delta * sp.signal.square(2 * np.pi / T * t + phase, duty=0.5)


class RectangleWave:
    def __init__(self, signal):
        self.signal = signal

    def find_periode(self):
        pass

    def find_phase(self):
        pass