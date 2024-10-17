import numpy as np
import scipy.signal
from scipy.signal import find_peaks

from EvaluationSoftware.main import *
from EvaluationSoftware.helper_modules import LineShape, mean_diodes

mapping = Path('../Files/mapping.xlsx')
data = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:])-1 for k in data['direction_2']]

readout, position_parser = lambda x, y: ams_channel_assignment_readout(x, y, channel_assignment=channel_assignment), standard_position

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_111024/')
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Results_111024/')
dark_path = Path('/Users/nico_brosda//Cyrce_Messungen/matrix_111024/')
norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_230924/')

dark_paths_array1 = ['Dark_QuickYScan_0_um_2_nA_.csv']

norm = ['uniformity_scan_']

# Data imported from .txt file
data_wheel = pd.read_csv('../Files/energies_after_wheel_diffusor200.txt', sep='\t', header=4,
                         names=['position', 'thickness', 'energie'])

'''
# Full aperture:
crit = 'Start_QuickScan'
A = Analyzer((1, 128), 0.4, 0.1, readout=readout)
A.set_measurement(folder_path, crit)
A.set_dark_measurement(dark_path, dark_paths_array1)
# Normalization - correct assignment
A.normalization(norm_path, norm, normalization_module=normalization_from_translated_array_v2)
A.load_measurement(readout_module=readout)
A.create_map(inverse=[True, False])
signal = A.get_signal_yline()
pos_y = A.maps[0]['y']

fig, ax = plt.subplots()
ax.plot(pos_y, signal, c='k', marker='x', lw=1.5, label=r'Full aperture with wheel')
# Parameter to define the middle of the geometry (in data coordinates)
# The round aperture with wheel had a size of 24 mm, we positioned the wedge so that we used the 24 mm
# The array has a length of 64 mm, the midddle is in my conversion 70 + 32 = 102 mm
middle = 20
print('PEEK material wedge from ', middle - 20, ' mm until ', middle + 20, ' mm')

shape = LineShape([[0, 10], [40, 0]], distance_mode=True)
shape.print_shape()
# shape.mirror()
shape.position(102, 16)
shape.add_to_plot(0.0, 0.5, color='grey', alpha=0.7, zorder=-1, edgecolor='k')
ax.set_title(r'Full aperture with wheel')
format_save(results_path / 'Bragg/', crit, legend=False)
plt.close('all')
'''

energy_cmap = sns.color_palette("crest_r", as_cmap=True)
energy_colormapper = lambda energy: color_mapper(energy, np.min(data_wheel['energie']), np.max(data_wheel['energie']))
energy_color = lambda energy: energy_cmap(energy_colormapper(energy))

data_list = []
thickness_list = []
for i in range(20):
    crit = 'BraggP{number:02d}_'.format(number=i+1)
    print('-' * 50)
    print(crit)
    print('-' * 50)

    A = Analyzer((1, 128), 0.4, 0.1, readout=readout)

    A.set_measurement(folder_path, crit)

    A.set_dark_measurement(dark_path, dark_paths_array1)

    # Normalization - correct assignment
    A.normalization(norm_path, norm, normalization_module=normalization_from_translated_array_v2)

    A.load_measurement(readout_module=readout)

    A.create_map(inverse=[True, False])
    signal = A.get_signal_yline()
    pos_y = A.maps[0]['y']

    fig, ax = plt.subplots()
    ax.plot(pos_y, signal, c=energy_color(data_wheel['energie'][i]), marker='x', lw=1.5,
            label=r'Wheel position {pos} - Al thickness {thick:.3f}$\,$mm - proton energy {energy: .3f}$\,$MeV'
            .format(pos=data_wheel['position'][i], thick=data_wheel['thickness'][i], energy=data_wheel['energie'][i]))
    # Parameter to define the middle of the geometry (in data coordinates)
    middle = 20
    print('PEEK material wedge from ', middle - 20, ' mm until ', middle + 20, ' mm')

    shape = LineShape([[0, 10], [40, 0]], distance_mode=True)
    shape.print_shape()
    # shape.mirror()
    shape.position(110.05150214592274, 40)
    shape.add_to_plot(0.0, 0.5, color='grey', alpha=0.7, zorder=-1, edgecolor='k')
    ax.set_title(r'Wheel position {pos} - Al thickness {thick:.3f}$\,$mm - proton energy {energy: .3f}$\,$MeV'
                 .format(pos=data_wheel['position'][i], thick=data_wheel['thickness'][i], energy=data_wheel['energie'][i]))
    ax.set_xlabel(r'Calculated real position of diodes (mm)')
    ax.set_ylabel(r'Signal Amplitude')
    format_save(results_path / 'Bragg/', crit, legend=False)
    plt.close('all')

    data_list.append([crit, pos_y, signal])

    # Find peak in the signal and select the highest
    peak = np.argmax(signal)

    fig, ax = plt.subplots()
    ax.plot(pos_y, signal, c=energy_color(data_wheel['energie'][i]), marker='x', lw=1.5)
    # Parameter to define the middle of the geometry (in data coordinates)
    middle = 20
    print('PEEK material wedge from ', middle - 20, ' mm until ', middle + 20, ' mm')

    shape = LineShape([[0, 10], [40, 0]], distance_mode=True)
    shape.print_shape()
    # shape.mirror()
    shape.position(110.05150214592274, 40)
    shape.add_to_plot(0.0, 0.5, color='grey', alpha=0.7, zorder=-1, edgecolor='k')
    ax.axvline(pos_y[peak], label='Material thickness = {thickness:.3f}$\,$mm'.format(thickness=shape.calculate_value(pos_y[peak])))
    ax.set_title(r'Wheel position {pos} - Al thickness {thick:.3f}$\,$mm - proton energy {energy: .3f}$\,$MeV'
                 .format(pos=data_wheel['position'][i], thick=data_wheel['thickness'][i],
                         energy=data_wheel['energie'][i]))
    ax.set_xlabel(r'Calculated real position of diodes (mm)')
    ax.set_ylabel(r'Signal Amplitude')
    format_save(results_path / 'Bragg/', crit+'_thickness_', legend=True)
    plt.close('all')
    thickness_list.append([data_wheel['energie'][i], shape.calculate_value(pos_y[peak])])

fig, ax = plt.subplots()
for i, point in enumerate(data_list):
    pos_y = point[1]
    signal = point[2]
    ax.plot(pos_y, signal, c=energy_color(data_wheel['energie'][i]), marker='', lw=1.5,
            label=r'Wheel position {pos} - Al thickness {thick:.3f}$\,$mm - proton energy {energy: .3f}$\,$MeV'
            .format(pos=data_wheel['position'][i], thick=data_wheel['thickness'][i], energy=data_wheel['energie'][i]))
# Parameter to define the middle of the geometry (in data coordinates)
middle = 20
print('PEEK material wedge from ', middle - 20, ' mm until ', middle + 20, ' mm')

shape = LineShape([[0, 10], [40, 0]], distance_mode=True)
shape.print_shape()
# shape.mirror()
# 0 of Bragg shape at 114 - 3.9484978540772535 = 110.05150214592274 mm
shape.position(110.05150214592274, 40)
shape.add_to_plot(0.0, 0.5, color='grey', alpha=0.7, zorder=-1, edgecolor='k')
ax.set_xlabel(r'Calculated real position of diodes (mm)')
ax.set_ylabel(r'Signal Amplitude')
format_save(results_path / 'Bragg/', 'All', legend=False)
plt.close('all')

fig, ax = plt.subplots()
for i, point in enumerate(thickness_list):
    ax.plot(point[0], point[1], c=energy_color(data_wheel['energie'][i]), marker='x')
ax.set_xlabel(r'Proton energy (MeV)')
ax.set_ylabel(r'PEEK Material thickness with maximal signal (mm)')
format_save(results_path / 'Bragg/', 'PEEK_thickness', legend=False)
plt.close('all')