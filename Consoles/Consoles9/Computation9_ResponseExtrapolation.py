import contextlib
import io

with contextlib.redirect_stdout(io.StringIO()):
    from Consoles.Consoles9.Console9_EnergyExtrapolation import *

def compute_response(proton_energy, loss_factor=None):
    simulated_res = np.interp(proton_energy, sim_energy, sim_res)
    if loss_factor is None:
        prop = c_200II_cut
    else:
        prop = c_200II_cut * (1/10.5 / c_200II_cut) / loss_factor
    return simulated_res * prop

def get_dark_current(ams_voltage):
    readout, position_parser, voltage_parser, current_parser = (
        lambda x, y: ams_2line_readout(x, y, channel_assignment=channel_assignment),
        standard_position,
        standard_voltage,
        current5
    )
    dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_260325/')

    dark= [f'exp1_dark_0nA_400um_nA_{ams_voltage:.1f}_x_20.0_y_68.0',
           f'exp64_darkEnd_0.5nA_400um_nA_{ams_voltage:.1f}_x_20.0_y_68.0',
           f'2exp66_Dark_0.0nA_0um_nA_{ams_voltage:.1f}_x_20.0_y_68.0',
           f'2exp138_DarkEnd_0nA_200um_nA_{ams_voltage:.1f}_x_20.0_y_68.0']

    A = Analyzer((2, 64), (0.4, 0.4), (0.1, 0.1), readout=readout,
                 diode_offset=[[0, - 0.25], np.zeros(64)], position_parser=position_parser,
                 voltage_parser=voltage_parser, current_parser=current_parser)

    with contextlib.redirect_stdout(io.StringIO()):
        A.set_dark_measurement(dark_path, dark)

    return A.signal_conversion(np.mean(A.dark)) * scale_dict[A.scale][0]


def required_protons_AMS(energy, xdark_current=2, loss_factor=7, output='current', ams_voltage=1.9):
    # Dark current in A
    dark_current = get_dark_current(ams_voltage)

    # Response of detector in #electrons per proton
    response = compute_response(energy, loss_factor)

    # Number of protons = current to reach / current response per proton
    n_p = xdark_current * dark_current / (response * e)
    if output == 'dose':
        return None
    elif output == 'current' :
        return n_p * e
    elif output == 'protons':
        return n_p
    else:
        raise ValueError('output must be "dose", "current" or "protons"')


for volt in np.arange(0.8, 2.1, 0.1):
    print(f'Dark current at {volt:.1f} V: {get_dark_current(ams_voltage=volt):.2e} A')

# print(required_protons_AMS(np.linspace(20, 230, 100), output='current', ams_voltage=1.9, loss_factor=None))

# ---------------- Plot 1: Extrapolation to higher energies (now vs energy) --------------------
energy_cmap = sns.color_palette("crest_r", as_cmap=True)
energy_colormapper = lambda energy: color_mapper(energy, np.min(sim_energy), np.max(sim_energy))
energy_color = lambda energy: energy_cmap(energy_colormapper(energy))

fig, ax = plt.subplots()

x = np.linspace(20, 230, 200)
dat = required_protons_AMS(x, xdark_current=2, output='current', ams_voltage=1.9, loss_factor=None) * 1e9
ax.plot(x, dat, c='k', ls='-')

ax.set_xlim(ax.get_xlim())
ax.set_ylim(ax.get_ylim())

text = f'Required beam current @ detector \n to measure {2} x dark current \n CYRCé conditions (loss = 7) \n AMS voltage = 1.9$\\,$V'
ax.text(*transform_axis_to_data_coordinates(ax, [0.03, 0.92]), text, color='k', ha='left', va='top',
        fontsize=13, bbox={'facecolor': 'w', 'alpha': 0.8, 'pad': 0, 'edgecolor': 'w'})
# '''
energies = [23.69003946750678, 60, 120, 226.7]
for i, energy in enumerate(energies):
    res = required_protons_AMS(energy, xdark_current=2, output='current', ams_voltage=1.9, loss_factor=None) * 1e9
    print(energy, res)
    if i == 0:
        text = f'{res: .1f}$\\,$nA for\nE ={energy: .2f}$\\,$MeV'
        ax.text(energy+4, res, text, color=energy_color(energy), ha='left', va='center', fontsize=13,
                bbox={'facecolor': 'w', 'alpha': 0.8, 'pad': 0, 'edgecolor': 'w'})
    elif i < len(energies)-1:
        text = f'{res: .1f}$\\,$nA for\nE ={energy: .2f}$\\,$MeV'
        ax.text(energy+4, res, text, color=energy_color(energy), ha='left', va='top', fontsize=13,
                bbox={'facecolor': 'w', 'alpha': 0.8, 'pad': 0, 'edgecolor': 'w'})
    else:
        text = f'{res: .1f}$\\,$nA for\nE ={energy: .2f}$\\,$MeV'
        ax.text(energy-2, res-0.02, text, color=energy_color(energy), ha='right', va='top', fontsize=13,
                bbox={'facecolor': 'w', 'alpha': 0.8, 'pad': 0, 'edgecolor': 'w'})

    ax.axvline(energy, ymin=0, ymax=transform_data_to_axis_coordinates(ax, [energy, res])[1], c=energy_color(energy), ls='--')
    ax.axhline(res, xmin=0, xmax=transform_data_to_axis_coordinates(ax, [energy, res])[0], c=energy_color(energy), ls='--')
# '''
ax.set_xlabel(f'Energy of incident proton (MeV)')
ax.set_ylabel(f'Required proton current (nA)')

format_save(results_path, f'RequiredCurrent_CYRCe', legend=False, axes=[ax])

# ---------------- Plot 2: Required number of protons / s --------------------
energy_cmap = sns.color_palette("crest_r", as_cmap=True)
energy_colormapper = lambda energy: color_mapper(energy, np.min(sim_energy), np.max(sim_energy))
energy_color = lambda energy: energy_cmap(energy_colormapper(energy))

fig, ax = plt.subplots()

x = np.linspace(20, 230, 200)
dat = required_protons_AMS(x, xdark_current=2, output='protons', ams_voltage=1.9, loss_factor=1.5) / 1e+9
print(dat)
ax.plot(x, dat, c='k', ls='-')

ax.set_xlim(ax.get_xlim())
ax.set_ylim(ax.get_ylim())

text = f'Required proton flux @ detector \n to measure {2} x dark current \n general conditions (loss = 1.5) \n AMS voltage = 1.9$\\,$V'
ax.text(*transform_axis_to_data_coordinates(ax, [0.03, 0.92]), text, color='k', ha='left', va='top',
        fontsize=13, bbox={'facecolor': 'w', 'alpha': 0.8, 'pad': 0, 'edgecolor': 'w'})
# '''
energies = [23.69003946750678, 60, 120, 226.7]
for i, energy in enumerate(energies):
    res = required_protons_AMS(energy, xdark_current=2, output='protons', ams_voltage=1.9, loss_factor=1.5) / 1e+9
    print(energy, res)
    if i == 0:
        text = f'{res: .1f} for\nE ={energy: .2f}$\\,$MeV'
        ax.text(energy+4, res, text, color=energy_color(energy), ha='left', va='center', fontsize=13,
                bbox={'facecolor': 'w', 'alpha': 0.8, 'pad': 0, 'edgecolor': 'w'})
    elif i < len(energies)-1:
        text = f'{res: .1f} for\nE ={energy: .2f}$\\,$MeV'
        ax.text(energy+4, res, text, color=energy_color(energy), ha='left', va='top', fontsize=13,
                bbox={'facecolor': 'w', 'alpha': 0.8, 'pad': 0, 'edgecolor': 'w'})
    else:
        text = f'{res: .1f} for\nE ={energy: .2f}$\\,$MeV'
        ax.text(energy-2, res-0.02, text, color=energy_color(energy), ha='right', va='top', fontsize=13,
                bbox={'facecolor': 'w', 'alpha': 0.8, 'pad': 0, 'edgecolor': 'w'})

    ax.axvline(energy, ymin=0, ymax=transform_data_to_axis_coordinates(ax, [energy, res])[1], c=energy_color(energy), ls='--')
    ax.axhline(res, xmin=0, xmax=transform_data_to_axis_coordinates(ax, [energy, res])[0], c=energy_color(energy), ls='--')
# '''

ax.set_xlabel(f'Energy of incident proton (MeV)')
ax.set_ylabel(f'Required proton flux (10$^9$ s$^{'{-1}'}$)')

format_save(results_path, f'RequiredProtons_General', legend=False, axes=[ax])