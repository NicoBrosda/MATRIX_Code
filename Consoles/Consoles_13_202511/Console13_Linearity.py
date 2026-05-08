before = 'exp1_25MeV_at_Control_'
after = 'exp5_25MeV_at_Control_'
mid = 'exp3_25MeV_at_Control_'

from EvaluationSoftware.standard_processes import *
from EvaluationSoftware.parameter_parsing_modules import *

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_171225/')
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/ResultsPEEK_161225/')

save_format = '.png'

mapping = Path('../../Files/mapping.xlsx')
data = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:])-1 for k in data['direction_2']]
readout, position_parser, voltage_parser, current_parser = (
    lambda x, y: ams_2line_fast_avg(x, y, channel_assignment=channel_assignment),
    standard_position,
    set_voltage,
    current6
)
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

# linearity(folder_path, results_path / 'linearity_before/', before, dark, A)
# linearity(folder_path, results_path / 'linearity_after/', after, dark, A)

currents, fit_currents, signal, fit, std, fit_std, fit_r2, std_r2 = linearity_return(folder_path, before, dark, A)
currents2, fit_currents2, signal2, fit2, std2, fit_std2, fit_r22, std_r22 = linearity_return(folder_path, after, dark, A)
currents3, fit_currents3, signal3, fit3, std3, fit_std3, fit_r23, std_r23 = linearity_return(folder_path, mid, dark, A)

currents = currents / (np.pi * (30e-1)**2) * 1e3
fit_currents = fit_currents / (np.pi * (30e-1)**2) * 1e3
currents2 = currents2 / (np.pi * (30e-1)**2) * 1e3
fit_currents2 = fit_currents2 / (np.pi * (30e-1)**2) * 1e3
currents3 = currents3 / (np.pi * (30e-1)**2) * 1e3
fit_currents3 = fit_currents3 / (np.pi * (30e-1)**2) * 1e3

linearity_colour = sns.color_palette("husl", 8)

fig, ax = plt.subplots()
ax.plot(currents, signal, marker='x', color='k', label='Signal', ls='-')
ax.plot(fit_currents, fit, marker='', color=linearity_colour[-1], label='Linear fit', ls='--')
ax.plot(currents, std, marker='o', color='k', label='Signal Std', ls='-')
ax.plot(fit_currents, fit_std, marker='', color=linearity_colour[-2], label='Std sqrt fit', ls='--')
ax.set_xlabel('Proton current density (pA$\\,$cm$^{-2}$)')
ax.set_ylabel(f'Signal current ({scale_dict[A.scale][1]}A)')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim([1e+1, 1.7e+3])
ax.set_ylim(ax.get_ylim()[0]/3, ax.get_ylim()[1])
ax.text(1.1e+1, signal[2], r'Signal linear fit \\ $\bar{\mathrm{R}}^2$' + ' = {x:.5f}'.format(x=fit_r2), fontsize=12,
        ha='left', va='bottom', c=linearity_colour[-1])
ax.text(1.1e+1, std[2], r'Std sqrt fit \\ $\bar{\mathrm{R}}^2$' + ' = {x:.3f}'.format(x=std_r2), fontsize=12,
        ha='left', va='bottom', c=linearity_colour[-2])
leg = ax.legend(
loc='lower right',
bbox_to_anchor=(1.01, 0),  # 50% across, 50% up inside the axes
bbox_transform=ax.transAxes,
frameon=False,
fontsize=12,
)
ax.text(*transform_axis_to_data_coordinates(ax, [0.03, 0.97]), r'\textbf{(before)}', fontsize=14, ha='left',
        va='top', color='k')
format_save(save_path=results_path, save_name=f"LinearityBefore", dpi=300, save_format=save_format, fig=fig, legend=False)

fig, ax = plt.subplots()
ax.plot(currents3, signal3, marker='x', color='k', label='Signal', ls='-')
ax.plot(fit_currents3, fit3, marker='', color=linearity_colour[-1], label='Linear fit', ls='--')
ax.plot(currents3, std3, marker='o', color='k', label='Signal Std', ls='-')
ax.plot(fit_currents3, fit_std3, marker='', color=linearity_colour[-2], label='Std sqrt fit', ls='--')
ax.set_xlabel('Proton current density (pA$\\,$cm$^{-2}$)')
ax.set_ylabel(f'Signal current ({scale_dict[A.scale][1]}A)')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim([1e+1, 1.7e+3])
ax.set_ylim(ax.get_ylim()[0]/3, ax.get_ylim()[1])
ax.text(1.1e+1, signal3[2], r'Signal linear fit \\ $\bar{\mathrm{R}}^2$' + ' = {x:.5f}'.format(x=fit_r23), fontsize=12,
        ha='left', va='bottom', c=linearity_colour[-1])
ax.text(1.1e+1, std3[2], r'Std sqrt fit \\ $\bar{\mathrm{R}}^2$' + ' = {x:.3f}'.format(x=std_r23), fontsize=12,
        ha='left', va='bottom', c=linearity_colour[-2])
leg = ax.legend(
loc='lower right',
bbox_to_anchor=(1.01, 0),  # 50% across, 50% up inside the axes
bbox_transform=ax.transAxes,
frameon=False,
fontsize=12,
)
ax.text(*transform_axis_to_data_coordinates(ax, [0.03, 0.97]), r'\textbf{(mid)}', fontsize=14, ha='left',
        va='top', color='k')
format_save(save_path=results_path, save_name=f"LinearityMid", dpi=300, save_format=save_format, fig=fig, legend=False)

fig, ax = plt.subplots()
ax.plot(currents2, signal2, marker='x', color='k', label='Signal', ls='-')
ax.plot(fit_currents2, fit2, marker='', color=linearity_colour[-1], label='Linear fit', ls='--')
ax.plot(currents2, std2, marker='o', color='k', label='Signal Std', ls='-')
ax.plot(fit_currents2, fit_std2, marker='', color=linearity_colour[-2], label='Std sqrt fit', ls='--')
ax.set_xlabel('Proton current density (pA$\\,$cm$^{-2}$)')
ax.set_ylabel(f'Signal current ({scale_dict[A.scale][1]}A)')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim([1e+1, 1.7e+3])
ax.set_ylim(ax.get_ylim()[0]/3, ax.get_ylim()[1])
ax.text(1.1e+1, signal2[2], r'Signal linear fit \\ $\bar{\mathrm{R}}^2$' + ' = {x:.5f}'.format(x=fit_r22), fontsize=12,
        ha='left', va='bottom', c=linearity_colour[-1])
ax.text(1.1e+1, std2[2], r'Std sqrt fit \\ $\bar{\mathrm{R}}^2$' + ' = {x:.3f}'.format(x=std_r22), fontsize=12,
        ha='left', va='bottom', c=linearity_colour[-2])
leg = ax.legend(
loc='lower right',
bbox_to_anchor=(1.01, 0),  # 50% across, 50% up inside the axes
bbox_transform=ax.transAxes,
frameon=False,
fontsize=12,
)
ax.text(*transform_axis_to_data_coordinates(ax, [0.03, 0.97]), r'\textbf{(after)}', fontsize=14, ha='left',
        va='top', color='k')
format_save(save_path=results_path, save_name=f"LinearityAfter", dpi=300, save_format=save_format, fig=fig, legend=False)


fig, ax = plt.subplots()
ax.plot(currents, signal, marker='x', color='b', label='Before', ls='')
ax.plot(fit_currents, fit, marker='', color='b', ls='--')
ax.plot(currents2, signal2, marker='x', color='r', label='After', ls='')
ax.plot(fit_currents2, fit2, marker='', color='r', ls='--')
ax.set_xlabel('Proton current density (pA$\\,$cm$^{-2}$)')
ax.set_ylabel(f'Signal current ({scale_dict[A.scale][1]}A)')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim([0.9e+1, 1.7e+3])
ax.set_ylim(0.5e-2, ax.get_ylim()[1] * 2)
ax.text(*transform_axis_to_data_coordinates(ax, [0.98, 0.11]),
        r'Before $\bar{\mathrm{R}}^2$' + ' = {x:.5f}'.format(x=fit_r2), fontsize=12,
        ha='right', va='bottom', c='b')
ax.text(*transform_axis_to_data_coordinates(ax, [0.98, 0.1]),
        r'After $\bar{\mathrm{R}}^2$' + ' = {x:.5f}'.format(x=fit_r22), fontsize=12,
        ha='right', va='top', c='r')
ax.legend(loc='upper left', fontsize=12)
format_save(save_path=results_path, save_name=f"LinearityBeforeAfter", dpi=300, save_format=save_format, fig=fig, legend=False)

fig, ax = plt.subplots()
ax.plot(currents, signal, marker='x', color='b', label='Before', ls='')
ax.plot(fit_currents, fit, marker='', color='b', ls='--')
ax.plot(currents2, signal2, marker='x', color='r', label='After', ls='')
ax.plot(fit_currents2, fit2, marker='', color='r', ls='--')
ax.plot(currents3, signal3, marker='x', color='gold', label='Mid', ls='')
ax.plot(fit_currents3, fit3, marker='', color='gold', ls='--')
ax.set_xlabel('Proton current density (pA$\\,$cm$^{-2}$)')
ax.set_ylabel(f'Signal current ({scale_dict[A.scale][1]}A)')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim([0.9e+1, 1.7e+3])
ax.set_ylim(0.5e-2, ax.get_ylim()[1] * 2)
ax.text(*transform_axis_to_data_coordinates(ax, [0.98, 0.2]),
        r'Before $\bar{\mathrm{R}}^2$' + ' = {x:.5f}'.format(x=fit_r2), fontsize=12,
        ha='right', va='top', c='b')
ax.text(*transform_axis_to_data_coordinates(ax, [0.98, 0.15]),
        r'After $\bar{\mathrm{R}}^2$' + ' = {x:.5f}'.format(x=fit_r23), fontsize=12,
        ha='right', va='top', c='y')
ax.text(*transform_axis_to_data_coordinates(ax, [0.98, 0.1]),
        r'After $\bar{\mathrm{R}}^2$' + ' = {x:.5f}'.format(x=fit_r22), fontsize=12,
        ha='right', va='top', c='r')
ax.legend(loc='upper left', fontsize=12)
format_save(save_path=results_path, save_name=f"LinearityBeforeMidAfter", dpi=300, save_format=save_format, fig=fig, legend=False)

# ---------------------------------------------------------------------
# Linearity per diode:
# ---------------------------------------------------------------------
from Console13_DosePerDiode import *

def reorder_diodes(a):
    """
    Interleave diode axis (size 128):
    [0..63, 64..127] → [64,0,65,1,...]
    Works for:
    - (128,)
    - (N, 128)
    - (128, M)
    - (N, 128, M), etc.
    Leaves arrays without a 128-axis unchanged.
    """
    if not hasattr(a, "shape"):
        return a
    for axis, size in enumerate(a.shape):
        if size == 128:
            if size % 2 != 0:
                raise ValueError(f"Diode axis not even: {a.shape}")
            # --- 1D special case ---
            if a.ndim == 1:
                return (
                    a.reshape(2, -1)[::-1]
                     .T
                     .reshape(-1)
                )
            # --- general case ---
            a_swapped = np.moveaxis(a, axis, 0)  # bring diode axis to front
            a_reordered = (
                a_swapped.reshape(2, -1, *a_swapped.shape[1:])
                         [::-1, ...]
                         .transpose(1, 0, *range(2, a_swapped.ndim + 1))
                         .reshape(128, *a_swapped.shape[1:])
            )
            return np.moveaxis(a_reordered, 0, axis)
    return a


def process_linearity(path, dataset):
    currents, fit_currents, *rest = linearity_return(
        path, dataset, dark, A, per_diode=True
    )
    rest = [reorder_diodes(np.array(a)) for a in rest]
    return (currents, fit_currents, *rest)

'''
currents, fit_currents, signal, fit, std, fit_std, fit_r2, std_r2 = linearity_return(folder_path, before, dark, A, per_diode=True)
currents2, fit_currents2, signal2, fit2, std2, fit_std2, fit_r22, std_r22 = linearity_return(folder_path, after, dark, A, per_diode=True)
currents3, fit_currents3, signal3, fit3, std3, fit_std3, fit_r23, std_r23 = linearity_return(folder_path, mid, dark, A, per_diode=True)
print(np.shape(currents), np.shape(fit_currents), np.shape(signal), np.shape(fit), np.shape(std), np.shape(fit_std), np.shape(fit_r2), np.shape(std_r2))
# '''

(currents,  fit_currents,  signal,  fit,  std,  fit_std,  fit_r2,  std_r2)  = process_linearity(folder_path, before)
(currents2, fit_currents2, signal2, fit2, std2, fit_std2, fit_r22, std_r22) = process_linearity(folder_path, after)
(currents3, fit_currents3, signal3, fit3, std3, fit_std3, fit_r23, std_r23) = process_linearity(folder_path, mid)
print(np.shape(currents), np.shape(fit_currents), np.shape(signal), np.shape(fit), np.shape(std), np.shape(fit_std), np.shape(fit_r2), np.shape(std_r2))

currents = currents / (np.pi * (30e-1)**2) * 1e3
fit_currents = fit_currents / (np.pi * (30e-1)**2) * 1e3
currents2 = currents2 / (np.pi * (30e-1)**2) * 1e3
fit_currents2 = fit_currents2 / (np.pi * (30e-1)**2) * 1e3
currents3 = currents3 / (np.pi * (30e-1)**2) * 1e3
fit_currents3 = fit_currents3 / (np.pi * (30e-1)**2) * 1e3

'''
plt.rcParams.update({
    "text.usetex": False,
})
plt.rcParams["font.family"] = ["Arial"]

# --- Plots linearity per diode ---
for i in range(128):
    # break
    position = 25-16.5+i*0.25
    th = shape.calculate_value(position)
    sim_pixel = int(position/0.25)
    dose_per_diode = proton_count * np.mean(dose_norm[sim_pixel - 2:sim_pixel + 2, 98:102]) * np.mean(
        proton_density_distribution[sim_pixel, 98:102]) / 1e+6
    fig, ax = plt.subplots()
    ax.plot(currents, signal[:,i], marker='x', color=diode_colour(i / 128), ls='', alpha=1)
    ax.plot(fit_currents, fit[i], marker='', color='b', ls='--', label='Before', alpha=1)
    # ax.plot(currents, std[:,i], marker='o', color=diode_colour(i / 128), ls='', alpha=0.5)
    # ax.plot(fit_currents, fit_std[i], marker='', color='r', ls=':', alpha=0.5)

    ax.plot(currents3, signal3[:, i], marker='^', color=diode_colour(i / 128), ls='', alpha=1)
    ax.plot(fit_currents3, fit3[i], marker='', color='gold', ls='--', label='Mid', alpha=1)

    ax.plot(currents2, signal2[:,i], marker='v', color=diode_colour(i / 128), ls='', alpha=1)
    ax.plot(fit_currents2, fit2[i], marker='', color='r', ls='--', label='After', alpha=1)
    # ax.plot(currents2, std2[:,i], marker='o', color=diode_colour(i / 128), ls='', alpha=0.5)
    # ax.plot(fit_currents2, fit_std2[i], marker='', color='b', ls=':', alpha=0.5)
    ax.set_xlim([0.9e+1, 1.7e+3])
    ax.set_xlabel('Proton current density (pA$\\,$cm$^{-2}$)')
    ax.set_ylabel(f'Signal current ({scale_dict[A.scale][1]}A)')
    ax.set_yscale('log'), ax.set_xscale('log')
    ax.text(*transform_axis_to_data_coordinates(ax, [0.98, 0.2]),
            r'Before $\bar{\mathrm{R}}^2$' + ' = {x:.5f}'.format(x=fit_r2[i]), fontsize=12,
            ha='right', va='top', c='b')
    ax.text(*transform_axis_to_data_coordinates(ax, [0.98, 0.15]),
            r'Mid $\bar{\mathrm{R}}^2$' + ' = {x:.5f}'.format(x=fit_r23[i]), fontsize=12,
            ha='right', va='top', c='y')
    ax.text(*transform_axis_to_data_coordinates(ax, [0.98, 0.1]),
            r'After $\bar{\mathrm{R}}^2$' + ' = {x:.5f}'.format(x=fit_r22[i]), fontsize=12,
            ha='right', va='top', c='r')
    ax.set_title(f'Diode {i+1} ({dose_per_diode[-1]:.2f} MGy), E_mean: {np.mean(median_energy[sim_pixel, 98:102]):.2f} MeV')
    format_save(save_path=results_path/'LinearityPerDiode/', save_name=f"Linearity_Diode{i}", dpi=300, save_format=save_format, fig=fig,
                legend=False)

plt.rcParams.update({
    "text.usetex": True,
})
plt.rcParams["font.family"] = ["Latin Modern Roman"]
'''

# --- Plot slope of linearity vs diode ---
fig, ax = plt.subplots()

ax.plot(fit[:, -1] * 1e+3 / fit_currents[-1], marker='x', color='b', ls='-', label='Before', alpha=1)
ax.plot(fit3[:, -1] * 1e+3 / fit_currents3[-1], marker='x', color='y', ls='-', label='Mid', alpha=1)
ax.plot(fit2[:, -1] * 1e+3 / fit_currents2[-1], marker='x', color='r', ls='-', label='After', alpha=1)

ax.set_xlabel('Array Diode')
ax.set_ylabel(f'Linearity slope (cm$^{-2}$)')
format_save(save_path=results_path, save_name=f"LinearitySlope_vs_Diode", dpi=300, save_format=save_format, fig=fig,
            legend=True)

# --- Plot slope of linearity vs energy ---
fig, ax = plt.subplots()
ax.plot(median_energy_per_diode, fit[:, -1] * 1e+3 / fit_currents[-1], marker='x', color='b', ls='-', label='Before', alpha=1)
ax.plot(median_energy_per_diode, fit3[:, -1] * 1e+3 / fit_currents3[-1], marker='x', color='y', ls='-', label='Mid', alpha=1)
ax.plot(median_energy_per_diode, fit2[:, -1] * 1e+3 / fit_currents2[-1], marker='x', color='r', ls='-', label='After', alpha=1)

ax.set_xlabel('Median Proton energy (MeV)')
ax.set_ylabel(f'Linearity slope (cm$^{-2}$)')
format_save(save_path=results_path, save_name=f"LinearitySlope_vs_Energy", dpi=300, save_format=save_format, fig=fig,
            legend=True)

# --- Plot normed slope of linearity vs energy ---
fig, ax = plt.subplots()

slopes1 = fit[:, -1] * 1e+3 / fit_currents[-1]
slopes1 /= slopes1.max()
slopes3 = fit3[:, -1] * 1e+3 / fit_currents3[-1]
slopes3 /= slopes3.max()
slopes2 = fit2[:, -1] * 1e+3 / fit_currents2[-1]
slopes2 /= slopes2.max()

ax.plot(median_energy_per_diode, slopes1, marker='x', color='b', ls='-', label='Before',
        alpha=1)
ax.plot(median_energy_per_diode, slopes3, marker='x', color='y', ls='-', label='Mid',
        alpha=1)
ax.plot(median_energy_per_diode, slopes2, marker='x', color='r', ls='-', label='After',
        alpha=1)
ax.set_xlim(ax.get_xlim())

srim_sim = df['Elec. dE/dx']+df['Nuclear dE/dx']
srim_sim /= srim_sim.max()
ax.plot(df['Ion Energy'], srim_sim, c='k', ls='--', label='SRIM in GaN')

ax.set_xlabel('Median Proton energy (MeV)')
ax.set_ylabel(f'Normed Linearity slope')
format_save(save_path=results_path, save_name=f"NormedLinearitySlope_vs_Energy", dpi=300, save_format=save_format, fig=fig,
            legend=True)

# --- Plot normed slope of linearity vs energy ---
dose_exp = []
for i in range(128):
    position = 25 - 16.5 + i * 0.25
    sim_pixel = int(position / 0.25)
    dose_exp.append(np.mean(dose[sim_pixel, 98:102]))
dose_exp = np.array(dose_exp)
dose_exp /= dose_exp.max()

fig, ax = plt.subplots()

slopes1 = fit[:, -1] * 1e+3 / fit_currents[-1]
slopes1 /= slopes1.max()
slopes3 = fit3[:, -1] * 1e+3 / fit_currents3[-1]
slopes3 /= slopes3.max()
slopes2 = fit2[:, -1] * 1e+3 / fit_currents2[-1]
slopes2 /= slopes2.max()

ax.plot(dose_exp, slopes1, marker='x', color='b', ls='-', label='Before',
        alpha=1)
ax.plot(dose_exp, slopes3, marker='x', color='y', ls='-', label='Mid',
        alpha=1)
ax.plot(dose_exp, slopes2, marker='x', color='r', ls='-', label='After',
        alpha=1)
ax.plot([0, 1], [0, 1], c='k', ls='--')
ax.set_xlabel('Relative Exposition Sim of the diodes')
ax.set_ylabel(f'Normed Linearity slope')
format_save(save_path=results_path, save_name=f"NormedLinearitySlope_vs_NormedExpositionSim", dpi=300, save_format=save_format, fig=fig,
            legend=True)

# --- Plot normed slope of linearity vs energy ---
dose_exp = cache[1]/cache[1].max()

fig, ax = plt.subplots()

slopes1 = fit[:, -1] * 1e+3 / fit_currents[-1]
slopes1 /= slopes1.max()
slopes3 = fit3[:, -1] * 1e+3 / fit_currents3[-1]
slopes3 /= slopes3.max()
slopes2 = fit2[:, -1] * 1e+3 / fit_currents2[-1]
slopes2 /= slopes2.max()

ax.plot(dose_exp, slopes1, marker='x', color='b', ls='-', label='Before',
        alpha=1)
ax.plot(dose_exp, slopes3, marker='x', color='y', ls='-', label='Mid',
        alpha=1)
ax.plot(dose_exp, slopes2, marker='x', color='r', ls='-', label='After',
        alpha=1)
ax.plot([0, 1], [0, 1], c='k', ls='--')
ax.set_xlabel('Relative Exposition after array response of the diodes')
ax.set_ylabel(f'Normed Linearity slope')
format_save(save_path=results_path, save_name=f"NormedLinearitySlope_vs_NormedExpositionArray", dpi=300, save_format=save_format, fig=fig,
            legend=True)

# --- Plot slope of linearity vs energy ---
median_energy_per_diode = []
for i in range(128):
    position = 25-16.5+i*0.25
    sim_pixel = int(position/0.25)
    median_energy_per_diode.append(np.mean(median_energy[sim_pixel, 98:102]))

norm1 = np.mean(signal, axis=0) / np.mean(signal, axis=0).max()
norm3 = np.mean(signal3, axis=0) / np.mean(signal3, axis=0).max()
norm2 = np.mean(signal2, axis=0) / np.mean(signal2, axis=0).max()

fig, ax = plt.subplots()

slopes1 = fit[:, -1] * 1e+3 / fit_currents[-1]
slopes1 /= slopes1.max()
slopes3 = fit3[:, -1] * 1e+3 / fit_currents3[-1]
slopes3 /= slopes3.max()
slopes2 = fit2[:, -1] * 1e+3 / fit_currents2[-1]
slopes2 /= slopes2.max()
ax.plot(median_energy_per_diode, slopes1 / norm1, marker='x', color='b', ls='-', label='Before', alpha=1)
ax.plot(median_energy_per_diode, slopes3 / norm3, marker='x', color='y', ls='-', label='Mid', alpha=1)
ax.plot(median_energy_per_diode, slopes2 / norm2, marker='x', color='r', ls='-', label='After', alpha=1)

ax.set_xlabel('Median Proton energy (MeV)')
ax.set_ylabel(f'Normed linearity slope')
format_save(save_path=results_path, save_name=f"NormedLinearitySlope_vs_Energy", dpi=300, save_format=save_format, fig=fig,
            legend=True)

# --- Plot of linearity over array to identify strange behavior of middle channels ---
current_colour = sns.color_palette("flare", as_cmap=True)
# Before
fig, ax = plt.subplots()
for i, cur in enumerate(currents):
    ax.plot(signal[i], marker='x', color=current_colour(i / len(currents)), ls='--', label=f'{cur:.2f} pAcm${'^{-2}'}$', alpha=1)
ax.set_xlabel('Array Diode')
ax.set_ylabel(f'Signal current ({scale_dict[A.scale][1]}A)')
ax.legend(loc='upper left', fontsize=9)
format_save(save_path=results_path, save_name=f"LinearityArray_Before", dpi=300, save_format=save_format, fig=fig,
            legend=False)
# Mid
fig, ax = plt.subplots()
for i, cur in enumerate(currents3):
    ax.plot(signal3[i], marker='x', color=current_colour(i / len(currents3)), ls='--', label=f'{cur:.2f} pAcm${'^{-2}'}$', alpha=1)
ax.set_xlabel('Array Diode')
ax.set_ylabel(f'Signal current ({scale_dict[A.scale][1]}A)')
ax.legend(loc='upper left', fontsize=9)
format_save(save_path=results_path, save_name=f"LinearityArray_Mid", dpi=300, save_format=save_format, fig=fig,
            legend=False)
# After
fig, ax = plt.subplots()
for i, cur in enumerate(currents2):
    ax.plot(signal2[i], marker='x', color=current_colour(i / len(currents2)), ls='--', label=f'{cur:.2f} pAcm${'^{-2}'}$', alpha=1)

ax.set_xlabel('Array Diode')
ax.set_ylabel(f'Signal current ({scale_dict[A.scale][1]}A)')
ax.legend(loc='upper left', fontsize=9)
format_save(save_path=results_path, save_name=f"LinearityArray_After", dpi=300, save_format=save_format, fig=fig,
            legend=False)