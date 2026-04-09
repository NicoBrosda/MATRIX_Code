import SimpleITK as sitk
from scipy.ndimage import gaussian_filter1d
from Plot_Methods.plot_standards import *


def depth_dose_plot(run_name, output_path=Path('/Users/nico_brosda/GateSimulation/GATE10/Simulation/output/')):
    output_file = output_path / (f"{run_name[:run_name.rindex('_')]}/" + f"_{run_name}_dose.mhd")
    # img = sitk.ReadImage(str(output_file).replace(".mhd", "_dose.mhd"))
    img = sitk.ReadImage(str(output_file).replace(".mhd", "_edep.mhd"))
    data_raw = np.array(sitk.GetArrayFromImage(img))
    print(np.shape(data_raw))
    return data_raw.flatten()



data_proton = gaussian_filter1d(depth_dose_plot('1e+07ThesisProtonBragg_param100') / 1e7, sigma=0.1)
data_proton = data_proton / data_proton.max()

data_proton_nonuc = gaussian_filter1d(depth_dose_plot('1e+07ThesisProtonBraggNoNuc_param100') / 1e7, sigma=0.1)
data_proton_nonuc = data_proton_nonuc / data_proton_nonuc.max()

data_carbon = gaussian_filter1d(depth_dose_plot(f'1e+06ThesisCarbonBragg_param{12*185.5}') / 1e6, sigma=0.1)
data_carbon = data_carbon / data_carbon.max()

data_carbon_nonuc = gaussian_filter1d(depth_dose_plot(f'1e+06ThesisCarbonBraggNoNuc_param{12*185.5}') / 1e6, sigma=0.1)
data_carbon_nonuc = data_carbon_nonuc / data_carbon_nonuc.max()

data_helium = gaussian_filter1d(depth_dose_plot(f'1e+06ThesisHeliumBragg_param{4*99.25}') / 1e6, sigma=0.1)
data_helium = data_helium / data_helium.max()

data_helium_nonuc = gaussian_filter1d(depth_dose_plot(f'1e+06ThesisHeliumBraggNoNuc_param{4*99.25}') / 1e6, sigma=0.1)
data_helium_nonuc = data_helium_nonuc / data_helium_nonuc.max()

data_electron = gaussian_filter1d(depth_dose_plot(f'1e+06ThesisElectron_param{20}') / 1e6, sigma=5)
data_electron = data_electron / data_electron.max()

data_gamma = gaussian_filter1d(depth_dose_plot(f'1e+07ThesisGamma_param{5}') / 1e7, sigma=40)
data_gamma = data_gamma / data_gamma.max()

data_oxygen = gaussian_filter1d(depth_dose_plot(f'1e+06ThesisOxygenBragg_param{219.25*16}') / 1e6, sigma=0.1)
data_oxygen = data_oxygen / data_oxygen.max()

z_pixel=1000/1000

fig, ax = plt.subplots()
ax.plot([i * z_pixel for i in range(np.shape(data_proton)[0])], data_proton, label='Proton')
ax.plot([i * z_pixel for i in range(np.shape(data_proton_nonuc)[0])], data_proton_nonuc, label='Proton Nonuc')
ax.plot([i * z_pixel for i in range(np.shape(data_carbon)[0])], data_carbon, label='Carbon')
ax.plot([i * z_pixel for i in range(np.shape(data_carbon_nonuc)[0])], data_carbon_nonuc, label='Carbon Nonuc')
ax.plot([i * z_pixel for i in range(np.shape(data_helium)[0])], data_helium, label='Helium')
ax.plot([i * z_pixel for i in range(np.shape(data_helium)[0])], data_helium_nonuc, label='Helium Nonuc')
ax.plot([i * z_pixel for i in range(np.shape(data_electron)[0])], data_electron, label='electron')
ax.plot([i * z_pixel for i in range(np.shape(data_gamma)[0])], data_gamma, label='gamma')
ax.plot([i * z_pixel for i in range(np.shape(data_carbon)[0])], data_oxygen, label='Oxygen')


ax.axhline(0, c='k', ls='--')
ax.legend()

plt.show()