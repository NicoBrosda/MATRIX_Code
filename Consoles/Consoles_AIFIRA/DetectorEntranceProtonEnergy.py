from Consoles.StyleConsoles.Utils_ImageLoad import *
from EvaluationSoftware.simulation_connectors import *

save_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/Results_03_26_AIFIRA/DetectorEntranceSignal/')

save_format = '.png'


output_path = Path("/Users/nico_brosda/GateSimulation/GATE10/Simulation/output/")

Np = 1e4   # <-- your number of primaries

run_name = "1e4WholeDetectorEnergyVar_param"

dose_base_path = output_path / run_name[:run_name.rindex('_')]

param_list = []
edep_list = []

params = list(np.arange(0, 1, 0.05)) + list(np.arange(0.4, 0.8, 0.02))
params = np.sort(params)

for param in params:
    print(param)
    try:
        output_file = dose_base_path / f"_{run_name}{param}_dose.mhd"
        dose_img = sitk.ReadImage(str(output_file).replace(".mhd", "_dose.mhd"))
        edep_img = sitk.ReadImage(str(output_file).replace(".mhd", "_edep.mhd"))
        print(np.shape(edep_img))
        edep_list.append(edep_img[0])
        param_list.append(param)
    except Exception as e:
        print(e)
        continue


param_list = np.array(param_list)
edep_list = np.array(edep_list) * 1e3 / Np

fig, ax = plt.subplots()
ax.plot(param_list, edep_list, ls='-', c='k')
ax.set_xlabel("Entrance Proton Energy (MeV)")
ax.set_ylabel("E_dep in 2 um active GaN layer (keV)")

format_save(save_path, "DetectorEntranceSignal", fig=fig, save_format=save_format)