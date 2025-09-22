import numpy as np
from scipy.constants import e


def doseCalcGaN(protons, thickness, area, model='GATE'):
    if model == 'GATE':
        edep = 16.512e+3 / 2
    else:
        edep = 17.58e+3 / 2

    return edep * thickness * protons * e / (thickness * area * 6.15e+3)


def doseCalc_Cyrce(area_target, thickness_target, density_target, edep_target, irradiation_time, proton_current,
                   conversion_factor=0.69, d_aperture=26e-3):
    protons = proton_current * conversion_factor * area_target / (np.pi * (d_aperture/2)**2) * irradiation_time / e
    edep = edep_target * thickness_target * protons * e
    dose = edep / (thickness_target * area_target * density_target)
    return dose


density_target = 6.15e+3  # kg / m^3
area_target = (0.4e-03)**2  # m^2
thickness_target = 4.5e-6  # m
edep_target = 16.512e+3 / 2 * 1e+6  # eV / m
edep_target = 17.58e+3 / 2 * 1e+6  # eV / m
irradiation_time = 15.53 * 3600  # s
proton_current = 25e-9  # A

print(f'{doseCalc_Cyrce(area_target, thickness_target, density_target, edep_target, irradiation_time, proton_current):.2e}')