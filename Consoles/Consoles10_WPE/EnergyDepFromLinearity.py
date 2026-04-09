import numpy as np
from scipy.constants import e
from uncertainties import ufloat

def to_ufloat(array):
    if np.std(array) < 1e-6:
        return np.mean(array)
    else:
        return ufloat(np.mean(array), np.std(array))

# The responses from Luisas linearity analysis:
resp_226p7MeV = ufloat(2.73, 0.06)  # pC per MU
resp_100MeV = ufloat(0.45047, 0.00861)
resp_100MeV_big = ufloat(1.25037, 0.07600)

# The areas:
d_100 = to_ufloat([7.947, 7.835])
d_226p7 = to_ufloat([3.210, 3.306])
A_100 = np.pi * d_100**2
A_226p7 = np.pi * d_226p7**2
A_diode = 0.4*0.4
A_big = 0.8*0.8

# For 100 MeV 9.011e+7 protons/MU | For 226.7 MeV 1.614e+8 protons/MU
# The scaled responses:
scaled_226p7MeV = resp_226p7MeV / 1.614e+8 * 1e-12 / e / (A_diode / A_226p7)
scaled_100MeV = resp_100MeV / 9.011e+7 * 1e-12 / e / (A_diode / A_100)
scaled_100MeV_big = resp_100MeV_big / 9.011e+7 * 1e-12 / e / (A_big / A_100)

scaled_226p7MeV2 = resp_226p7MeV / 1.614e+8 * 1e-12 / e / (A_diode / A_226p7) / 0.39
scaled_100MeV2 = resp_100MeV / 9.011e+7 * 1e-12 / e / (A_diode / A_100) / 0.39

print(scaled_226p7MeV, scaled_100MeV, scaled_100MeV_big)
print(scaled_226p7MeV2, scaled_100MeV2)

print(scaled_100MeV / scaled_226p7MeV)
print(5.62/3.17)
print(1 / scaled_100MeV * 66.67551781709251)
print(1 / scaled_226p7MeV * 37.64167548802502)

print(1 * scaled_100MeV / 66.67551781709251)
print(1 * scaled_226p7MeV / 37.64167548802502)

print(1 * scaled_100MeV / (66.67551781709251*36/30))
print(1 * scaled_226p7MeV / (37.64167548802502*36/30))

print(1 * scaled_100MeV2 / 66.67551781709251)
print(1 * scaled_226p7MeV2 / 37.64167548802502)

print(1 * scaled_100MeV2 / (66.67551781709251*36/30))
print(1 * scaled_226p7MeV2 / (37.64167548802502*36/30))

print(1/50)

print(66.67551781709251*36/30)
print(37.64167548802502*36/30)