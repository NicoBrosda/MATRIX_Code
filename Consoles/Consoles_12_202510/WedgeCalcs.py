from matplotlib import pyplot as plt
from uncertainties import ufloat
from uncertainties import umath
from uncertainties import unumpy
import numpy as np
from uncertainties.umath import atan, degrees


def to_ufloat(array):
    print(np.std(array))
    if np.std(array) < 1e-6:
        return np.mean(array)
    else:
        return ufloat(np.mean(array), np.std(array))


lx = to_ufloat([40.007, 39.691])
ly = to_ufloat([40.115, 39.981, 39.944, 39.975])
A = 11.909
B = 27.722 - 11.909
C = 39.691 - 27.722
height = to_ufloat([9.998, 10.060, 10.014])
wedge_len = to_ufloat([37.909, 37.909, 37.867, 37.792, 37.719, 37.686])
thin = ufloat(0.4, 0.1)

print("Wedge length", wedge_len)
print("Sides", lx, ly)
print('Height', height)

# Example: measured coordinates along wedge
x = np.array([37.836, 34.802, 31.800, 28.805, 25.803, 22.800, 19.802, 16.800, 13.800, 10.799, 7.805, 4.800, 1.800])   # -0.035 mm
z = np.array([-4.745, -3.669, -2.928, -2.132, -1.500, -0.691, 0.030, 0.763, 1.559, 2.403, 3.059, 3.924, 4.758])  # 4.718 mm

'''
fig, ax = plt.subplots()
ax.plot(x, z+np.min(z), 'x-', color='k')
ax.set_xlabel('Wedge Length [mm]')
ax.set_ylabel('Wedge Height [mm]')
plt.show()
'''

volume1 = (ly* A * height + ly * C * height + 0.5 * wedge_len * B * (height-thin) + wedge_len * B * thin)
volume2 = (ly* A * height + ly * C * height + 0.5 * wedge_len * B * height)


print(volume1, volume2)

density = 11.8899 / volume1 * 1e3
print(11.8899 / (volume1 * 0.8)* 1e3)
print(11.8899 / (volume1 * 1.2)* 1e3)

print('Density', density)

print('DensityRealistic', density*1.03, 'Uncertaintiy:', density*1.03*0.05)


# --- Linear fit ---
# slope (a) and intercept (b)
a, b = np.polyfit(x, z, 1)

# Residuals and standard deviation of residuals
z_fit = np.polyval([a, b], x)
residuals = z - z_fit
sigma = np.std(residuals, ddof=2)  # residual std

# Standard error of slope (simple linear regression formula)
Sxx = np.sum((x - np.mean(x))**2)
slope_std = sigma / np.sqrt(Sxx)

# Define ufloat for slope
a_uf = ufloat(a, slope_std)

# Calculate angle (in degrees)
angle = degrees(atan(a_uf))

print(f"Slope = {a_uf}")
print(f"Wedge angle = {angle:.3f}°")

# Optional: assess flatness (residuals)
print(f"Residual std (flatness): {sigma:.4f} mm")




