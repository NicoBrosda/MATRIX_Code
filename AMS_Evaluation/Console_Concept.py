import matplotlib.pyplot as plt

from read_MATRIX import *
from FitFuncs import current_conversion

save_path = Path('/Users/nico_brosda/Desktop/iphc_python_misc/Results/concept/')
example = '/Users/nico_brosda/Desktop/iphc_python_misc/matrix_27052024/e2_5000p_nA_2.csv'
example2 = '/Users/nico_brosda/Desktop/iphc_python_misc/matrix_27052024/e2_50p_nA_2.csv'

paths = ['/Users/nico_brosda/Desktop/iphc_python_misc/matrix_27052024/e2_500p_bottom_nA_2.csv',
         '/Users/nico_brosda/Desktop/iphc_python_misc/matrix_27052024/e2_500p_nA_2.csv',
         '/Users/nico_brosda/Desktop/iphc_python_misc/matrix_27052024/e2_500p_top_nA_2.csv']

# ---------------------------------------------------------------------------------------------------------------------
# Example Plot recorded data:
fig, ax = plt.subplots()
data = read(example)
ax.plot(data['Ch35'], label=r'Proton Current: 5$\,$nA', c='k')
ax.set_xlabel(r'Sample (Total range = 5$\,$s)')
ax.set_ylabel(r'Measured Amplitude')
save_name = 'Concept_DataSamples'
format_save(save_path=save_path, save_name=save_name)
plt.show()

# Example Plot recorded data:
fig, ax = plt.subplots()
data = read(example2)
ax.plot(data['Ch35'], label=r'Proton Current: 50$\,$pA', c='k', lw=0.5)
ax.set_xlabel(r'Sample (Total range = 5$\,$s)')
ax.set_ylabel(r'Measured Amplitude')
save_name = 'Concept_DataSamples2'
format_save(save_path=save_path, save_name=save_name)
plt.show()

# ---------------------------------------------------------------------------------------------------------------------
# Example histogram (as otsu motivation)
fig, ax = plt.subplots()
data = read(example)
ax.hist(data['Ch35'], bins=100, label=r'Proton Current: 5$\,$nA', color='k')
ax.set_xlabel(r'Measured Amplitude')
ax.set_ylabel(r'$\#$Samples')
save_name = 'Concept_Histogram'
format_save(save_path=save_path, save_name=save_name)
plt.show()

fig, ax = plt.subplots()
data = read(example2)
ax.hist(data['Ch35'], bins=100, label=r'Proton Current: 50$\,$pA', color='k')
ax.set_xlabel(r'Measured Amplitude')
ax.set_ylabel(r'$\#$Samples')
save_name = 'Concept_Histogram2'
format_save(save_path=save_path, save_name=save_name)
plt.show()

fig, ax = plt.subplots()
data = read(example)
ax.hist(data['Ch35'], bins=100, label=r'Proton Current: 5$\,$nA', color='k')
ax.axvline(threshold_otsu(data['Ch35']), label="Result Otsu's method", color='r')
ax.set_xlabel(r'Measured Amplitude')
ax.set_ylabel(r'$\#$Samples')
save_name = 'Concept_Histogram_Otsu'
format_save(save_path=save_path, save_name=save_name)
plt.show()

fig, ax = plt.subplots()
data = read(example2)
ax.hist(data['Ch35'], bins=100, label=r'Proton Current: 50$\,$pA', color='k')
ax.axvline(threshold_otsu(data['Ch35']), label="Result Otsu's method", color='r')
ax.set_xlabel(r'Measured Amplitude')
ax.set_ylabel(r'$\#$Samples')
save_name = 'Concept_Histogram2_Otsu'
format_save(save_path=save_path, save_name=save_name)
plt.show()

# ---------------------------------------------------------------------------------------------------------------------
# Example diode array
fig, ax = plt.subplots()
data = read(example)
channels = np.arange(0, 66, 1)
samples = np.arange(0, len(data['Ch1']), 1)
X, Y, Z = samples, channels, np.array([np.array(data[i]) for i in data])
print(np.shape(Z))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "white"])

intensity_limits = [-np.inf, np.inf]
intensity_limits2 = (max(np.min(Z), intensity_limits[0]), min(np.max(Z), intensity_limits[1]))
intensity_limits2 = intensity_limits = (-1000, 10000)
levels = np.linspace(intensity_limits2[0], intensity_limits2[1], 100)
# color_map = ax.contourf(X, Y, Z, cmap=cmap, extend='neither', levels=levels, *args, **kwargs)
if np.min(Z) < intensity_limits[0] and np.max(Z) > intensity_limits[1]:
    color_map = ax.contourf(X, Y, Z, cmap=cmap, extend='both', levels=levels)
elif np.min(Z) < intensity_limits[0]:
    color_map = ax.contourf(X, Y, Z, cmap=cmap, extend='min', levels=levels)
elif np.max(Z) > intensity_limits[1]:
    color_map = ax.contourf(X, Y, Z, cmap=cmap, extend='max', levels=levels)
else:
    color_map = ax.contourf(X, Y, Z, cmap=cmap, extend='neither', levels=levels)

norm = matplotlib.colors.Normalize(vmin=intensity_limits2[0], vmax=intensity_limits2[1])
sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map.cmap)
sm.set_array([])
bar = fig.colorbar(sm, ax=ax, extend='max', ticks=color_map.levels)
ax.set_ylabel(r'Position ($64 - \#$ Diode Channel)')
ax.set_xlabel(r'$\#$Samples')
bar.set_label('Measured Amplitude')

save_name = 'Concept_DiodeArray'
format_save(save_path=save_path, save_name=save_name)
plt.show()

# ---------------------------------------------------------------------------------------------------------------------
# Example response in proton beam (mean)
fig, ax = plt.subplots()
data = read(example)
ax.plot(read_channels(data, excluded_channel=[38]), label=r'Proton Current: 5$\,$nA', c='k')
ax.set_xlabel(r'Channel ($\#$Diode in linear array)')
ax.set_ylabel(r'Signal (Amplitude "Beam On" - "Beam Off" )')
save_name = 'Diode_Array'
format_save(save_path=save_path, save_name=save_name)
plt.show()
# Example diode array
fig, ax = plt.subplots()
data = read(example2)
ax.plot(read_channels(data, excluded_channel=[38]), label=r'Proton Current: 5$\,$nA', c='k')
ax.set_xlabel(r'Channel ($\#$Diode in linear array)')
ax.set_ylabel(r'Signal (Amplitude "Beam On" - "Beam Off" )')
save_name = 'Diode_Array2'
format_save(save_path=save_path, save_name=save_name)
plt.show()

# ---------------------------------------------------------------------------------------------------------------------
# Motivation normalization method so far
fig, ax = plt.subplots()
for path in paths:
    data = read(path)
    if '_bottom' in path:
        label = 'Proton Beam Shifted to bottom of diode array'
    elif '_top' in path:
        label = 'Proton Beam Shifted to top of diode array'
    else:
        label = 'Proton Beam in the middle diode array'
    ax.plot(read_channels(data, excluded_channel=[38]), label=label)
ax.set_xlabel(r'Channel ($\#$Diode in linear array)')
ax.set_ylabel(r'Signal (Amplitude "Beam On" - "Beam Off" )')
save_name = 'Shifted_Diodes'
format_save(save_path=save_path, save_name=save_name)
plt.show()

fig, ax = plt.subplots()
data = read(example)
ax.plot(read_channels(data, excluded_channel=[38]), label='Raw Signal')
ax.plot(read_channels(data, excluded_channel=[38])*normalization(paths, excluded_channel=[38]), label='Signal after normalization')
ax.set_xlabel(r'Channel ($\#$Diode in linear array)')
ax.set_ylabel(r'Signal (Amplitude "Beam On" - "Beam Off" )')
save_name = 'Normalization'
format_save(save_path=save_path, save_name=save_name)
plt.show()

format_save(save_path=save_path, save_name=save_name)
plt.show()

fig, ax = plt.subplots()
data = read(example2)
ax.hist(data['Ch35'], bins=100, label=r'Proton Current: 50$\,$pA', color='k')
ax.set_xlabel(r'Measured Amplitude')
ax.set_ylabel(r'$\#$Samples')
save_name = 'Concept_Histogram2'
format_save(save_path=save_path, save_name=save_name)
plt.show()

fig, ax = plt.subplots()
data = read(example)
ax.hist(data['Ch35'], bins=100, label=r'Proton Current: 5$\,$nA', color='k')
ax.axvline(threshold_otsu(data['Ch35']), label="Result Otsu's method", color='r')
ax.set_xlabel(r'Measured Amplitude')
ax.set_ylabel(r'$\#$Samples')
save_name = 'Concept_Histogram_Otsu'
format_save(save_path=save_path, save_name=save_name)
plt.show()

fig, ax = plt.subplots()
data = read(example2)
ax.hist(data['Ch35'], bins=100, label=r'Proton Current: 50$\,$pA', color='k')
ax.axvline(threshold_otsu(data['Ch35']), label="Result Otsu's method", color='r')
ax.set_xlabel(r'Measured Amplitude')
ax.set_ylabel(r'$\#$Samples')
save_name = 'Concept_Histogram2_Otsu'
format_save(save_path=save_path, save_name=save_name)
plt.show()

# ---------------------------------------------------------------------------------------------------------------------
# Example diode array
fig, ax = plt.subplots()
data = read(example)
channels = np.arange(0, 66, 1)
samples = np.arange(0, len(data['Ch1']), 1)
X, Y, Z = samples, channels, np.array([np.array(data[i]) for i in data])
print(np.shape(Z))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "white"])

intensity_limits = [-np.inf, np.inf]
intensity_limits2 = (max(np.min(Z), intensity_limits[0]), min(np.max(Z), intensity_limits[1]))
intensity_limits2 = intensity_limits = (-1000, 10000)
levels = np.linspace(intensity_limits2[0], intensity_limits2[1], 100)
# color_map = ax.contourf(X, Y, Z, cmap=cmap, extend='neither', levels=levels, *args, **kwargs)
if np.min(Z) < intensity_limits[0] and np.max(Z) > intensity_limits[1]:
    color_map = ax.contourf(X, Y, Z, cmap=cmap, extend='both', levels=levels)
elif np.min(Z) < intensity_limits[0]:
    color_map = ax.contourf(X, Y, Z, cmap=cmap, extend='min', levels=levels)
elif np.max(Z) > intensity_limits[1]:
    color_map = ax.contourf(X, Y, Z, cmap=cmap, extend='max', levels=levels)
else:
    color_map = ax.contourf(X, Y, Z, cmap=cmap, extend='neither', levels=levels)

norm = matplotlib.colors.Normalize(vmin=intensity_limits2[0], vmax=intensity_limits2[1])
sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map.cmap)
sm.set_array([])
bar = fig.colorbar(sm, ax=ax, extend='max', ticks=color_map.levels)
ax.set_ylabel(r'Position ($64 - \#$ Diode Channel)')
ax.set_xlabel(r'$\#$Samples')
bar.set_label('Measured Amplitude')

save_name = 'Concept_DiodeArray'
format_save(save_path=save_path, save_name=save_name)
plt.show()

# ---------------------------------------------------------------------------------------------------------------------
# Example conversion amplitude - current
fig, ax = plt.subplots()
data = read(example)
ax.plot(current_conversion(read_channels(data, excluded_channel=[38])), label=r'Proton Current: 5$\,$nA', c='k')
ax.set_xlabel(r'Channel ($\#$Diode in linear array)')
ax.set_ylabel(r'Signal (Amplitude "Beam On" - "Beam Off" )')
save_name = 'Current_Conversion'
format_save(save_path=save_path, save_name=save_name)
plt.show()
