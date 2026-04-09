import pandas as pd

from Plot_Methods.plot_standards import *
from scipy.interpolate import make_interp_spline

data_protons = pd.read_csv(Path('/Users/nico_brosda/Desktop/MATRIX_Code/Files/PTCOGProton.csv'), names=['x', 'y'], skiprows=1)
data_carbon = pd.read_csv(Path('/Users/nico_brosda/Desktop/MATRIX_Code/Files/PTCOGCarbon.csv'), names=['x', 'y'], skiprows=1)

data_pt = pd.read_csv(Path('/Users/nico_brosda/Desktop/MATRIX_Code/Files/Scopus_ProtonCitation.csv'), skiprows=4, usecols=np.arange(7, 33), nrows=2)

data_FlASH = pd.read_csv(Path('/Users/nico_brosda/Desktop/MATRIX_Code/Files/Scopus_FLASHCitation.csv'), skiprows=4, usecols=np.arange(7, 32), nrows=2)

data_pt = data_pt.to_numpy()
data_FlASH = data_FlASH.to_numpy()

print(data_pt)

plot_size = [fullsize_plot[0], fullsize_plot[1]*0.5]
fig, [ax, ax2]= plt.subplots(1, 2, figsize=plot_size)
fig.subplots_adjust(wspace=0.35)

# Original data
x = data_protons['x']
y = data_protons['y']

# Create smooth x-grid
x_smooth = np.linspace(x.min(), x.max(), 500)

# Cubic spline interpolation
spline = make_interp_spline(x, y, k=3)  # k=3 -> cubic spline
y_smooth = spline(x_smooth)

ax.plot(x_smooth, y_smooth, ls='-', label='protons')

# Original data
x = data_carbon['x']
y = data_carbon['y']

# Create smooth x-grid
x_smooth = np.linspace(x.min(), x.max(), 500)

# Cubic spline interpolation
spline = make_interp_spline(x, y, k=3)  # k=3 -> cubic spline
y_smooth = spline(x_smooth)

ax.plot(x_smooth, y_smooth, ls='-', label='carbon ions')

ax.set_xlabel('Years')
ax.set_ylabel('Treated patients')
ax.legend(
    loc='upper left',
    bbox_to_anchor=(0.00, 0.93),
    bbox_transform=ax.transAxes
)
ax.set_xlim(ax.get_xlim()), ax.set_ylim(0, ax.get_ylim()[1]*1.1)
ax.text(*transform_axis_to_data_coordinates(ax, [0.04, 0.97]),'Data from PTCOG (2026)', ha='left', va='top', fontsize=9, color='grey')

fig.title ='Statistics of the growing particle therapy relevance'

pt_curve = ax2.plot(data_pt[0], data_pt[1], ls='-', label=f'Proton Therapy')
FLASH_curve = ax2.plot(data_FlASH[0], data_FlASH[1], ls='-', label=f'FLASH Therapy')

ax2.legend()
ax2.set_xlim(1998, 2025), ax2.set_ylim(0, ax2.get_ylim()[1]*1.1)

ax2.set_xlabel('Years')
ax2.set_ylabel('Scopus Citation Statistics')

ax2.text(*transform_axis_to_data_coordinates(ax2, [0.04, 0.7]),'h-index = 150', ha='left', va='top', fontsize=9, color=pt_curve[0].get_color())
ax2.text(*transform_axis_to_data_coordinates(ax2, [0.04, 0.6]),'h-index = 91', ha='left', va='top', fontsize=9, color=FLASH_curve[0].get_color())

ax.text(*transform_axis_to_data_coordinates(ax, [0.97, 0.97]), r'\textbf{(a)}', fontsize=10, ha='right',
         va='top', color='k')
ax2.text(*transform_axis_to_data_coordinates(ax2, [0.97, 0.97]), r'\textbf{(b)}', fontsize=10, ha='right',
         va='top', color='k')

format_save(Path('/Users/nico_brosda/Desktop/Dissertation/plots'), 'PTCOG_PatientStatistics',
            save_format='.pdf', dpi=300, plot_size=plot_size, legend=False, fig=fig)