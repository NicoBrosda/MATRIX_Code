import pandas as pd

from Plot_Methods.plot_standards import *

# Data imported from .txt file
data = pd.read_csv('../Files/energies_after_wheel_diffusor200.txt', sep='\t', header=4)

fig, ax = plt.subplots()
ax.plot(data['thickness alu (mm)'], data['energie'], c='k')

ax.set_xlabel('Thickness of Al (mm)')
ax.set_ylabel('Energy of the proton beam (MeV)')

format_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_230924/BeamProps/', 'Beam_Energy',
            legend=False)