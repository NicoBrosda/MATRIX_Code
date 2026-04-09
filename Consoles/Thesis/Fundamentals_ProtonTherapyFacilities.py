import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.colors as mcolors
from Plot_Methods.plot_standards import *

plot_size = [fullsize_plot[0], fullsize_plot[1]*0.8]

try:
    # Load the CSV
    df = pd.read_csv(Path("/Users/nico_brosda/Cyrce_Messungen/Info_Files/HadronTherapySites.csv"))
except Exception as e:
    print(e)
    # CSV content from my corrected dataset
    data = {
        "Country": [
            "Argentina", "Australia", "Austria", "Belgium", "Canada", "China",
            "Czech Republic", "Denmark", "Egypt", "France", "Germany", "Georgia",
            "India", "Indonesia", "Israel", "Italy", "Japan", "Norway", "Poland",
            "Romania", "Russia", "Saudi Arabia", "Singapore", "Slovak Rep",
            "South Korea", "Spain", "Sweden", "Switzerland", "Taiwan (China)",
            "The Netherlands", "United Kingdom", "USA",
            "Emirates"
        ],

        # -------------------------
        # WORKING FACILITIES
        # -------------------------
        "Total_Facilities": [
            0, 0, 2, 1, 0, 10,
            1, 1, 0, 3, 7, 0,
            2, 0, 0, 5, 26, 0,
            1, 0, 5, 1, 3, 0,
            3, 2, 1, 1, 5, 3,
            3, 29, 0
        ],

        "Proton_Facilities": [
            0, 0, 1, 1, 0, 6,
            1, 1, 0, 3, 5, 0,
            2, 0, 0, 4, 19, 0,
            1, 0, 5, 1, 3, 0,
            2, 2, 1, 1, 4, 3,
            3, 29, 0
        ],

        "Carbon_Facilities": [
            0, 0, 1, 0, 0, 4,
            0, 0, 0, 0, 2, 0,
            0, 0, 0, 1, 7, 0,
            0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 1, 0,
            0, 0, 0
        ],

        "Patients_Treated": [
            0, 0, 2165, 225, 0, 11365,
            7749, 1034, 0, 22040, 24988, 0,
            1400, 0, 0, 7568, 102681, 0,
            1083, 0, 6570, 0, 0, 0,
            11944, 1226, 2451, 10785, 6525, 5896,
            4834, 171645, 0
        ],

        # NEW: Patients by modality
        "Patients_Treated_Proton": [
            0, 0, 1648, 225, 0, 4152,
            7749, 1034, 0, 22040, 18262, 0,
            1400, 0, 0, 4899, 85000, 0,
            1083, 0, 6570, 0, 0, 0,
            11944, 1226, 2451, 10785, 6523, 5896,
            4834, 171645, 0
        ],

        "Patients_Treated_Carbon": [
            0, 0, 517, 0, 0, 7213,
            0, 0, 0, 0, 6726, 0,
            0, 0, 0, 2669, 17681, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 2, 0,
            0, 0, 0
        ],

        # -------------------------
        # UNDER CONSTRUCTION
        # -------------------------
        "Under_Construction": [
            1, 1, 0, 0, 0, 8,
            0, 0, 0, 1, 0, 0,
            1, 0, 2, 3, 2, 2,
            0, 0, 1, 0, 0, 1,
            2, 1, 0, 0, 0, 0,
            0, 7, 1
        ],

        "Under_Construction_Proton": [
            1, 1, 0, 0, 0, 8,
            0, 0, 0, 0, 0, 0,
            1, 0, 2, 3, 2, 2,
            0, 0, 1, 0, 0, 1,
            1, 1, 0, 0, 0, 0,
            0, 6, 1
        ],

        "Under_Construction_Carbon": [
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0,
            0, 1, 0
        ],

        # -------------------------
        # PLANNING STAGE
        # -------------------------
        "Planning_Stage": [
            0, 1, 0, 1, 1, 7,
            0, 0, 1, 0, 0, 1,
            1, 1, 0, 0, 0, 0,
            1, 1, 1, 0, 0, 0,
            0, 10, 0, 2, 2, 0,
            0, 6, 0
        ],

        "Planning_Proton": [
            0, 1, 0, 1, 1, 6,
            0, 0, 1, 0, 0, 1,
            1, 1, 0, 0, 0, 0,
            1, 1, 1, 0, 0, 0,
            0, 10, 0, 2, 2, 0,
            0, 6, 0
        ],

        "Planning_Carbon": [
            0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0
        ]
    }

    df = pd.DataFrame(data)

    # Save as proper CSV (plain text)
    df.to_csv("/Users/nico_brosda/Cyrce_Messungen/Info_Files/HadronTherapySites.csv", index=False)

# --- Helper to lighten a color slightly ---
def grey_overlay(color, amount=0.3, grey=0.6, alpha=1):
    """
    amount = 0 → original color
    amount = 1 → fully grey
    grey   = brightness of grey (0=black, 1=white)
    """
    base = np.array(mcolors.to_rgb(color))
    grey_rgb = np.array([grey, grey, grey])
    return tuple(((1 - amount) * base + amount * grey_rgb, alpha))

# --- Helper to lighten a color slightly ---
def lighten(color, amount=0.3):
    c = np.array(mcolors.to_rgb(color))
    return tuple(c + (1 - c) * amount)

# --- Color definitions ---
c_op_p = plt.cm.tab10(0)      # operational proton
c_op_c = plt.cm.tab10(1)      # operational carbon

# Under construction
c_uc_p = lighten(c_op_p, amount=0.45)
c_uc_c = lighten(c_op_c, amount=0.45)

# Planned → more greyed
c_pl_p = lighten(c_op_p, amount=0.85)
c_pl_c = lighten(c_op_c, amount=0.85)

# Patients treated background (faded operational colors)
c_pat_p = grey_overlay(c_op_p, amount=0.65, alpha=0.7)
c_pat_c = grey_overlay(c_op_c, amount=0.65, alpha=0.7)

# --- Plot setup ---
countries = df['Country']
x = np.arange(len(countries))
width = 0.4

fig, ax1 = plt.subplots(figsize=(18, 8))
ax2 = ax1.twinx()

ax2.set_ylabel('Patients Treated', fontsize=14)
ax2.set_ylim(0, df['Patients_Treated'].max()*1.2)

# ------------------------
# Patients Treated (split by modality)
# ------------------------
ax2.bar(x, df['Patients_Treated_Carbon'], 0.9,
        bottom=df['Patients_Treated_Proton'],
        color=c_pat_c, label='Patients (Carbon)', zorder=-1)

ax2.bar(x, df['Patients_Treated_Proton'], 0.9,
        color=c_pat_p, label='Patients (Proton)', zorder=-1)

ax2.set_zorder(0)
ax2.patch.set_alpha(1)

# ------------------------
# Planning stage (split)
# ------------------------
ax1.bar(x, df['Planning_Carbon'], width,
        bottom=df['Total_Facilities'] + df['Under_Construction_Carbon'] + df['Planning_Proton'] + df['Under_Construction_Proton'],
        color=c_pl_c, zorder=3)

ax1.bar(x, df['Under_Construction_Carbon'], width,
        bottom=df['Total_Facilities'] + df['Planning_Proton'] + df['Under_Construction_Proton'],
        color=c_uc_c, zorder=3)

ax1.bar(x, df['Planning_Proton'], width,
        bottom=df['Total_Facilities'] + df['Under_Construction_Proton'],
        color=c_pl_p, zorder=3)

ax1.bar(x, df['Under_Construction_Proton'], width,
        bottom=df['Total_Facilities'],
        color=c_uc_p, zorder=3)

# ------------------------
# Operational (split)
# ------------------------
ax1.bar(x, df['Carbon_Facilities'], width,
        bottom=df['Proton_Facilities'],
        color=c_op_c, label='Operational / Under constrcution / Planned (Carbon)', zorder=3)

ax1.bar(x, df['Proton_Facilities'], width,
        color=c_op_p, label='Operational / Under constrcution / Planned (Proton)', zorder=3)

# ------------------------
# Labels and layout
# ------------------------
ax1.set_xlabel('Country', fontsize=11)
ax1.set_ylabel('Number of Facilities', fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(countries, rotation=90, fontsize=8)
ax1.tick_params(axis='x', labelrotation=60)
for label in ax1.get_xticklabels(): label.set_horizontalalignment('right')
ax1.set_ylim(0, ax1.get_ylim()[1]*1.1)

ax1.legend(loc='upper left', fontsize=8)
ax1.set_zorder(2)
ax1.patch.set_alpha(0)
ax1.set_title('Hadron Therapy Facilities and Patients per Country', fontsize=12)

just_save(Path('/Users/nico_brosda/Desktop/Dissertation/plots'), 'PTCOG_Facilities',
            save_format='.pdf', dpi=300, plot_size=plot_size, legend=True, fig=fig, minor_xticks=False, minor_yticks=True)