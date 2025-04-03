import os

from EvaluationSoftware.standard_processes import *
from EvaluationSoftware.parameter_parsing_modules import *

# ----------------------- Short summary log of measurements -----------------------
# Exp 1 : Dark voltage scan ["dark_"]
# Exp 2 - 5 : Linearity (100 pA - 1 nA at target with 400 um diffuser) ["linearity1_"]
# -> exp 6 empty (not possible)
# -> Until here no um in filename for diffuser thickness - ================= Fixed =================
# Exp 7 - 9 : Norming at different detector voltages ["norm{}V_"]
# Exp 10 - 13 : Norming at different proton energies ["norm{}V_P{}_"]
# Exp 14 - 32 : Mapping of wheel aperture for all wheel positions, P0 - P18 ["energydiffmap_P{}_"]
# Exp 33 - 35 : Norming at different voltages for low proton energy ["norm{}V_P{}_"]
# -> Exp 33 named with 1.9 V instead of 1,9 V - ================= Fixed =================
# Exp 36 - 43 : Mapping of PEEK wedge (wrong position) P0 - P6, P18 ["PEEKwedge_P{}_"]
# Exp 44 - 62 : Mapping of PEEK wedge (correct position) P0, P18, P1 - P17 ["PEEKwedge_P{}_"]
# -> exp 47 with P3 instead of P2 in filename!!! - ================= Fixed =================
# Exp 63 : Increased distance 10 mm, P0 ["LargerGap10mm_P{}_"]
# Exp 64 : Dark Voltage End of Day ["darkEnd_"]
# Exp 65 : Increased distance 10 mm, P7 ["LargerGap10mm_P{}_"]
# ------------------------ End day1 ----------------------------------------------
# Exp 66 : Dark Voltage Scan Day2 1-2V ["Dark_"]
# Exp 67 : Dark Voltage Scan Day2 0-2V ["Dark2_"]
# Exp 68 : Voltage Scan 0.8-2V with beam (PEEK wedge, P7, 10 mm distance) ["BeamCurrent1_"]
# Exp 69 - 71 : Increased distance 10 mm, P7, P12, P16 ["Distance10mm_P{}_"]
# -> exp69 wrongly named with P12, P7 is correct - ================= Fixed =================
# Exp 72 - 75 : Increased distance 20 mm, P0, P7, P12, P16 ["Distance20mm_P{}_"]
# Gafchromic I - VII
# -> Switch to 200 um diffuser
# Exp 76 : Norming day2 P0 and 1.9 V ["normday2_"]
# Exp 77 - 95 : Mapping of wheel aperture for all wheel positions, P0 - P18 ["energyDep_"]
# -> Exp 80 contains two runs - only the run with _bis_ is good (no beam in other run) - ====== Fixed =======
# Exp 96 - 97 : Mapping of PEEK wedge (wrong position) P0, P18 ["PEEKWedge_P{}_"]
# Exp 98 - 117 : Mapping of PEEK wedge (correct position) P18, P0 - P17, P19 ["PEEKWedge_P{}_"]
# Gafchromic VIII - XI
# Exp 118 - 125 : Wedge border in middle of aperture P19 - P12 ["PEEKWedgeMiddle_P{}_"]
# -> Exp 120 named labeled falsely with 118 - needs to be identified with P19 / P17 for real Exp 120 - ==== Fixed =====
# Gafchromic XII
# Exp 126 - 128 : Straggling test distance 5 mm - P0, P12, Misc ["Round8mm_5mm_P{}_", "Misc_5mm_P0_"]
# Exp 129 - 131 : Straggling test distance 10 mm - Misc, P0, P12 ["Round8mm_10mm_P{}_", "Misc_10mm_P0_"]
# Exp 132 - 134 : Straggling test distance 20 mm - P12, P0, Misc ["Round8mm_20mm_P{}_", "Misc_20mm_P0_"]
# Exp 135 - 137 : Straggling test distance 40 mm - Misc, P0, P12 ["Round8mm_40mm_P{}_", "Misc_40mm_P0_"]
# Exp 138 : Dark voltage scan end 0-2V ["DarkEnd_"]

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_260325/')

files = os.listdir(folder_path)

def replace(folder_path, files, phrase, replace):
    for file in files:
        if phrase in file:
            left = file.index(phrase)
            rename = file[0:left] + replace + file[left + len(phrase):]
            os.rename(folder_path / file, folder_path / rename)

# Add 'um' in file names
'''
files_change = array_txt_file_search(files, searchlist=['exp1_', 'exp2_', 'exp3_', 'exp4_', 'exp5_'], txt_file=False,
                                     file_suffix='.csv')
print(files_change)
phrase = '_nA'
replace = 'um_nA'
for file in files_change:
    left = file.index(phrase)
    rename = file[0:left] + replace + file[left+len(phrase):]
    os.rename(folder_path / file, folder_path / rename)

files_change = array_txt_file_search(files,searchlist=['exp1_', 'exp2_', 'exp3_', 'exp4_', 'exp5_'], txt_file=False,
                                     file_suffix='.csv')
print(files_change)
# '''

# Rename Exp33 to 1,9 V instead of 1.9 V
'''
files_change = array_txt_file_search(files, searchlist=['exp33_'], txt_file=False, file_suffix='.csv')
print(files_change)
for file in files_change:
    left = file.index('1.9V')
    rename = file[0:left] + '1,9V' + file[left+4:]
    os.rename(folder_path / file, folder_path / rename)

files_change = array_txt_file_search(files, searchlist=['exp33_'], txt_file=False, file_suffix='.csv')
print(files_change)
'''

# Add P2 instead of P3 for experiment 47
'''
searchlist = ['exp46_']
files_change = array_txt_file_search(files, searchlist=searchlist, txt_file=False,
                                     file_suffix='.csv')
print(files_change)
phrase = '_P2_'
replace = '_P1_'
for file in files_change:
    if phrase in file:
        left = file.index(phrase)
        rename = file[0:left] + replace + file[left+len(phrase):]
        os.rename(folder_path / file, folder_path / rename)

files_change = array_txt_file_search(files, searchlist=searchlist, txt_file=False,
                                     file_suffix='.csv')                 
print(files_change)
# '''

'''
searchlist = ['exp47_']
files_change = array_txt_file_search(files, searchlist=searchlist, txt_file=False,
                                     file_suffix='.csv')
print(files_change)
phrase = '_P3_'
replace = '_P2_'
for file in files_change:
    if phrase in file:
        left = file.index(phrase)
        rename = file[0:left] + replace + file[left + len(phrase):]
        os.rename(folder_path / file, folder_path / rename)

files_change = array_txt_file_search(files, searchlist=searchlist, txt_file=False,
                                     file_suffix='.csv')
print(files_change)
# '''

# Add P7 instead of P12 for experiment 69
'''
searchlist = ['2exp69_']
files_change = array_txt_file_search(files, searchlist=searchlist, txt_file=False,
                                     file_suffix='.csv')
print(files_change)
phrase = '_P12_'
replace = '_P7_'
for file in files_change:
    if phrase in file:
        left = file.index(phrase)
        rename = file[0:left] + replace + file[left+len(phrase):]
        os.rename(folder_path / file, folder_path / rename)

files_change = array_txt_file_search(files, searchlist=searchlist, txt_file=False,
                                     file_suffix='.csv')
print(files_change)
# '''

# Rename 1st run of experiment 80
'''
searchlist = ['2exp80_']
files_change = array_txt_file_search(files, searchlist=searchlist, txt_file=False,
                                     file_suffix='.csv')
print(files_change)
phrase = '2exp80_'
replace = '2exp00_nobeam80_'
add_crit = '_bis_'
for file in files_change:
    if phrase in file and add_crit not in file:
        left = file.index(phrase)
        rename = file[0:left] + replace + file[left + len(phrase):]
        os.rename(folder_path / file, folder_path / rename)

files_change = array_txt_file_search(files, searchlist=searchlist, txt_file=False,
                                     file_suffix='.csv')
print(files_change)
# '''
'''
searchlist = ['2exp80_']
files_change = array_txt_file_search(files, searchlist=searchlist, txt_file=False,
                                     file_suffix='.csv')
print(files_change)
phrase = '_bis_'
replace = '_'
for file in files_change:
    if phrase in file:
        left = file.index(phrase)
        rename = file[0:left] + replace + file[left + len(phrase):]
        os.rename(folder_path / file, folder_path / rename)

files_change = array_txt_file_search(files, searchlist=searchlist, txt_file=False,
                                     file_suffix='.csv')
print(files_change)
# '''

# Rename P19 files of experiment 118 to exp120
'''
searchlist = ['2exp118_']
files_change = array_txt_file_search(files, searchlist=searchlist, txt_file=False,
                                     file_suffix='.csv')
print(files_change)
phrase = '2exp118_'
replace = '2exp120_'
add_crit = '_P17_'
for file in files_change:
    if phrase in file and add_crit in file:
        left = file.index(phrase)
        rename = file[0:left] + replace + file[left + len(phrase):]
        os.rename(folder_path / file, folder_path / rename)

files_change = array_txt_file_search(files, searchlist=searchlist, txt_file=False,
                                     file_suffix='.csv')
print(files_change)
# '''