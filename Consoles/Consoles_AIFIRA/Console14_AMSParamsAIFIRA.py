from pathlib import Path
from Method14_AMSParams import run_case
# predict_gain, predict_const, predict_tint etc. already defined above

# --------- example prediction ---------
def predict_gain(signal, param_tick):
    # your current model: param_values are tick indices 0,1,2,...
    return signal / (param_tick)

def predict_const(signal, param_tick):
    return signal

def predict_power(signal, param):
    return signal * ((9+1)/(9+param))**2

def predict_power(signal, param):
    return signal * ((9+1)/(9+param))**3

def predict_power_noD(signal, param):
    return signal * (1/param)**2

def predict_tint(signal, param, start=10000):
    return int(signal * (param/start)**3)


folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_3D_AIFIRA_2026w12/Nouveau dossier/SettingTest')

# ----- CASE: Gain_1ms_BeamOn -----
name = 'Gain_1ms_BeamOn'
results_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/Results_03_26_AIFIRA/AIFIRA_AMSparam/{name}')
params = ['G1', 'G2', 'G4']
param_values = [1, 2, 4]
files = ['Beam_1m_G1_temp.csv', 'Beam_1m_G2_temp.csv', 'Beam_1m_G4_temp.csv']

run_case(
    folder_path=folder_path,
    results_path=results_path,
    params=params,
    files=files,
    base_name=name,
    predict_func=predict_gain,
    param_values=param_values,
    random_channels=[0, 1, 2],
    random_channel_count=None,
    random_seed=42,
    plot_std_directly=False,
    annotate_ratios=True,
    close_after_save=False,
)


# ----- CASE: Gain_50us_BeamOn -----
name = 'Gain_50us_BeamOn'
results_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/Results_03_26_AIFIRA/AIFIRA_AMSparam/{name}')
params = ['G1', 'G2', 'G4']
param_values = [1, 2, 4]
files = ['Beam_50_G1_temp.csv', 'Beam_50_G2_temp.csv', 'Beam_50_G4_temp.csv']

run_case(
    folder_path=folder_path,
    results_path=results_path,
    params=params,
    files=files,
    base_name=name,
    predict_func=predict_gain,
    param_values=param_values,
    random_channels=[0, 1, 2],
    random_channel_count=None,
    random_seed=42,
    plot_std_directly=False,
    annotate_ratios=True,
    close_after_save=False,
)

# ----- CASE: CLK Divider at 1ms -----
name = 'CLKDivider_1ms'
results_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/Results_03_26_AIFIRA/AIFIRA_AMSparam/{name}')
params = ['CLKD1', 'CLKD2', 'CLKD4', 'CLKD8']
param_values = [1, 2, 4, 8]
files = ['Beam_1m_CLKD1_temp.csv',
         'Beam_1m_CLKD2_temp.csv',
         'Beam_1m_CLKD4_temp.csv',
         'Beam_1m_CLKD8_temp.csv']

run_case(
    folder_path=folder_path,
    results_path=results_path,
    params=params,
    files=files,
    base_name=name,
    predict_func=predict_power,
    param_values=param_values,
    random_channels=[0, 1, 2, 64, 127],
    random_channel_count=None,
    random_seed=42,
    plot_std_directly=False,
    annotate_ratios=True,
    close_after_save=False,
)

# ----- CASE: Gain_500us_BeamOff -----
# BeamOff_500_temp.csv is G1 by default; there are explicit G2/G4 files too.
name = 'Gain_500us_BeamOff'
results_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/Results_03_26_AIFIRA/AIFIRA_AMSparam/{name}')
params = ['G1', 'G2', 'G4']
param_values = [1, 2, 4]
files = ['BeamOff_500_temp.csv', 'BeamOff_500_G2_temp.csv', 'BeamOff_500_G4_temp.csv']

run_case(
    folder_path=folder_path,
    results_path=results_path,
    params=params,
    files=files,
    base_name=name,
    predict_func=predict_gain,
    param_values=param_values,
    random_channels=[0, 1, 2],
    random_channel_count=None,
    random_seed=42,
    plot_std_directly=False,
    annotate_ratios=True,
    close_after_save=False,
)

# ----- CASE: Noise_1ms_Gain -----
name = 'Noise_1ms_Gain'
results_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/Results_03_26_AIFIRA/AIFIRA_AMSparam/{name}')
params = ['G1', 'G2', 'G4']
param_values = [1, 2, 4]   # Ref = NOISE21.csv
files = ['Noise_1m_G1_temp.csv',
         'Noise_1m_G2_temp.csv',
         'Noise_1m_G4_temp.csv',
         ]

run_case(
    folder_path=folder_path,
    results_path=results_path,
    params=params,
    files=files,
    base_name=name,
    predict_func=predict_gain,
    param_values=param_values,
    random_channels=[0, 1, 2, 64, 127],
    random_channel_count=None,
    random_seed=42,
    plot_std_directly=False,
    annotate_ratios=True,
    close_after_save=False,
)


# ----- CASE: Noise_500us_Gain -----
name = 'Noise_500us_Gain'
results_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/Results_03_26_AIFIRA/AIFIRA_AMSparam/{name}')
params = ['G1', 'G2', 'G4']
param_values = [1, 2, 4]
files = ['Noise_500u_G1_temp.csv',
         'Noise_500u_G2_temp.csv',
         'Noise_500u_G4_temp.csv',]

run_case(
    folder_path=folder_path,
    results_path=results_path,
    params=params,
    files=files,
    base_name=name,
    predict_func=predict_gain,
    param_values=param_values,
    random_channels=[0, 1, 2, 64, 127],
    random_channel_count=None,
    random_seed=42,
    plot_std_directly=False,
    annotate_ratios=True,
    close_after_save=False,
)

# ----- CASE: IntTime_Beam_G1_All -----
name = 'IntTime_Beam_G1_All'
results_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/Results_03_26_AIFIRA/AIFIRA_AMSparam/{name}')
params = ['1ms', '500us', '100us', '50us']
param_values = [1000, 500, 100, 50]   # for predict_tint
files = ['Beam_1m_G1_temp.csv',
         'BeamOn_500_temp.csv',
         'Beam_100_G1_temp.csv',
         'Beam_50_G1_temp.csv']

run_case(
    folder_path=folder_path,
    results_path=results_path,
    params=params,
    files=files,
    base_name=name,
    predict_func=lambda s, p: predict_tint(s, p, start=1000),  # 1ms as reference
    param_values=param_values,
    random_channels=[0, 1, 2, 64, 127],
    random_channel_count=None,
    random_seed=42,
    plot_std_directly=False,
    annotate_ratios=True,
    close_after_save=False,
    logy=False,
)

# ----- CASE: IntTime_Beam_Off_All -----
name = 'IntTime_BeamOff_G1_All'
results_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/Results_03_26_AIFIRA/AIFIRA_AMSparam/{name}')
params = ['1ms', '500us', '100us', '50us']
param_values = [1000, 500, 100, 50]   # for predict_tint
files = ['Noise_1m_G1_temp.csv',
         'BeamOff_500_temp.csv',
         'BeamOFF_100_G1_temp.csv',
         'BeamOff_50_G1_temp.csv']

run_case(
    folder_path=folder_path,
    results_path=results_path,
    params=params,
    files=files,
    base_name=name,
    predict_func=lambda s, p: predict_tint(s, p, start=1000),  # 1ms as reference
    param_values=param_values,
    random_channels=[0, 1, 2, 64, 127],
    random_channel_count=None,
    random_seed=42,
    plot_std_directly=False,
    annotate_ratios=True,
    close_after_save=False,
    logy=False,
)

# ----- CASE: IntTime_Beam_Off_All -----
name = 'IntTime_Beam_G4_All'
results_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/Results_03_26_AIFIRA/AIFIRA_AMSparam/{name}')
params = ['1ms', '500us', '50us']
param_values = [1000, 500, 50]   # for predict_tint
files = ['Beam_1m_G4_temp.csv',
         'BeamOn_500_G4_temp.csv',
         'Beam_50_G4_temp.csv']

run_case(
    folder_path=folder_path,
    results_path=results_path,
    params=params,
    files=files,
    base_name=name,
    predict_func=lambda s, p: predict_tint(s, p, start=1000),  # 1ms as reference
    param_values=param_values,
    random_channels=[0, 1, 2, 64, 127],
    random_channel_count=None,
    random_seed=42,
    plot_std_directly=False,
    annotate_ratios=True,
    close_after_save=False,
    logy=False,
)
