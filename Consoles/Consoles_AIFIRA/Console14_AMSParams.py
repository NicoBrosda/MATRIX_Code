from pathlib import Path
from Method14_AMSParams import run_case


# --------- example prediction ---------
def predict_gain(signal, param_tick):
    # your current model: param_values are tick indices 0,1,2,...
    return signal / (param_tick)

def predict_const(signal, param_tick):
    return signal

def predict_power(signal, param):
    return signal * ((9+1)/(9+param))**2

def predict_power_noD(signal, param):
    return signal * (1/param)**2

def predict_tint(signal, param, start=10000):
    return int(signal * (param/start)**3)


# --------- CASE 1 ----------
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_071025_UVTest/BigMatrix/Param_Analyze/')
name = 'GainUV1'
results_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/Results_03_26_AIFIRA/AMSparam/{name}')

params = ['G1', 'G2', 'G4']
param_values = [1, 2, 4]
files = ['Signal_1,9_.csv', 'Signal_1,9_G2.csv', 'Signal_1,9_G4.csv']

run_case(folder_path=folder_path, results_path=results_path, params=params, files=files, base_name=name,
         predict_func=predict_gain, param_values=param_values, random_channels=[0, 1, 2], random_channel_count=None,
         random_seed=42, plot_std_directly=False, annotate_ratios=True, close_after_save=False,
)

# --------- CASE2 1 ----------
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_071025_UVTest/Settings/')
name = 'GainUV2'
results_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/Results_03_26_AIFIRA/AMSparam/{name}')

params = ['G1', 'G2', 'G4']
param_values = [1, 2, 4]
files = ['G1_P4.csv', 'G2_P4.csv', 'G4_P4.csv']

run_case(folder_path=folder_path, results_path=results_path, params=params, files=files, base_name=name,
         predict_func=predict_gain, param_values=param_values, random_channels=[0, 1, 2], random_channel_count=None,
         random_seed=42, plot_std_directly=False, annotate_ratios=True, close_after_save=False,
)

# --------- CASE 2 ----------
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_071025_UVTest/BigMatrix/Param_Analyze/')
name = 'PowerUV1'
results_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/Results_03_26_AIFIRA/AMSparam/{name}')
predict = predict_const
params = ['P1', 'P2', 'P3', 'P4']
param_values = [0, 1, 2, 3]
files = ['P1.csv', 'P2.csv', 'P3.csv', 'P4.csv']

run_case(folder_path=folder_path, results_path=results_path, params=params, files=files, base_name=name,
         predict_func=predict, param_values=param_values, random_channels=[0, 1, 2, 64, 127], random_channel_count=None,
         random_seed=42, plot_std_directly=False, annotate_ratios=True, close_after_save=False,
)

# --------- CASE2 2 ----------
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_071025_UVTest/Settings/')
name = 'G4PowerUV2'
results_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/Results_03_26_AIFIRA/AMSparam/{name}')
predict = predict_const
params = ['P1', 'P4']
param_values = [0, 1]
files = ['P1_G4.csv', 'G4_P4.csv']

run_case(folder_path=folder_path, results_path=results_path, params=params, files=files, base_name=name,
         predict_func=predict, param_values=param_values, random_channels=[0, 1, 2, 64, 127], random_channel_count=None,
         random_seed=42, plot_std_directly=False, annotate_ratios=True, close_after_save=False,
)

# --------- CASE 3 ----------
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_071025_UVTest/BigMatrix/Param_Analyze/')
name = 'CLKRangeUV1'
results_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/Results_03_26_AIFIRA/AMSparam/{name}')
predict = predict_const
params = ['CLK1', 'CLK2', 'CLK3', 'CLK4']
param_values = [0, 1, 2, 3]
files = ['CLK1.csv', 'CLK2.csv', 'CLK3.csv', 'CLK4.csv']

run_case(folder_path=folder_path, results_path=results_path, params=params, files=files, base_name=name,
         predict_func=predict, param_values=param_values, random_channels=[0, 1, 2, 64, 127], random_channel_count=None,
         random_seed=42, plot_std_directly=False, annotate_ratios=True, close_after_save=False,
)

# --------- CASE2 3 ----------
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_071025_UVTest/Settings/')
name = 'CLKRangeUV2'
results_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/Results_03_26_AIFIRA/AMSparam/{name}')
predict = predict_const
params = ['CLK1', 'CLK2', 'CLK3', 'CLK4']
param_values = [0, 1, 2, 3]
files = ['G1_P4_CLKR1.csv', 'G1_P4_CLKR2.csv', 'G1_P4_CLKR3.csv', 'G1_P4_CLKR4.csv']

run_case(folder_path=folder_path, results_path=results_path, params=params, files=files, base_name=name,
         predict_func=predict, param_values=param_values, random_channels=[0, 1, 2, 64, 127], random_channel_count=None,
         random_seed=42, plot_std_directly=False, annotate_ratios=True, close_after_save=False,
)

# --------- CASE 4 ----------
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_071025_UVTest/BigMatrix/Param_Analyze/')
name = 'CLKDividerUV1'
results_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/Results_03_26_AIFIRA/AMSparam/{name}')
predict = predict_power
params = ['CLKD1', 'CLKD2', 'CLKD4', 'CLKD8']
param_values = [1, 2, 4, 8]
files = ['CLKD1.csv', 'CLKD2.csv', 'CLKD4.csv', 'CLKD8.csv']

run_case(folder_path=folder_path, results_path=results_path, params=params, files=files, base_name=name,
         predict_func=predict, param_values=param_values, random_channels=[0, 1, 2, 64, 127], random_channel_count=None,
         random_seed=42, plot_std_directly=False, annotate_ratios=True, close_after_save=False,
)

# --------- CASE2 4 ----------
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_071025_UVTest/Settings/')
name = 'CLKDividerUV2'
results_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/Results_03_26_AIFIRA/AMSparam/{name}')
predict = predict_power
params = ['CLKD1', 'CLKD2', 'CLKD3', 'CLKD4', 'CLKD5', 'CLKD6', 'CLKD7', 'CLKD8']
param_values = [1, 2, 3, 4, 5, 6, 7, 8]
files = ['CLKD1.csv', 'CLKD2.csv', 'CLKD3.csv', 'CLKD4.csv', 'CLKD5.csv', 'CLKD6.csv', 'CLKD7.csv', 'CLKD8.csv']

run_case(folder_path=folder_path, results_path=results_path, params=params, files=files, base_name=name,
         predict_func=predict, param_values=param_values, random_channels=[0, 1, 2, 64, 127], random_channel_count=None,
         random_seed=42, plot_std_directly=False, annotate_ratios=True, close_after_save=False,
)

# --------- CASE 5 ----------
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_071025_UVTest/BigMatrix/Param_Analyze/')
name = 'Settings'
results_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/Results_03_26_AIFIRA/AMSparam/{name}')
predict = predict_const
params = ['FastReset', 'FastResetEnable', 'NoRemove', 'RawDataOut']
param_values = [0, 1, 2, 3]
files = ['FastReset.csv', 'FastResetEnable.csv', 'NoRemove.csv', 'RawDataOut.csv']

run_case(folder_path=folder_path, results_path=results_path, params=params, files=files, base_name=name,
         predict_func=predict, param_values=param_values, random_channels=[0, 1, 2, 64, 127], random_channel_count=None,
         random_seed=42, plot_std_directly=False, annotate_ratios=True, close_after_save=False,
)

# --------- CASE 6 ----------
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_071025_UVTest/BigMatrix/Param_Analyze/')
name = 'ResetCycles'
results_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/Results_03_26_AIFIRA/AMSparam/{name}')
predict = predict_const
params = ['Reset1', 'Reset5', 'Reset8']
param_values = [1, 5, 8]
files = ['Reset1.csv', 'Reset5.csv', 'Reset8.csv']

run_case(folder_path=folder_path, results_path=results_path, params=params, files=files, base_name=name,
         predict_func=predict, param_values=param_values, random_channels=[0, 1, 2, 64, 127], random_channel_count=None,
         random_seed=42, plot_std_directly=False, annotate_ratios=True, close_after_save=False,
)

# --------- CASE 6 ----------
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_071025_UVTest/BigMatrix/IntTimeDark/')
name = 'Integration Time Dark'
results_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/Results_03_26_AIFIRA/AMSparam/{name}')
predict = predict_tint
params = ['10000us', '5000us', '1000us', '500us', '400us', '300us', '200us', '100us']
param_values = [10000, 5000, 1000, 500, 400, 300, 200, 100]
files = ['IntTime_10000us.csv', 'IntTime_5000us.csv', 'IntTime_1000us.csv', 'IntTime_500us.csv', 'IntTime_400us.csv',
         'IntTime_300us.csv', 'IntTime_200us.csv', 'IntTime_100us.csv']

run_case(folder_path=folder_path, results_path=results_path, params=params, files=files, base_name=name,
         predict_func=predict, param_values=param_values, random_channels=[0, 1, 2, 64, 127], random_channel_count=None,
         random_seed=42, plot_std_directly=False, annotate_ratios=True, close_after_save=False,
         logy=False,
)

# --------- CASE 6.5 ----------
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_071025_UVTest/BigMatrix/IntTimeDark/')
name = 'Integration Time Dark 2'
results_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/Results_03_26_AIFIRA/AMSparam/{name}')
predict = lambda a, b: predict_tint(a, b, start=1000)
params = ['1000us', '500us', '400us', '300us', '200us', '100us']
param_values = [1000, 500, 400, 300, 200, 100]
files = ['IntTime_1000us.csv', 'IntTime_500us.csv', 'IntTime_400us.csv',
         'IntTime_300us.csv', 'IntTime_200us.csv', 'IntTime_100us.csv']

run_case(folder_path=folder_path, results_path=results_path, params=params, files=files, base_name=name,
         predict_func=predict, param_values=param_values, random_channels=[0, 1, 2, 64, 127], random_channel_count=None,
         random_seed=42, plot_std_directly=False, annotate_ratios=True, close_after_save=False,
         logy=False,
)

# --------- CASE 6.5 ----------
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_071025_UVTest/BigMatrix/IntTimeDark/')
name = 'Integration Time Dark 3'
results_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/Results_03_26_AIFIRA/AMSparam/{name}')
predict = lambda a, b: predict_tint(a, b, start=4200)
params = ['5000us', '1000us', '500us']
param_values = [4200, 1000, 500]
files = ['IntTime_5000us.csv', 'IntTime_1000us.csv', 'IntTime_500us.csv']

run_case(folder_path=folder_path, results_path=results_path, params=params, files=files, base_name=name,
         predict_func=predict, param_values=param_values, random_channels=[0, 1, 2, 64, 127], random_channel_count=None,
         random_seed=42, plot_std_directly=False, annotate_ratios=True, close_after_save=False,
         logy=False,
)

# --------- CASE 6.5 ----------
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_071025_UVTest/BigMatrix/IntTimeDark/')
name = 'Integration Time Dark 4'
results_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/Results_03_26_AIFIRA/AMSparam/{name}')
predict = lambda a, b: predict_tint(a, b, start=4200)
params = ['10000us', '1000us']
param_values = [4200, 1000]
files = ['IntTime_10000us.csv', 'IntTime_1000us.csv']

run_case(folder_path=folder_path, results_path=results_path, params=params, files=files, base_name=name,
         predict_func=predict, param_values=param_values, random_channels=[0, 1, 2, 64, 127], random_channel_count=None,
         random_seed=42, plot_std_directly=False, annotate_ratios=True, close_after_save=False,
         logy=False,
)

# --------- CASE 6.5 ----------
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_071025_UVTest/BigMatrix/IntTimeDark/')
name = 'Integration Time Dark 5'
results_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/Results_03_26_AIFIRA/AMSparam/{name}')
predict = lambda a, b: predict_tint(a, b, start=10000)
params = ['10000us', '1000us']
param_values = [10000, 1000]
files = ['IntTime_10000us.csv', 'IntTime_1000us.csv']

run_case(folder_path=folder_path, results_path=results_path, params=params, files=files, base_name=name,
         predict_func=predict, param_values=param_values, random_channels=[0, 1, 2, 64, 127], random_channel_count=None,
         random_seed=42, plot_std_directly=False, annotate_ratios=True, close_after_save=False,
         logy=False,
)

# --------- CASE 6.5 ----------
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_071025_UVTest/BigMatrix/IntTimeDark/')
name = 'Integration Time Dark 6'
results_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/Results_03_26_AIFIRA/AMSparam/{name}')
predict = lambda a, b: predict_tint(a, b, start=1850)
params = ['10000us', '1000us']
param_values = [1850, 1000]
files = ['IntTime_10000us.csv', 'IntTime_1000us.csv']

run_case(folder_path=folder_path, results_path=results_path, params=params, files=files, base_name=name,
         predict_func=predict, param_values=param_values, random_channels=[0, 1, 2, 64, 127], random_channel_count=None,
         random_seed=42, plot_std_directly=False, annotate_ratios=True, close_after_save=False,
         logy=False,
)

# --------- CASE 7 ----------
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_071025_UVTest/BigMatrix/IntTimeSignal/')
name = 'Integration Time Signal'
results_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/Results_03_26_AIFIRA/AMSparam/{name}')
predict = predict_tint
params = ['10000us', '5000us', '1000us', '500us', '400us', '300us', '200us', '100us']
param_values = [10000, 5000, 1000, 500, 400, 300, 200, 100]
files = ['SIntTime_10000us.csv', 'SIntTime_5000us.csv', 'SIntTime_1000us.csv', 'SIntTime_500us.csv', 'SIntTime_400us.csv',
         'SIntTime_300us.csv', 'SIntTime_200us.csv', 'SIntTime_100us.csv']

run_case(folder_path=folder_path, results_path=results_path, params=params, files=files, base_name=name,
         predict_func=predict, param_values=param_values, random_channels=[0, 1, 2, 64, 127], random_channel_count=None,
         random_seed=42, plot_std_directly=False, annotate_ratios=True, close_after_save=False,
         logy=False,
)

# --------- CASE 7 ----------
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_071025_UVTest/BigMatrix/IntTimeSignal/')
name = 'Integration Time Signal 2'
results_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/Results_03_26_AIFIRA/AMSparam/{name}')
predict = lambda a, b: predict_tint(a, b, start=1000)
params = ['1000us', '500us', '400us', '300us', '200us', '100us']
param_values = [1000, 500, 400, 300, 200, 100]
files = ['SIntTime_1000us.csv', 'SIntTime_500us.csv', 'SIntTime_400us.csv',
         'SIntTime_300us.csv', 'SIntTime_200us.csv', 'SIntTime_100us.csv']

run_case(folder_path=folder_path, results_path=results_path, params=params, files=files, base_name=name,
         predict_func=predict, param_values=param_values, random_channels=[0, 1, 2, 64, 127], random_channel_count=None,
         random_seed=42, plot_std_directly=False, annotate_ratios=True, close_after_save=False,
         logy=False,
)