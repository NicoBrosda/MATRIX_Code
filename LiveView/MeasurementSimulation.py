import os
import time
from pathlib import Path
import numpy as np
from EvaluationSoftware.helper_modules import path_check, save_text

measurement_time = 0.5
step_width = 0.5
measurements = 200

start_x = 0.0
start_y = 0.0

channels = 128
signal_level = 1000
noise_level = 100

folder_path = Path('/Users/nico_brosda/Desktop/NewMaps/')
save_name = 'Test'

for file in os.listdir(folder_path):
    os.remove(folder_path/file)

for i in range(measurements):
    print('Measurement: ', i)
    # Time spacing
    time.sleep(measurement_time)

    # Header
    cache = 'Voltage,'
    for channel in range(channels):
        cache += 'Col'+str(channel+1)+','

    file = [cache]

    # Simulated data
    for file_lines in range(50):
        cache = '1,'
        for channel in range(channels):
            if measurements / 5 < i < measurements * 4 / 5 and channels/4 < channel < channels*4/5:
                cache += str(signal_level+noise_level*np.random.random())+','
            else:
                cache += str(noise_level * np.random.random())+','
        file.append(cache)

    # Saving the file
    save_text(file, folder_path, save_name+'_x_'+str(start_x+i*step_width)+'_y_'+str(start_y)+'.csv', newline=True)

