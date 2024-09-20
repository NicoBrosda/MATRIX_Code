import os
import time
from pathlib import Path
import numpy as np
from EvaluationSoftware.helper_modules import save_text
from EvaluationSoftware.helper_modules import array_txt_file_search

measurement_time = 1
step_width = 0.5
measurements = 10

start_x = 0.0
start_y = 0.0

channels = 128
signal_level = 1000
noise_level = 100

folder_path = Path('./TestMap/')
save_name = 'Test'

for file in array_txt_file_search(os.listdir(folder_path), searchlist=[save_name], file_suffix='.csv', txt_file=False):
    os.remove(folder_path/file)

time.sleep(5)

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

