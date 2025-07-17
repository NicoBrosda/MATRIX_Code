import pandas as pd

from EvaluationSoftware.standard_processes import *
from EvaluationSoftware.parameter_parsing_modules import *

folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_irradiation_111224/')
results_path = Path('/Users/nico_brosda/Cyrce_Messungen/Results_irradiation_111224/')

measurements = ['25nA_11_12_2024_10_01_41.txt', '25nA_12_12_2024_15_10_11.txt']


for i, measure in enumerate(measurements):
    if i == 0:
        data_frame = pd.read_csv(folder_path / measure, sep=' ', header=None, skiprows=21, names=['Mean64', 'Mean1', 'Sigma64', 'Sigma1', 'Date'])
    else:
        data_frame = pd.concat([data_frame, pd.read_csv(folder_path / measure, sep=' ', header=None, skiprows=0, names=['Mean64', 'Mean1', 'Sigma64', 'Sigma1', 'Date'])], ignore_index=True)

time = data_frame['Date']


class TimeFormat:
    def __init__(self, param_dict):
        self.year = self.month = self.day = self.hour = self.minute = self.second = 0
        for param in param_dict:
            self.__dict__[param] = param_dict[param]

    def value(self):
        return (self.year*365*24*60*60 + self.month*30*24*60*60 + self.day*24*60*60 + self.hour*60*60 + \
            self.minute*60 + self.second)

    def __sub__(self, other):
        return np.abs(self.value() - other.value())


def time_parser(input_string, order=['day', 'month', 'year', 'hour', 'minute', 'second']):
    input_string = str(input_string)
    time_dict = {}
    for element in order[::-1]:
        try:
            pos = input_string.rindex('_')
        except ValueError:
            try:
                time_dict[element] = float(input_string[0:])
            except ValueError:
                time_dict[element] = 0
            break

        try:
            time_dict[element] = float(input_string[pos+1:])
        except ValueError:
            time_dict[element] = 0
        input_string = input_string[:pos]
    return TimeFormat(time_dict)


times = np.array([time_parser(data_frame['Date'][0]) - time_parser(i) for i in data_frame['Date']]) / 60 / 60
print(data_frame['Date'])

fig, ax = plt.subplots()
ax2 = ax.twinx()
filter = ((data_frame['Mean64'] > 4e4) & (data_frame['Mean64'] < 1.3e5))
times, data = times[filter], data_frame['Mean64'][filter]
# ax.errorbar(times, data_frame['Mean64'], yerr=data_frame['Sigma64'])
A = Analyzer((1, 128), 0.4, 0.1)
A.scale = 'nano'
ax.plot(times[times < 7.7], A.signal_conversion(data[times < 7.7]), ls='', marker='.', color='k')
ax.plot(times[times > 22] - 22 + 7.7, A.signal_conversion(data[times > 22]), ls='', marker='.', color='k')

ax2.plot([0, np.max(times[times > 22] - 22 + 7.7)], [0, 3], c='r')
ax.set_ylim(0, ax.get_ylim()[1])

# ax.plot(times, data_frame['Mean1'])

ax.set_xlabel('Irradiation time (h)')
ax.set_ylabel(f'Signal current ({scale_dict[A.scale][1]}A)')
ax2.set_ylabel('Approximation of cumulated dose (MGy)', color='red')

# ax.set_yscale('log')
# ax.set_ylim(0, 2e5)

# plot_size = (21*cm, 16*cm)
format_save(save_path=results_path, save_name='longterm_irradiation', save_format=save_format, fig=fig)