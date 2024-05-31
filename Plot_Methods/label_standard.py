# Global language definition:
language_english = True

# Defining the labels you want to use in plots of a certain type offers some advantages: e.g. you have ~20 plots with
# the same layout in your document, but typed a unit wrong. Now you can simply change the label in this document to
# correct the mistake in all plots using these parameters.
# For this application using a python dict seems to be helpful, but feel free to adapt this idea in any way you like.

if not language_english:
    ax_labels = {
        'PL_spectrum': {
            'x': 'Wellenlänge [nm]',
            'y': r'Counts [$10^{4}$ 1/s]',
            'x2': 'Energie [meV]'
        },

        'CV_spectrum': {
            'x': '',
            'y': r'',
            'x2': ''
        },

        'band-gap': {
            'x': '',
            'y': '$E_{g}$ [eV]',
            'y2': 'Wellenlänge [nm]'
        },

        'IV': {
            'x': 'Spannung [V]',
            'y': 'Strom [A]'
        },

        'Intensity_map': {
            'x': 'Position [mm]',
            'y': 'Position [mm]',
            'color': 'Counts'
        }
    }

    meas_param_labels = {'center_wavelength': ['Wellenlänge [nm]'], 'time': ['Integrationszeit [min]'],
                         'laser_power': ['Laserleistung [mW]'], 'name': [''],
                         'laser_temperature': ['Lasertemperatur [°C]'],
                         'grating': ['Gitterkonstante [$\frac{g}{\text{mm}}$]'],
                         'temperature': ['Probentemperatur [K]'],
                         'date': ['Datum der Messung'], 'RTA': ['Ausheiltemperatur [°C]'],
                         'fluence': [r'Fluenz [$10^{13}$ cm$^{-2}$]'], 'Implantation': ['Implantationsparameter']}

else:
    ax_labels = {
        'PL_spectrum': {
            'x': 'Wavelength (nm)',
            'y': r'Counts ($10^{4}$ 1/s)',
            'x2': 'Energy (meV)'
        },

        'CV_spectrum': {
            'x': '',
            'y': r'',
            'x2': ''
        },

        'band-bap': {
            'x': '',
            'y': '$E_{g}$ (eV)',
            'y2': 'Wavelength (nm)'
        },

        'IV': {
            'x': 'Bias (V)',
            'y': 'Current (A)'
        },

        'Intensity_map': {
            'x': 'Position (mm)',
            'y': 'Position (mm)',
            'color': 'Counts'
        }
    }

    meas_param_labels = {'center_wavelength': ['wavelength (nm)'], 'time': ['integration time (min)'],
                         'laser_power': ['Laser power (mW)'], 'name': [''],
                         'laser_temperature': ['Laser temperature (°C)'],
                         'grating': ['grating constant ($\frac{g}{\text{mm}}$)'],
                         'temperature': ['sample temperature (K)'],
                         'date': ['date of measurement'], 'RTA': ['RTA temperature (°C)'],
                         'fluence': [r'fluence ($10^{13}$ cm$^{-2}$)'],
                         'Implantation': ['implantation parameters']}

meas_param_units = {'center_wavelength': 'nm', 'time': 'min', 'laser_power': 'mW', 'name': '',
                    'laser_temperature': '°C', 'grating': '1/mm', 'temperature': 'K', 'date': '', 'RTA': '°C',
                    'fluence': r'$\cdot 10^{13}$ cm$^{-2}$', 'Implantation': ''}