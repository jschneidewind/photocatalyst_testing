import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
import pandas as pd

# Input files

calibration_file = 'H2O2_Data_Analysis/NB_029_Calibration.csv'
data_file = 'H2O2_Data_Analysis/NB_029_Experimental_Data.csv'

# Calibration

calibration_data = np.genfromtxt(calibration_file, skip_header = 1, delimiter = ';')
absorbance = calibration_data[:,1]
concentration = calibration_data[:,0]

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(absorbance, concentration)

fit_x_values = np.linspace(np.min(absorbance), np.amax(absorbance), 10)
fit = slope * fit_x_values + intercept

# Data analysis

data = pd.read_csv(data_file, delimiter = ';', index_col = 0)

concentration_results = []
names = []

for column in data.columns:
    absorbance_value = data.loc['Absorbance', column]
    dilution_factor = data.loc['Dilution_Factor', column]

    concentration_result = (slope * absorbance_value + intercept) * dilution_factor

    concentration_results.append(concentration_result)
    names.append(column)

# Visualization and plotting of results

fig, ax = plt.subplots(figsize = (6, 5))
fig.subplots_adjust(left = 0.12, right = 0.95, top = 0.95, bottom = 0.12)

ax.plot(fit_x_values, fit, color = 'green', label = 'Calibration line')
ax.plot(absorbance, concentration, '.', color = 'darkgreen', markersize = 20, label = 'Calibration data points')

ax.set_xlabel('Absorbance')
ax.set_ylabel('$H_{2}O_{2}$ concentration/ mg $L^{-1}$')
ax.annotate(f'$R^2$ = {r_value**2:.5f}', (0.2, 0.75), xycoords = 'figure fraction')
ax.annotate(f'Slope = {slope:.2E}', (0.2, 0.7), xycoords = 'figure fraction')
ax.annotate(f'Intercept = {intercept:.2E}', (0.2, 0.65), xycoords = 'figure fraction')

written_output = [
'_________________________________\n',
'             Calibration         \n',
'_________________________________\n',
f'Slope:      {slope}\n',
f'Intercept:  {intercept}\n',
f'R^2:        {r_value**2}\n',
'_________________________________\n',
'             Results             \n',
'_________________________________\n'
]

for counter, name in enumerate(names):
    written_output.append(f'{name} H2O2 mg/L:     {concentration_results[counter]}\n')

written_output.append('_________________________________\n')

ax.legend()

name_split = str.split(data_file, '.')
analysis_file_name = name_split[0] + '-Analysis.pdf'

fig.savefig(analysis_file_name)

output_text_file_name = name_split[0] + '-Results.txt'


with open(output_text_file_name, 'w') as file:
    for line in written_output:
        file.write(line)

for entry in written_output:
    print(entry)

plt.show()