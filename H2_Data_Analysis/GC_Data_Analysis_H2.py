import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt

# Input files

calibration_file = 'H2_Data_Analysis/NB_062_Calibration.csv'
data_file = 'H2_Data_Analysis/NB_062_Experimental Data.csv'

# Calibration

calibration_data = np.genfromtxt(calibration_file, skip_header = 1, delimiter = ';')
peak_area = calibration_data[:,1]
vol_perc = calibration_data[:,0]

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(peak_area, vol_perc)

fit_x_values = np.linspace(0, np.amax(peak_area), 10)
fit = slope * fit_x_values + intercept

# Data analysis

data = np.genfromtxt(data_file, skip_header = 1, delimiter = ';')
vol_perc_results = slope * data + intercept
names = np.genfromtxt(data_file, skip_footer = 1, delimiter = ';', dtype = 'str')

# Visualization and plotting of results

fig, ax = plt.subplots(figsize = (6, 5))
fig.subplots_adjust(left = 0.12, right = 0.95, top = 0.95, bottom = 0.12)

ax.plot(fit_x_values, fit, color = 'green', label = 'Calibration line')
ax.plot(peak_area, vol_perc, '.', color = 'darkgreen', markersize = 20, label = 'Calibration data points')

ax.set_xlabel('Peak area (absolute)')
ax.set_ylabel('Hydrogen / vol%')
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
    written_output.append(f'{name} Vol% H2:     {vol_perc_results[counter]}\n')
    #ax.plot(data[counter], vol_perc_results[counter], '.', label = name, color = 'orange', markersize = 20)

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