import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import lbc_wls as lbc
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = 12
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.it'] = 'Arial:italic'
rcParams['mathtext.bf'] = 'Arial:italic:bold'

def find_nearest(array, values):
    '''Find value(s) in `array` that are nearest to `values`.

    Parameters
    ----------
    array: ndarray
        Array to be searched. If `array` has more than one dimension, only
        the first column is used.
    values : float, ndarray or list
        Single float, ndarray or list with values for which the nearest entries
        in `array` should be found.

    Returns
    -------
    hits : list
        List of indices of closest values in `array`.
    '''
    
    if array.ndim != 1:
        array_1d = array[:,0]
    else:
        array_1d = array

    values = np.atleast_1d(values)
    hits = []

    for i in range(len(values)):

        idx = np.searchsorted(array_1d, values[i], side= "left")
        if idx > 0 and (idx == len(array_1d) or math.fabs(values[i] - array_1d[idx-1]) < math.fabs(values[i] - array_1d[idx])):
            hits.append(idx-1)
        else:
            hits.append(idx)

    return hits

def read_data(file_name, channel, return_dataframe = False):
	'''
	Reading data from FireStingO2 files. 
	channel 1 = gas phase O2 on channel 1
	channel 2 = liquid phase O2 on channel 2
	channel 3 = liquid phase O2 on channel 1
	'''

	channel_mapping = {1: {'O2': 'Oxygen (%O2) [A Ch.1 Main]', 
						   'dt': ' dt (s) [A Ch.1 Main]',
						   'Temp': 'Optical Temp. (°C) [A Ch.1 CompT]'},
					   2: {'O2': 'Oxygen (µmol/L) [A Ch.2 Main]',
					   	   'dt': ' dt (s) [A Ch.2 Main]',
					   	   'Temp': 'Sample Temp. (°C) [A Ch.2 CompT]'},
					   3: {'O2': 'Oxygen (µmol/L) [A Ch.1 Main]', 
						   'dt': ' dt (s) [A Ch.1 Main]',
						   'Temp': 'Sample Temp. (°C) [A Ch.1 CompT]'}}

	data = pd.read_csv(file_name, 
					encoding = 'ISO8859', sep = '	', 
					skip_blank_lines = True, comment = '#', parse_dates = [0], dayfirst = True)

	data = data.dropna(subset = ['Date [A Ch.1 Main]'])

	data_strings = channel_mapping[channel]

	o2_data = data[data_strings['O2']].to_numpy()
	time = data[data_strings['dt']].to_numpy()


	try:
		temp = data[data_strings['Temp']].to_numpy()
	except KeyError:
		temp = 0

	if return_dataframe:
		return time, o2_data, temp, data
	else:
		return time, o2_data, temp

def plot_properties(data, time, name, ax, position, color):

	if position == -1:
		plotting_ax = ax
	else:
		plotting_ax = ax.twinx()

	if position > 0:
		plotting_ax.spines['right'].set_position(('axes', 1 + position * 0.15))

	plotting_ax.yaxis.label.set_color(color)
	plotting_ax.tick_params(axis = 'y', colors = color)
	plotting_ax.set_ylabel(name)

	plot = plotting_ax.plot(time, data[name], color = color, label = name)

	return plot

def ae_399():
	time, o2_data, temp, data = read_data('O2_Data_Analysis/2024-12-05_135123_AE-399-2-Ch1.txt',
											channel = 1, return_dataframe = True)

	start = 1045
	end = 4645

	data_corrected, y_baseline, y_corrected = lbc.pre_signal_fitting(np.c_[time, o2_data], 0, start, 1, plotting = False)

	colors = plt.cm.plasma(np.linspace(0, 1, 4))[:-1]
	
	# Create figure with two subplots
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4))
	fig.subplots_adjust(left=0.08, right=0.95, bottom=0.15, top=0.95, wspace=0.25)

	# Plot A
	ax1.plot(time, o2_data, color='black', label='Experimental data')
	ax1.plot(time, y_baseline, '--', color='darkblue', label='Baseline')

	ax1.axvspan(time[0], start, alpha=0.2, color=colors[0], label='Pre-reaction')
	ax1.axvspan(start, end, alpha=0.2, color=colors[1], label='Irradiation')
	ax1.axvspan(end, time[-1], alpha=0.2, color=colors[2], label='Post-reaction')

	ax1.set_xlim(time[0], time[-1])
	ax1.set_xlabel('Time / s')
	ax1.set_ylabel('Oxygen / vol%')
	ax1.legend(loc='upper left')

	# Plot B
	ax2.plot(time, y_corrected, color='darkgreen', label='Baseline corrected data')

	ax2.axvspan(time[0], start, alpha=0.2, color=colors[0], label='Pre-reaction')
	ax2.axvspan(start, end, alpha=0.2, color=colors[1], label='Irradiation')
	ax2.axvspan(end, time[-1], alpha=0.2, color=colors[2], label='Post-reaction')

	ax2.set_xlim(time[0], time[-1])
	ax2.set_xlabel('Time / s')
	ax2.set_ylabel('Oxygen / vol%')
	ax2.legend(loc='upper left')

	# Add labels A and B outside plots
	fig.text(0.01, 0.93, 'A', fontsize=22, fontweight='bold')
	fig.text(0.51, 0.93, 'B', fontsize=22, fontweight='bold')

	# Ensure the layout is tight
	plt.tight_layout(rect=[0, 0, 1, 0.95]) 

	# Save figure
	#fig.savefig('/Users/jacob/Documents/Water_Splitting/Frontiers_in_Energy_Research/Data/O2_Data_Analysis.pdf')


if __name__ == '__main__':
	ae_399()

	plt.show()

