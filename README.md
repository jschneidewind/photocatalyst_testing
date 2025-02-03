# Testing of particulate photocatalysts for water splitting

This repository contains the data and code accompanying the manuscript "Testing of particulate photocatalysts for water splitting". 

# Requirements

```bash
numpy
scipy
pandas
matplotlib
```

# Contents

### O2_Data_Analysis

This subdirectory contains an examplary file containing in situ O2 detection data as well as Python scripts to read and analyze the data, including baseline correction. Furthermore, an Excel sheet for the calculation of the catalytic activity is included.

### H2_Data_Analysis

This subdirectory contains calibration and experimental data files with hydrogen peak areas obtained from GC measurements. Furthermore, it contains a Python script to calculate the linear calibration function and to calculate the H2 headspace concentration from the experimental data.

### H2O2_Data_Analysis

This subdirectory contains calibration and experimental data files for H2O2 quantification using the DPD-POD assay (the experimental values are absorbance @ 551 nm). Furthermore, it contains a Python script to calculate the linear calibration function and to calculate the sample H2O2 concentration (ing mg(H2O2)/L).

# License

This project is licensed under the MIT License - see the LICENSE file for details.
