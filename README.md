# PyStell

PyStell is a Python library designed to facilitate the analysis and visualization of data from various plasma physics codes, specifically focusing on data from the VMEC code and a Boozer coordinate transformation.

## Installation

To install PyStell, use the following pip command:

```bash
pip install pystell
```


## Classes

### 1. boozer

The `boozer` class is designed to open and read a Boozer file created by the Fortran code `xbooz_xform`.

### 2. vmec_data

The `vmec_data` class is used to read data from a VMEC wout file and plot various quantities of interest. It offers versatility, allowing users to either plot, plot and show, or export data.

### 3. vmec2booz

The `vmec2booz` class is designed to convert VMEC coordinates to Boozer coordinates. It performs the conversion for every flux surface except for the core.

## Dependencies

- `netcdf4`
- `numpy`
- `scipy`
- `matplotlib`

## Example Usage

Here is a simple example to plot the iota profile using `vmec_data`:

```python
from pystell import vmec_data

# Create an instance of vmec_data with the path to the VMEC file
data = vmec_data('path-to-file.nc')

# Plot iota profile and display the plot
data.plot_iota(show=True)
```

## License

This project is licensed under the [MIT License](LICENSE).