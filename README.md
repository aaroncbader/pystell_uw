# PyStell

PyStell is a Python library designed to facilitate the analysis and visualization of data from various plasma physics codes, specifically focusing on data from the Variational Moments Equilibrium Code (VMEC) and a Boozer coordinate transformation.

## Installation

To install PyStell, use the following pip command:

```bash
pip install pystell
```

## Dependencies

- `netcdf4`
- `numpy`
- `scipy`
- `matplotlib`
- `mayavi` (optional)
- `vtk` (optional)

## Classes

### 1. Boozer

The `Boozer` class is designed to open and read a Boozer file created by the Fortran code `xbooz_xform`.

**Example usage:**

```python
from pystell import Boozer

# Create an instance of Boozer with the path to the Boozer file
booz = Boozer('path-to-file.nc')

# Generate a contour plot of the magnetic field 
# strength in Boozer coordinates at a given normalized 
# toroidal flux coordinate.
booz.make_modb_contour(0.5, 100, 100, plot=True, show=True)
```

### 2. VMECData

The `VMECData` class is used to read data from a VMEC wout file and plot various quantities of interest. It offers versatility, allowing users to either plot, plot and show, or export data.

**Example usage:**

```python
from pystell import VMECData

# Create an instance of VMECData with the path to the VMEC file
data = VMECData('path-to-file.nc')

# Plot iota profile and display the plot
data.plot_iota(show=True)
```

### 3. VMEC2Booz

The `VMEC2Booz` class is designed to convert VMEC coordinates to Boozer coordinates. It performs the conversion for every flux surface except for the core.

**Example usage:**

```python
from pystell import VMEC2Booz

# Initialize VMEC2Booz object with the given VMEC data.
# nboz = 3, mboz = 4
v2b = VMEC2Booz("vmec.wout", 3, 4)

#Write the magnetic field data to a NetCDF file.
title = "Magnetic field data in Boozer coordinates"
v2b.write_boozmn(title)
```

## License

This project is licensed under the [MIT License](LICENSE).
