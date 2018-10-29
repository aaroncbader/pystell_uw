Currently there are two classfiles

read_vmec.py
read_boozmn.py

These read the wout files from VMEC and the boozmn files from xbooz_xform

There are also some plotting options for each.

It requires the following python libraries:
netcdf4
numpy
scipy
matplotlib

invocation is easy, for example to plot iota profile:

---------------------

from read_vmec import read_vmec

data = read_vmec('path-to-file.nc')
data.plot_iota(show=True)
