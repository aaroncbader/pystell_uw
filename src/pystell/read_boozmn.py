"""
Author: Aaron Bader, UW-Madison 2018
Co-author: Ahnaf Tahmid Chowdhury, MIST 2024

This file provides functionality to open and read a boozmn file 
created by the fortran code xbooz_xform.

It can also plot various quantities of interest.
"""


from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from scipy.optimize import minimize
import logging


class Boozer:
    """
    A class for reading and manipulating Boozer files created by the fortran code xbooz_xform.

    Attributes:
    - fname (str): File name of the boozmn file.
    - data (netCDF4.Dataset): The netCDF4 Dataset object containing Boozer file data.
    - bmnc (numpy.ndarray): Array of Boozer poloidal mode amplitudes.
    - rmnc (numpy.ndarray): Array of Boozer radial mode amplitudes.
    - zmns (numpy.ndarray): Array of Boozer poloidal mode phase angles.
    - pmns (numpy.ndarray): Array of Boozer radial mode phase angles.
    - gmn (numpy.ndarray): Array of Boozer normalization factors.
    - xm (numpy.ndarray): Array of poloidal mode numbers.
    - xn (numpy.ndarray): Array of radial mode numbers.
    - phi (numpy.ndarray): Array of Boozer toroidal flux values.
    - nfp (numpy.ndarray): Number of field periods.
    - It (numpy.ndarray): Array of toroidal current profiles.
    - Ip (numpy.ndarray): Array of poloidal current profiles.
    - s (numpy.ndarray): Normalized toroidal flux.
    - nrbooz (int): Number of Boozer radial modes.
    - nr (int): Number of radial grid points.
    - sr (numpy.ndarray): Normalized toroidal flux on half grid.
    - mnmodes (int): Number of poloidal mode numbers.
    - interpb_at (int): Index of radial grid point for interpolation of bmnc.
    - binterp (numpy.ndarray): Array for interpolated bmnc values.
    - interpr_at (int): Index of radial grid point for interpolation of rmnc.
    - rinterp (numpy.ndarray): Array for interpolated rmnc values.
    - interpz_at (int): Index of radial grid point for interpolation of zmns.
    - zinterp (numpy.ndarray): Array for interpolated zmns values.
    - interpp_at (int): Index of radial grid point for interpolation of pmns.
    - pinterp (numpy.ndarray): Array for interpolated pmns values.
    - charge (float): Electron charge in Coulombs.
    
    Examples:
    >>> from pystell import Boozer
    >>> b = Boozer('boozmn.nc')
    """

    def __init__(self, fname):
        """
        Initialize Boozer object with the given boozmn file.

        Args:
        - fname (str): File name of the boozmn file.
        """

        self.data = Dataset(fname)
        self.bmnc = np.array(self.data.variables["bmnc_b"][:])
        self.rmnc = np.array(self.data.variables["rmnc_b"][:])
        self.zmns = np.array(self.data.variables["zmns_b"][:])
        self.pmns = np.array(self.data.variables["pmns_b"][:])
        self.gmn = np.array(self.data.variables["gmn_b"][:])
        self.xm = np.array(self.data.variables["ixm_b"][:])
        self.xn = np.array(self.data.variables["ixn_b"][:])
        self.phi = np.array(self.data.variables["phi_b"][:])
        self.nfp = np.array(self.data.variables["nfp_b"])
        self.It = np.array(self.data.variables["bvco_b"])
        self.Ip = np.array(self.data.variables["buco_b"])
        self.s = self.phi / self.phi[-1]
        self.nrbooz = len(self.bmnc)
        self.nr = len(self.phi)
        self.sr = self.s[self.nr - self.nrbooz :]
        # sr is on the half grid
        self.sr = self.sr - self.sr[0] / 2
        self.mnmodes = len(self.xm)
        self.interpb_at = -1
        self.binterp = np.empty(self.mnmodes)
        self.interpr_at = -1
        self.rinterp = np.empty(self.mnmodes)
        self.interpz_at = -1
        self.zinterp = np.empty(self.mnmodes)
        self.interpp_at = -1
        self.pinterp = np.empty(self.mnmodes)
        self.charge = 1.602e-19

    def booz2rpz(self, s, theta, zeta):
        """
        Convert Boozer coordinates (s, theta, zeta) to cylindrical coordinates (r, phi, z).

        Args:
        - s (float): Normalized toroidal flux coordinate.
        - theta (float): Poloidal angle in radians.
        - zeta (float): Toroidal angle in radians.

        Returns:
        - r (float): Radial coordinate.
        - phi (float): Toroidal angle in radians.
        - z (float): Vertical coordinate.
        
        Examples:
        >>> b = Boozer.booz2rpz(0.5, 1.2, 0.8)
        """
        # get the r value
        r = self.field_at_point(s, theta, zeta, fourier="r")
        # get the z value
        z = self.field_at_point(s, theta, zeta, fourier="z")
        # get the phi value
        phi = zeta + self.field_at_point(s, theta, zeta, fourier="p")
        return r, phi, z

    def booz2xyz(self, s, theta, zeta):
        """
        Convert Boozer coordinates (s, theta, zeta) to Cartesian coordinates (x, y, z).

        Args:
        - s (float): Normalized toroidal flux coordinate.
        - theta (float): Poloidal angle in radians.
        - zeta (float): Toroidal angle in radians.

        Returns:
        - x (float): Cartesian x-coordinate.
        - y (float): Cartesian y-coordinate.
        - z (float): Cartesian z-coordinate.
        
        Examples:
        >>> b = Boozer.booz2xyz(0.5, 1.2, 0.8)
        """
        r, phi, z = self.booz2rpz(s, theta, zeta)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        return x, y, z

    def xyz2booz(self, x, y, z):
        """
        Convert Cartesian coordinates (x, y, z) to Boozer coordinates (s, theta, zeta).

        Args:
        - x (float): Cartesian x-coordinate.
        - y (float): Cartesian y-coordinate.
        - z (float): Cartesian z-coordinate.

        Returns:
        - s (float): Normalized toroidal flux coordinate.
        - theta (float): Poloidal angle in radians.
        - zeta (float): Toroidal angle in radians.
        
        Examples:
        >>> b = Boozer.xyz2booz(0.5, 1.2, 0.8)
        """
        # convert to polar
        # r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        r = np.sqrt(x**2 + y**2)

        # set up the booz vector guess
        booz_vec = np.empty(3)

        # get guesses for theta and s
        thguess = self.thetaguess(r, phi, z)
        booz_vec[0] = self.sguess(r, phi, z, thguess)
        booz_vec[1] = thguess
        booz_vec[2] = phi

        # this function takes a numpy array booz_coords
        # and returns a float representing the difference
        def solve_function(booz_coords):

            booz_coords[0] = abs(booz_coords[0])
            xg, yg, zg = self.booz2xyz(booz_coords[0], booz_coords[1], booz_coords[2])
            ans = 0
            ans += (x - xg) ** 2
            ans += (y - yg) ** 2
            ans += (z - zg) ** 2
            return np.sqrt(ans)

        # set bounds for s, theta and zeta
        bounds = (
            (0.0, 1.0),
            (booz_vec[1] - np.pi / 2, booz_vec[1] + np.pi / 2),
            (booz_vec[2] - np.pi / 2, booz_vec[2] + np.pi / 2),
        )

        sol = minimize(
            solve_function, booz_vec, method="L-BFGS-B", tol=1.0e-8, bounds=bounds
        )

        s = sol.x[0]
        mins = 1.0 / (self.nr * 3)
        maxs = 1.0 - mins
        if s < mins:
            logging.warning("s value of %.4f is too low, answer may be incorrect", s)
        if s > maxs:
            print("s value of %.4f is too high, answer may be incorrect", s)

        return sol.x

    def interp_bmn(self, s, fourier="b"):
        """
        Interpolate Boozer mode amplitudes or phase angles at a given normalized toroidal flux coordinate.

        Args:
        - s (float): Normalized toroidal flux coordinate.
        - fourier (str): Type of Fourier component to interpolate ("b", "r", "z", or "p") (default "b").

        Raises:
        - ValueError: If fourier is not one of "b", "r", "z", or "p".
        
        Examples:
        >>> b.interp_bmn(0.5, "r")
        """
        # self.dbdpsi = np.empty(self.mnmodes)
        for i in range(self.mnmodes):
            if fourier == "b":
                bspl = interp.CubicSpline(self.sr, self.bmnc[:, i])
                self.interpb_at = s
                self.binterp[i] = bspl(s)
            elif fourier == "r":
                bspl = interp.CubicSpline(self.sr, self.rmnc[:, i])
                self.interpr_at = s
                self.rinterp[i] = bspl(s)
            elif fourier == "z":
                bspl = interp.CubicSpline(self.sr, self.zmns[:, i])
                self.interpz_at = s
                self.zinterp[i] = bspl(s)
            elif fourier == "p":
                bspl = interp.CubicSpline(self.sr, self.pmns[:, i])
                self.interpp_at = s
                self.pinterp[i] = bspl(s)
            else:
                raise ValueError("fourier must be 'b', 'r', 'z' or 'p'")

    # calculate the b,r or z value at a given point, default is b
    def field_at_point(self, s, theta, zeta, fourier="b"):
        """
        Calculate the value of the Boozer magnetic field, radial mode amplitudes, or poloidal mode phase angles at a given point.

        Args:
        - s (float): Normalized toroidal flux coordinate.
        - theta (float): Poloidal angle in radians.
        - zeta (float): Toroidal angle in radians.
        - fourier (str): Type of Fourier component to interpolate ("b", "r", "z", or "p") (default "b").

        Returns:
        - v (float): Value of the specified Fourier component at the given point.
        
        Examples:
        >>> b.field_at_point(0.5, 0.5, 0.5, "r")
        """
        # make sure we've interpolated at the desired value
        if fourier == "b" and self.interpb_at != s:
            self.interp_bmn(s, fourier="b")
        elif fourier == "r" and self.interpr_at != s:
            self.interp_bmn(s, fourier="r")
        elif fourier == "z" and self.interpz_at != s:
            self.interp_bmn(s, fourier="z")
        elif fourier == "p" and self.interpp_at != s:
            self.interp_bmn(s, fourier="p")

        angle = self.xm * theta - self.xn * zeta
        if fourier == "b":
            v = np.sum(self.binterp * np.cos(angle))
        elif fourier == "r":
            v = np.sum(self.rinterp * np.cos(angle))
        elif fourier == "z":
            v = np.sum(self.zinterp * np.sin(angle))
        elif fourier == "p":
            v = np.sum(self.pinterp * np.sin(angle))

        return v

    def currents_and_derivs(self, s):
        """
        Calculate poloidal and toroidal currents at a given normalized toroidal flux coordinate, and their derivatives.

        Args:
        - s (float): Normalized toroidal flux coordinate.

        Returns:
        - ip (numpy.ndarray): Array containing poloidal current and its derivative at the given point.
        - it (numpy.ndarray): Array containing toroidal current and its derivative at the given point.
        
        Examples:
        >>> b.currents_and_derivs(0.5)
        """
        ipspl = interp.CubicSpline(self.s, self.Ip)
        ip = np.empty(2)
        ip[0] = ipspl(s)
        ip[1] = ipspl.derivatives(s)
        itspl = interp.CubicSpline(self.s, self.It)
        it = np.empty(2)
        it[0] = itspl(s)
        it[1] = itspl.derivatives(s)
        return ip, it

    def field_and_derivs(self, s, theta, zeta):
        """
        Calculate the magnetic field strength and its derivatives at a given point in Boozer coordinates.

        Args:
        - s (float): Normalized toroidal flux coordinate.
        - theta (float): Poloidal angle in radians.
        - zeta (float): Toroidal angle in radians.

        Returns:
        - b (numpy.ndarray): Array (modb, dbdpsi, dbdtheta, dbdzeta) containing magnetic field strength and its derivatives at the given point.
        
        Examples:
        >>> b.field_and_derivs(0.5, 0.5, 0.5)
        """
        if self.interp_at != s:
            self.interp_bmn(s)
        b = np.zeros(4)
        for i in range(self.mnmodes):
            # if self.xn[i] > 5 or self.xm[i] > 5:
            #    continue
            angle = self.xm[i] * theta - self.xn[i] * zeta
            b[0] += self.binterp[i] * np.cos(angle)
            # b[1] += self.dbdpsi[i] * np.cos(angle)
            b[2] += -self.xm[i] * self.binterp[i] * np.sin(angle)
            b[3] += self.xn[i] * self.binterp[i] * np.sin(angle)
        return b

    def dpsidt(self, s, theta, zeta):
        """
        Calculate the derivative of toroidal flux with respect to time at a given point in Boozer coordinates.

        Args:
        - s (float): Normalized toroidal flux coordinate.
        - theta (float): Poloidal angle in radians.
        - zeta (float): Toroidal angle in radians.

        Returns:
        - dpsidt (float): Derivative of toroidal flux with respect to time at the given point.
        
        Examples:
        >>> b.dpsidt(0.5, 0.5, 0.5)
        """
        # simple calculation doesn't need this
        # ip, it = self.currents_and_derivs(s)
        b = self.field_and_derivs(s, theta, zeta)
        # we need to know rho_parallel, or do the bounce average
        # gamma = self.charge*(
        return b[2]

    def make_modb_contour(self, s, ntheta, nzeta, plot=True, show=False):
        """
        Generate a contour plot of the magnetic field strength in Boozer coordinates at a given normalized toroidal flux coordinate.

        Args:
        - s (float): Normalized toroidal flux coordinate.
        - ntheta (int): Number of points to sample for the poloidal angle.
        - nzeta (int): Number of points to sample for the toroidal angle.
        - plot (bool): Whether to plot the contour (default True).
        - show (bool): Whether to display the plot (default False).

        Returns:
        - data (list): List containing theta values, zeta values, and the magnetic field strength array.
        
        Examples:
        >>> b.make_modb_contour(0.5, 100, 100, plot=True, show=True)
        """
        theta = np.linspace(0, 2 * np.pi, ntheta)
        zeta = np.linspace(0, 2 * np.pi, nzeta)
        b = np.empty([ntheta, nzeta])
        for i in range(nzeta):
            for j in range(ntheta):
                b[j, i] = self.field_at_point(s, theta[j], zeta[i])
            # print zeta[i], theta[j], b[j,i]

        if plot:
            plt.contourf(zeta, theta, b, 60, cmap="jet")
            # plt.colorbar()
            if show:
                plt.show()
        return [theta, zeta, b]

    def make_dpsidt_contour(self, s, ntheta, nzeta):
        """
        Generate a contour plot of the derivative of toroidal flux with respect to time in Boozer coordinates at a given normalized toroidal flux coordinate.

        Args:
        - s (float): Normalized toroidal flux coordinate.
        - ntheta (int): Number of points to sample for the poloidal angle.
        - nzeta (int): Number of points to sample for the toroidal angle.
        
        Examples:
        >>> b.make_dpsidt_contour(0.5, 81, 51)
        """
        theta = np.linspace(0, 2 * np.pi, ntheta)
        zeta = np.linspace(0, 2 * np.pi, nzeta)
        psidot = np.empty([ntheta, nzeta])
        for i in range(nzeta):
            for j in range(ntheta):
                psidot[j, i] = self.dpsidt(s, theta[j], zeta[i])
            # print zeta[i], theta[j], psidot[j,i]
        plt.contour(zeta, theta, psidot, 20)
        plt.colorbar()
        plt.show()

    def sguess(self, r, phi, z, theta, r0=None, z0=None):
        """
        Guess the normalized toroidal flux coordinate (s) given a point in cylindrical coordinates (r, phi, z) and a guess for theta.

        Args:
        - r (float): Radial coordinate.
        - phi (float): Toroidal angle in radians.
        - z (float): Vertical coordinate.
        - theta (float): Poloidal angle in radians.
        - r0 (float): Radial coordinate of a reference point (optional, default is None).
        - z0 (float): Vertical coordinate of a reference point (optional, default is None).

        Returns:
        - s_guess (float): Guessed normalized toroidal flux coordinate (s).
        
        Examples:
        >>> b.sguess(0.5, 0.5)
        """
        # if axis is not around, get it
        if r0 is None:
            r0, phi0, z0 = self.booz2rpz(0, 0, phi)

        # get r and z at lcfs
        r1, phi1, z1 = self.booz2rpz(1, theta, phi)

        # squared distances for plasma minor radius and our point at theta
        d_pl = (r1 - r0) ** 2 + (z1 - z0) ** 2
        d_pt = (r - r0) ** 2 + (z - z0) ** 2

        # s guess is normalized radius squared
        s_guess = d_pt / d_pl
        return s_guess

    def thetaguess(self, r, phi, z):
        """
        Guess the poloidal angle (theta) by considering the axis and the last closed flux surface (LCFS) at zeta = phi.

        Args:
        - r (float): Radial coordinate.
        - phi (float): Toroidal angle in radians.
        - z (float): Vertical coordinate.

        Returns:
        - th_guess (float): Guessed poloidal angle (theta).
        
        Examples:
        >>> b.thetaguess(0.5, 0.5)
        """
        r0, phi0, z0 = self.booz2rpz(0, 0, phi)
        r1, phi1, z1 = self.booz2rpz(1, 0, phi)

        # get relative r and z for plasma and our point
        r_pl = r1 - r0
        z_pl = z1 - z0

        r_pt = r - r0
        z_pt = z - z0

        # get theta for plasma and our point
        th_pl = np.arctan2(z_pl, r_pl)
        th_pt = np.arctan2(z_pt, r_pt)

        th_guess = th_pt - th_pl
        return th_guess

    def plot_largest_modes(
        self,
        fs=-1,
        n=10,
        rovera=True,
        ignore0=True,
        show=True,
        ax=None,
        xaxis=None,
        noqa=False,
    ):
        """
        Plot the largest Boozer modes.

        Args:
        - fs (int): Index of the flux surface to compare at (default -1).
        - n (int): Number of modes to plot (default 10).
        - rovera (bool): Whether to plot with respect to r/a or s (default True).
        - ignore0 (bool): Whether to ignore the B00 mode (default True).
        - show (bool): Whether to display the plot (default True).
        - ax (matplotlib.axes.Axes): Axes object to plot on (default None).
        - xaxis (numpy.ndarray): Array of x-axis values (default None).
        - noqa (bool): Whether to ignore the B00 mode when plotting (default False).
        
        Examples:
        >>> b.plot_largest_modes()
        """
        # get sorting index for the desired slice
        bslice = self.bmnc[fs, :]
        bslice = -1 * abs(bslice)
        sortvals = np.argsort(bslice)

        # decide whether to plot vs r over a, or s
        if rovera:
            xaxis = np.sqrt(self.sr)
        else:
            xaxis = self.sr

        if ignore0:
            startval = 1
        else:
            startval = 0

        leg = []  # legend

        # now plot the 10 largest
        if ax is None:
            for i in range(startval, n + startval):
                if noqa and self.xn[sortvals[i]] == 0:
                    continue
                plt.plot(xaxis, self.bmnc[:, sortvals[i]])
                legs = (
                    "n="
                    + str(self.xn[sortvals[i]])
                    + ", m="
                    + str(self.xm[sortvals[i]])
                )
                leg.append(legs)
            plt.legend(leg)
            if rovera:
                plt.xlabel("$r/a$")
            else:
                plt.xlabel("$s$")
            plt.ylabel("$B_{mn}$")
        else:
            for i in range(startval, n + startval):
                ax.plot(xaxis, self.bmnc[:, sortvals[i]])
                legs = (
                    "n="
                    + str(self.xn[sortvals[i]])
                    + ", m="
                    + str(self.xm[sortvals[i]])
                )
                leg.append(legs)
            ax.legend(leg)
            if rovera:
                ax.set_xlabel("$r/a$")
            else:
                ax.set_xlabel("$s$")
            ax.set_ylabel("$B_{mn}$")

        if show:
            plt.show()

