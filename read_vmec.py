#Author: Aaron Bader, UW-Madison 2018
#This is a file to read from a VMEC wout file and
#plot various quantities of interest
#
#It is designed to be versatile allowing to either plot, plot and show, 
#or export data. 

from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve


class vmec_data:
    def __init__(self, fname):
        self.data = Dataset(fname)
        self.rmnc = np.array(self.data.variables['rmnc'][:])
        self.zmns = np.array(self.data.variables['zmns'][:])
        self.lmns = np.array(self.data.variables['lmns'][:])
        self.bmnc = np.array(self.data.variables['bmnc'][:])
        self.xm = np.array(self.data.variables['xm'][:])
        self.xn = np.array(self.data.variables['xn'][:])
        self.xmnyq = np.array(self.data.variables['xm_nyq'][:])
        self.xnnyq = np.array(self.data.variables['xn_nyq'][:])
        self.nfp = np.array(self.data.variables['nfp'])
        self.psi = np.array(self.data.variables['phi'])
        self.ns = len(self.psi)
        self.nmn = len(self.xm)
        self.nmnnyq = len(self.xmnyq)
        self.iota = np.array(self.data.variables['iotas'])


   
        
    # plot a flux surface with flux surface index fs. 
    def fsplot(self, phi=0, fs=-1, ntheta = 50, plot=True, show=False):
        theta = np.linspace(0,2*np.pi,num=ntheta+1)
       
        r = np.zeros(ntheta+1)
        z = np.zeros(ntheta+1)
        for i in xrange(len(self.xm)):
            m = self.xm[i]
            n = self.xn[i]
            angle = m*theta - n*phi
            r += self.rmnc[fs,i]*np.cos(angle)
            z += self.zmns[fs,i]*np.sin(angle)

        if plot:
            plt.plot(r,z)
            plt.axis('equal')
            if show:
                plt.show()
        return r,z

    #Calculates the mirror term on a given flux surface by comparing
    #the outboard midplane value at phi=0, and the outboard midplane value
    #at the half period.  This is the ROSE definition
    def mirror(self, fs=-1):
        if fs < 0:
            fs = self.ns-1
        B1 = self.modb_at_point(fs, 0, 0)
        B2 = self.modb_at_point(fs, 0, np.pi/self.nfp)
        print B1, B2
        return (B1 - B2)/(B1 + B2)

    
    #Calculate modb at a point.
    def modb_at_point(self, fs, theta, phi):
        return sum(self.bmnc[fs,:]*np.cos(self.xmnyq*theta - self.xnnyq*phi))
        
            
    #Plot modb on a field line starting at the outboard midplane for flux
    #surface index fs
    def modb_on_fieldline(self, fs, show=False):
        
        phimax = 4*np.pi
        npoints = 1001
        iota = self.iota[fs]
        phi = np.linspace(0,phimax,npoints)
        thetastar = phi*iota
        theta = np.zeros(npoints)
        modB = np.zeros(npoints)
        rstart = sum(self.rmnc[fs,:])

        def theta_solve(x):
            lam = self.lmns[fs,:]
            lam1 = sum(lam*np.sin(self.xm*x - self.xn*phi[i]))
            return x + lam1 - thetastar[i]
        
        for i in xrange(npoints):
            theta[i] = fsolve(theta_solve, thetastar[i])
            
            modB[i] = sum(self.bmnc[fs,:]*np.cos(
                self.xmnyq*theta[i] - self.xnnyq*phi[i]))            
            
        plt.plot(phi, modB)
        if show:
            plt.show()

    def modb_on_surface(self, fs=-1, ntheta=64, nphi=64, plot=True,
                        show=False):
        #first attempt will use trisurface, let's see how it looks
        r = np.zeros([nphi,ntheta])
        z = np.zeros([nphi,ntheta])
        x = np.zeros([nphi,ntheta])
        y = np.zeros([nphi,ntheta])
        b = np.zeros([nphi,ntheta])
        

        theta = np.linspace(0,2*np.pi,num=ntheta)
        phi = np.linspace(0,2*np.pi/self.nfp, num=nphi)
        for phii in xrange(nphi):
            p = phi[phii]
            for ti in xrange(ntheta):
                th = theta[ti]
                r[phii,ti] = self.r_at_point(fs,th,p)
                z[phii,ti] = self.z_at_point(fs,th,p)
                x[phii,ti] += r[phii,ti]*np.cos(p)
                y[phii,ti] += r[phii,ti]*np.sin(p)
                b[phii,ti] = self.modb_at_point(fs, th, p)
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            my_col = cm.jet((b-np.min(b))/(np.max(b)-np.min(b)))
            
            ax.plot_surface(x,y,z,facecolors=my_col,norm=True)
           
            if show:
                plt.show()
    
    #Plot rotational transform as a function of s
    def plot_iota(self, show=False):
        s = self.psi[1:]/self.psi[-1]
        plt.plot(s, self.iota[1:])
        if show:
            plt.show()

    
    def r_at_point(self, fs, theta, phi):
        return sum(self.rmnc[fs,:]*np.cos(self.xm*theta - self.xn*phi))
    
    def z_at_point(self, fs, theta, phi):
        return sum(self.zmns[fs,:]*np.sin(self.xm*theta - self.xn*phi))   
