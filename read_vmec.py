from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
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


    #Plot rotational transform as a function of s
    def plot_iota(self, show=False):
        s = self.psi[1:]/self.psi[-1]
        plt.plot(s, self.iota[1:])
        if show:
            plt.show()
    
