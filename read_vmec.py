#Author: Aaron Bader, UW-Madison 2018
#This is a file to read from a VMEC wout file and
#plot various quantities of interest
#
#It is designed to be versatile allowing to either plot, plot and show, 
#or export data. 

from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import imp
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve
import scipy.integrate as integrate
import scipy.interpolate as interpolate

try:
    imp.find_module('mayavi')
    use_mayavi = True
    from mayavi import mlab
    import vtk
except ImportError:
    use_mayavi = False


class vmec_data:
    def __init__(self, fname):
        self.data = Dataset(fname)
        self.rmnc = np.array(self.data.variables['rmnc'][:])
        self.zmns = np.array(self.data.variables['zmns'][:])
        self.lmns = np.array(self.data.variables['lmns'][:])
        self.bmnc = np.array(self.data.variables['bmnc'][:])
        self.gmnc = np.array(self.data.variables['gmnc'][:])
        self.xm = np.array(self.data.variables['xm'][:])
        self.xn = np.array(self.data.variables['xn'][:])
        self.xmnyq = np.array(self.data.variables['xm_nyq'][:])
        self.xnnyq = np.array(self.data.variables['xn_nyq'][:])
        self.raxis = np.array(self.data.variables['raxis_cc'][:])
        self.zaxis = np.array(self.data.variables['zaxis_cs'][:])
        self.nfp = np.array(self.data.variables['nfp'])
        self.a = np.array(self.data.variables['Aminor_p'])
        self.psi = np.array(self.data.variables['phi'])
        self.s = self.psi/self.psi[-1]
        self.volume = np.array(self.data.variables['volume_p'])
        self.b0 = np.array(self.data.variables['b0'])
        self.ns = len(self.psi)
        self.nmn = len(self.xm)
        self.nmnnyq = len(self.xmnyq)
        self.iota = np.array(self.data.variables['iotas'])
        self.pres = np.array(self.data.variables['pres'])


    #convert a normalized flux value s to a flux surface index
    def s2fs(self, s, isint=True):
        fs = s*(self.ns-1)
        if isint:
            fs = int(round(fs))
        return fs

    #convert a flux surface index (integer or not) into a normalized flux s
    def fs2s(self, fs):
        s = float(fs)/(self.ns-1)
        return s
    
    
    #Compute the minor radius by evaluating the outboard and inboard R values
    def bean_radius_horizontal(self):
        Rout = 0.0
        Rin = 0.0

        for i in xrange(len(self.xm)):
             Rout += self.rmnc[-1,i]
             if self.xm[i] % 2 == 1:
                 Rin -= self.rmnc[-1,i]
             else:
                 Rin += self.rmnc[-1,i]
        return (Rout - Rin)/2       

 
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
    def modb_on_fieldline(self, fs, phimax=4*np.pi, npoints=1001,
                          plot=True, show=False):
        
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
            
        if plot:
            plt.plot(phi, modB)
            if show:
                plt.show()
        return phi,modB

    #This works, but there is an issue with plot display at high
    #resolution.  I have not figured out how to fix it yet
    def modb_on_surface(self, fs=-1, ntheta=64, nphi=64, plot=True,
                        show=False, outxyz=None, full=False, mayavi=True):
        #first attempt will use trisurface, let's see how it looks
        r = np.zeros([nphi,ntheta])
        z = np.zeros([nphi,ntheta])
        x = np.zeros([nphi,ntheta])
        y = np.zeros([nphi,ntheta])
        b = np.zeros([nphi,ntheta])

        if full:
            divval = 1
        else:
            divval = self.nfp

        theta = np.linspace(0,2*np.pi,num=ntheta)
        phi = np.linspace(0,2*np.pi/divval, num=nphi)
        
        for phii in xrange(nphi):
            p = phi[phii]
            for ti in xrange(ntheta):
                th = theta[ti]
                r[phii,ti] = self.r_at_point(fs,th,p)
                z[phii,ti] = self.z_at_point(fs,th,p)
                x[phii,ti] += r[phii,ti]*np.cos(p)
                y[phii,ti] += r[phii,ti]*np.sin(p)
                b[phii,ti] = self.modb_at_point(fs, th, p)
        my_col = cm.jet((b-np.min(b))/(np.max(b)-np.min(b)))        
        if plot and (not use_mayavi or not mayavi):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            #my_col = cm.jet((b-np.min(b))/(np.max(b)-np.min(b)))
            
            ax.plot_surface(x,y,z,facecolors=my_col,norm=True)
            #set axis to equal
            max_range = np.array([x.max()-x.min(), y.max()-y.min(),
                                  z.max()-z.min()]).max() / 2.0

            mid_x = (x.max()+x.min()) * 0.5
            mid_y = (y.max()+y.min()) * 0.5
            mid_z = (z.max()+z.min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
           
            if show:
                plt.show()

        elif plot and use_mayavi:
            mlab.figure(bgcolor=(1.0, 1.0, 1.0), size=(800,600))
            mlab.mesh(x,y,z, scalars=b)
            if show:                
                mlab.show()
                
        if outxyz is not None:
            wf = open(outxyz, 'w')
            for phii in xrange(nphi):
                for ti in xrange(ntheta):
                    s = (str(x[phii,ti]) + '\t' + str(y[phii,ti]) + '\t'
                         + str(z[phii,ti]) + '\n')
                    wf.write(s)
        #return [x, y, z, b]
            

    def axis(self, phi):
        r = 0
        z = 0
        for i in xrange(len(self.raxis)):
            r += self.raxis[i]*np.cos(i*self.nfp*phi)
            z += self.zaxis[i]*np.sin(i*self.nfp*phi)
        return r,z
        
    #Plot rotational transform as a function of s
    def plot_iota(self, plot=True, show=False):
        s = self.psi[1:]/self.psi[-1]
        if plot:
            plt.plot(s, self.iota[1:])
            if show:
                plt.show()
        return s,self.iota[1:]


    def pressure(self, plot=True, show=False):
        s = self.psi[1:]/self.psi[-1]
        pres = self.pres[1:]
        if plot:
            plt.plot(s, pres)
            if show:
                plt.show()
        return s,pres
    
    def r_at_point(self, fs, theta, phi):
        return sum(self.rmnc[fs,:]*np.cos(self.xm*theta - self.xn*phi))
    
    def z_at_point(self, fs, theta, phi):
        return sum(self.zmns[fs,:]*np.sin(self.xm*theta - self.xn*phi))


    # return dvds, the volume derivative, which is 4 pi^2 abs(g_00). 
    def dvds(self, s, interp=False):        

        if not interp:
            # if we don't want to interpolate, then get an actual value
            fs = self.s2fs(s)
            g = self.gmnc[fs,0]
        else:
            gfunc = interpolate.interp1d(self.s,self.gmnc[:,0])
            g = gfunc(s)
            
        dvds_val = abs(4 * np.pi**2 * g)
        return dvds_val
        
