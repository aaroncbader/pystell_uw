#Author: Aaron Bader, UW-Madison 2020
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
import scipy.interpolate as interp
from scipy.optimize import minimize

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
        self.bsubumnc = np.array(self.data.variables['bsubumnc'][:])
        self.bsubvmnc = np.array(self.data.variables['bsubvmnc'][:])

        self.xm = np.array(self.data.variables['xm'][:])
        self.xn = np.array(self.data.variables['xn'][:])
        self.xmnyq = np.array(self.data.variables['xm_nyq'][:])
        self.xnnyq = np.array(self.data.variables['xn_nyq'][:])
        self.raxis = np.array(self.data.variables['raxis_cc'][:])
        self.zaxis = np.array(self.data.variables['zaxis_cs'][:])
        self.nfp = np.array(self.data.variables['nfp'])
        self.ntor = np.array(self.data.variables['ntor'])
        self.mpol = np.array(self.data.variables['mpol'])
        self.mnyq = int(max(abs(self.xmnyq)))
        self.nnyq = int(max(abs(self.xnnyq))/self.nfp)
        self.a = np.array(self.data.variables['Aminor_p'])
        self.psi = np.array(self.data.variables['phi'])
        self.psips = np.array(self.data.variables['phips'])
        self.s = self.psi/self.psi[-1] #integer grid
        self.shalf = self.s - self.s[1]/2 #half grid
        self.volume = np.array(self.data.variables['volume_p'])
        self.b0 = np.array(self.data.variables['b0'])
        self.volavgB = np.array(self.data.variables['volavgB'])
        self.ns = len(self.psi)
        self.nmn = len(self.xm)
        self.nmnnyq = len(self.xmnyq)
        self.iota = np.array(self.data.variables['iotaf'])
        self.hiota = np.array(self.data.variables['iotas']) #half grid
        self.jdotb = np.array(self.data.variables['jdotb'])
        self.pres = np.array(self.data.variables['pres'])
        self.betavol = np.array(self.data.variables['beta_vol'])
        self.bvco = np.array(self.data.variables['bvco'])
        self.buco = np.array(self.data.variables['buco'])

        self.aspect = np.array(self.data.variables['aspect'])
        self.rmax_surf = np.array(self.data.variables['rmax_surf'])
        self.rmin_surf = np.array(self.data.variables['rmin_surf'])
        self.zmax_surf = np.array(self.data.variables['zmax_surf'])
        self.betaxis = np.array(self.data.variables['betaxis'])
	

        #interpolation stuff
        self.interpb_at = -1
        self.binterp = np.empty(self.nmnnyq)
        self.interpr_at = -1
        self.rinterp = np.empty(self.nmn)
        self.interpz_at = -1
        self.zinterp = np.empty(self.nmn)
        self.interpl_at = -1
        self.linterp = np.empty(self.nmn)

        #splines get filled as needed
        self.iotaspl = None
        

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

        for i in range(len(self.xm)):
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
        for i in range(len(self.xm)):
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
    def mirror(self, s=1.0):
        B1 = self.modb_at_point(s, 0, 0)
        B2 = self.modb_at_point(s, 0, np.pi/self.nfp)
        #print B1, B2
        return (B1 - B2)/(B1 + B2)

    
    #Calculate modb at a point.
    def modb_at_point(self, s, theta, phi):
    
        self.interp_val(s,fourier='b')
        #remember bmnc is on the half grid
        return sum(self.binterp
                   *np.cos(self.xmnyq*theta - self.xnnyq*phi))
        
            
    #Plot modb on a field line starting at the outboard midplane for flux
    #surface index fs
    #This is mostly deprecated and now calls xyz_on_fieldline
    def modb_on_fieldline(self, fs, phimax=4*np.pi, npoints=1001,
                          phistart = 0, thoffset = 0, plot=True, show=False):

        s = float(fs)/self.ns
        phi, modB, theta = self.xyz_on_fieldline(s,thoffset,phistart,
                            phimax=phimax, npoints=npoints,
                            invmec=True, plot=False,
                            onlymodB = True)

        if plot:
            plt.plot(phi, modB)
            if show:
                plt.show()
        return phi, modB, theta

    #Get x,y,z and modB on a fieldline, options to return r,phi,z instead
    #and some plotting options. Input is three starting coordinates, field following is in the forward direction for now only
    # if starting coordinates are x,y,z no flags are needed
    # if starting coordinates are r,p,z need to set inrpz
    # if starting coordinates are vmec coords, set invmec

    def xyz_on_fieldline(self, x, y, z, phimax=4*np.pi, npoints=1001,
                         invmec=False, inrpz=False, retrpz=False,
                         plot = True, show = False, onlymodB = False):
        if not invmec and not inrpz:
            s, theta, zeta = self.xyz2vmec(x,y,z)
        elif not invmec and inrpz:
            s, theta, zeta = self.rpz2vmec(x,y,z)
        elif invmec and not inrpz:
            s = x
            theta = y
            zeta = z
        else:
            print("Cannot set both invmec and inrpz, silly")
            return

        #interp the rz grids
        self.interp_val(s, fourier='r')
        self.interp_val(s, fourier='z')
        self.interp_val(s, fourier='l')
        self.interp_val(s, fourier='b')

        if self.iotaspl == None:
             self.iotaspl = interp.CubicSpline(self.s, self.iota)
        iota = self.iotaspl(s)
        phi = np.linspace(zeta, zeta+phimax, npoints)
        thetastar = phi*iota + theta
        theta = np.zeros(npoints)
        modB = np.zeros(npoints)
        if not onlymodB:
            rarr = np.zeros(npoints)
            zarr = np.zeros(npoints)

        def theta_solve(x):
            lam = sum(self.linterp*np.sin(self.xm*x - self.xn*phi[i]))
            return x + lam - thetastar[i]

        for i in range(npoints):
            theta[i] = fsolve(theta_solve, thetastar[i])
            modB[i] = sum(self.binterp*np.cos(self.xmnyq*theta[i] -
                                              self.xnnyq*phi[i]))
            angle = self.xm*theta[i] - self.xn*phi[i]
            if not onlymodB:
                rarr[i] = sum(self.rinterp*np.cos(angle))
                zarr[i] = sum(self.zinterp*np.sin(angle))
                          
        if not onlymodB:
            xarr = rarr*np.cos(phi)
            yarr = rarr*np.sin(phi)

        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(xarr,yarr,zarr)
            if show:
                plt.show()
        if retrpz:
            return rarr, phi, zarr, modB
        if onlymodB:
            return phi, modB, theta
        else:
            return xarr, yarr, zarr, modB
            
        

    #This works, but there is an issue with plot display at high
    #resolution.  I have not figured out how to fix it yet
    def modb_on_surface(self, s=1, ntheta=64, nphi=64, plot=True,
                        show=False, outxyz=None, full=False, alpha=1,
                        mayavi=True):
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
        
        for phii in range(nphi):
            p = phi[phii]
            for ti in range(ntheta):
                th = theta[ti]
                r[phii,ti] = self.r_at_point(s,th,p)
                z[phii,ti] = self.z_at_point(s,th,p)
                x[phii,ti] += r[phii,ti]*np.cos(p)
                y[phii,ti] += r[phii,ti]*np.sin(p)
                b[phii,ti] = self.modb_at_point(s, th, p)
        
        my_col = cm.jet((b-np.min(b))/(np.max(b)-np.min(b)))
        
        if plot and (not use_mayavi or not mayavi):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            #my_col = cm.jet((b-np.min(b))/(np.max(b)-np.min(b)))
            
            ax.plot_surface(x,y,z,facecolors=my_col,norm=True, alpha=alpha)
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

        elif plot and use_mayavi and mayavi:
            mlab.figure(bgcolor=(1.0, 1.0, 1.0), size=(800,600))
            mlab.contour3d(x,y,z, b)
            if show:
                mlab.show()
                
        if outxyz is not None:
            wf = open(outxyz, 'w')
            for phii in range(nphi):
                for ti in range(ntheta):
                    s = (str(x[phii,ti]) + '\t' + str(y[phii,ti]) + '\t'
                         + str(z[phii,ti]) + '\n')
                    wf.write(s)
        return [x, y, z, b]
            

    def axis(self, phi):
        r = 0
        z = 0
        for i in range(len(self.raxis)):
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
    
    def current(self, plot=True, show=False):
        s = self.psi[1:]/self.psi[-1]
        jdotb = self.jdotb[1:]
        if plot:
            plt.plot(s, jdotb)
            if show:
                plt.show()
        return s,jdotb

    def r_at_point(self, s, theta, phi):
        self.interp_val(s,fourier='r')
        return sum(self.rinterp*np.cos(self.xm*theta - self.xn*phi))
    
    def z_at_point(self, s, theta, phi):
        self.interp_val(s,fourier='z')
        return sum(self.zinterp*np.sin(self.xm*theta - self.xn*phi))

    #interpolation on the half grid
    def interp_half(self, val, s, mn):
        if s < self.shalf[1]:
            v = val[1,mn]
        elif s > self.shalf[-1]:
            v = val[-1,0]
        else:
            vfunc = interp.interp1d(self.shalf,val[:,mn])
            v = vfunc(s)
        return v


    # return dvds, the volume derivative, which is 4 pi^2 abs(g_00). 
    def dvds(self, s, interpolate=False):        
        if not interpolate:
            # if we don't want to interpolate, then get an actual value
            fs = self.s2fs(s)
            g = self.gmnc[fs,0]
        else:
            g = self.interp_half(self.gmnc, s, 0)
            
        dvds_val = abs(4 * np.pi**2 * g)
        return dvds_val
        
    def well(self, s):
        #interpolate for bmn and gmn
        bslice = np.empty(self.nmn)
        gslice = np.empty(self.nmn)
        for mn in range(self.nmn):
            bslice[mn] = self.interp_half(self.bmnc, s, mn)
            gslice[mn] = self.interp_half(self.gmnc, s, mn)

        vol = 4*np.pi**2 * abs(gslice[0])
        #print(vol)
        #print gslice
        def bsqfunc(th, ze):
            b = 0
            for mn in range(self.nmn):
                b += bslice[mn]*np.cos(self.xm[mn]*th - self.xn[mn]*ze)
            return b*b

        def gfunc(th, ze):
            g = 0
            for mn in range(self.nmn):
                g += gslice[mn]*np.cos(self.xm[mn]*th - self.xn[mn]*ze)
            return abs(g)

        def bsqg(th, ze):
            return bsqfunc(th,ze)*gfunc(th,ze)

        #get flux surface average B**2
        val, err = integrate.dblquad(bsqg, 0, 2*np.pi, lambda x: 0, lambda x: 2*np.pi)
        return val/vol
        #print bslice[0]**2
    

    #simple vacuum well, uses B_00 as <B> which isn't quite right
    def well_simp(self, s):
        b00_spl = interp.CubicSpline(self.shalf, self.bmnc[:,0])
        g00_spl = interp.CubicSpline(self.shalf,
                                               4*np.pi**2 * abs(self.gmnc[:,0]))
        vol_spl = g00_spl.antiderivative()
        db00_spl = b00_spl.derivative()

        #print some values
        print(vol_spl(s))
        print(b00_spl(s))
        print(db00_spl(s))

        svals = np.linspace(0,1,51)
        plt.plot(svals, b00_spl(svals))
        #plt.plot(svals, db00_spl(svals))
        plt.show()


    def interp_val(self, s, fourier='b'):
        if fourier=='b':
            if self.interpb_at == s:
                return
            for i in range(self.nmnnyq):
                bspl = interp.CubicSpline(self.shalf, self.bmnc[:,i])
                self.interpb_at = s
                self.binterp[i] = bspl(s)
        elif fourier=='r':
            if self.interpr_at == s:
                return
            for i in range(self.nmn):
                bspl = interp.CubicSpline(self.s, self.rmnc[:,i])
                self.interpr_at = s         
                self.rinterp[i] = bspl(s)
        elif fourier=='z':
            if self.interpz_at == s:
                return
            for i in range(self.nmn):
                bspl = interp.CubicSpline(self.s, self.zmns[:,i])
                self.interpz_at = s
                self.zinterp[i] = bspl(s)
        elif fourier=='l':
            if self.interpl_at == s:
                return
            for i in range(self.nmn):
                bspl = interp.CubicSpline(self.s, self.lmns[:,i])
                self.interpl_at = s
                self.linterp[i] = bspl(s)
        else:
                print('wrong value passed to interp_val')

        
    #convert vmec to cylindrical 
    def vmec2rpz(self, s, theta, zeta):
        #interpolate the rmnc, and zmns arrays
        if self.interpr_at != s:
            self.interp_val(s, fourier='r')
        if self.interpz_at != s:
            self.interp_val(s, fourier='z')

        
        angle = self.xm*theta - self.xn*zeta
        r = sum(self.rinterp*np.cos(angle))
        z = sum(self.zinterp*np.sin(angle))

        return r,zeta,z                  
        

    def vmec2xyz(self,s,theta,zeta):
        r,phi,z = self.vmec2rpz(s,theta,zeta)
        x = r*np.cos(phi)
        y = r*np.sin(phi)
        return x,y,z

    def xyz2vmec(self, x, y, z):
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y,x)
        return self.rpz2vmec(r, phi, z)

    def rpz2vmec(self,r,phi,z):
        vmec_vec = np.empty(3)
        vmec_vec[2] = phi
        #we don't know what flux surface we're on so we can't really
        #make use of the lambda variable
        thguess = self.thetaguess(r,phi,z)
        vmec_vec[0] = self.sguess(r,phi,z,thguess)
        vmec_vec[1] = thguess

        #this function takes a numpy array vmec_coords
        #and returns a float representing the difference
        def solve_function(vmec_coords):
            
            vmec_coords[0] = abs(vmec_coords[0])
            rg, pg, zg = self.vmec2rpz(vmec_coords[0], vmec_coords[1],
                                       vmec_coords[2])
            #p is by definition correct, so we just look at r and z
            ans = 0
            ans += (r-rg)**2
            #ans += (y-yg)**2
            ans += (z-zg)**2
            return np.sqrt(ans)

        # set bounds for s, theta and zeta
        bounds = ((0.0,1.0),(vmec_vec[1]-np.pi/2,vmec_vec[1]+np.pi/2),
                  (vmec_vec[2]-0.001,vmec_vec[2]+0.001))

        sol = minimize(solve_function, vmec_vec, method='L-BFGS-B',tol = 1.E-8,
                       bounds=bounds)

        s = sol.x[0]
        mins = 1.0/(self.ns*3)
        maxs = 1.0-mins
        if s < mins:
            print("warning: s value of ",s," is too low, answer may be incorrect")
        if s > maxs:
            print("warning: s value of ",s," is too high, answer may be incorrect")

        return sol.x

    def thetaguess(self, r, phi, z):
        r0, phi0, z0 = self.vmec2rpz(0,0,phi)
        r1, phi1, z1 = self.vmec2rpz(1,0,phi)

        #get relative r and z for plasma and our point
        r_pl = r1 - r0
        z_pl = z1 - z0

        r_pt = r - r0
        z_pt = z - z0

        #get theta for plasma and our point
        th_pl = np.arctan2(z_pl, r_pl)
        th_pt = np.arctan2(z_pt, r_pt)

        return th_pt - th_pl

    def sguess(self, r, phi, z, theta, r0=None, z0=None):
        #if axis is not around, get it
        if r0 is None:
            r0, phi0, z0 = self.vmec2rpz(0,0,phi)

        #get r and z at lcfs
        r1, phi1, z1 = self.vmec2rpz(1, theta, phi)

        #squared distances for plasma minor radius and our point at theta
        d_pl = (r1 - r0)**2 + (z1 - z0)**2
        d_pt = (r - r0)**2 + (z - z0)**2

        #s guess is normalized radius squared
        return d_pt/d_pl
        
