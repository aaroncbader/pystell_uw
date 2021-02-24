#Author: Aaron Bader, UW-Madison 2018
#
#This class will open and read a boozmn file created by the 
#fortran code xbooz_xform.
#
#It can also plot various quantities of interest

from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from scipy.optimize import minimize


class boozer:
    def __init__(self, fname):
        self.data = Dataset(fname)
        self.bmnc = np.array(self.data.variables['bmnc_b'][:])
        self.rmnc = np.array(self.data.variables['rmnc_b'][:])
        self.zmns = np.array(self.data.variables['zmns_b'][:])
        self.pmns = np.array(self.data.variables['pmns_b'][:])
        self.gmn = np.array(self.data.variables['gmn_b'][:])
        self.xm =  np.array(self.data.variables['ixm_b'][:])
        self.xn =  np.array(self.data.variables['ixn_b'][:])
        self.phi = np.array(self.data.variables['phi_b'][:])
        self.nfp = np.array(self.data.variables['nfp_b'])
        self.It = np.array(self.data.variables['bvco_b'])
        self.Ip = np.array(self.data.variables['buco_b'])
        self.s = self.phi/self.phi[-1]
        self.nrbooz =  len(self.bmnc) 
        self.nr = len(self.phi)
        self.sr = self.s[self.nr-self.nrbooz:]
        #sr is on the half grid
        self.sr = self.sr - self.sr[0]/2 
        self.mnmodes = len(self.xm)
        self.interpb_at = -1
        self.binterp = np.empty(self.mnmodes)
        self.interpr_at = -1
        self.rinterp = np.empty(self.mnmodes)
        self.interpz_at = -1
        self.zinterp = np.empty(self.mnmodes)
        self.interpp_at = -1
        self.pinterp = np.empty(self.mnmodes)
        self.charge = 1.602E-19

    #convert a boozer s, theta, zeta to r,z,phi
    def booz2rpz(self, s, theta, zeta):
        
        #get the r value
        r = self.field_at_point(s, theta, zeta, fourier='r')
        #get the z value
        z = self.field_at_point(s, theta, zeta, fourier='z')
        #get the phi value
        phi = zeta + self.field_at_point(s, theta, zeta, fourier='p')
        return r, phi, z

    def booz2xyz(self, s, theta, zeta):
        r,phi,z = self.booz2rpz(s, theta, zeta)
        x = r*np.cos(phi)
        y = r*np.sin(phi)
        return x,y,z

    #convert x,y,z coordinates to boozer coordinates
    def xyz2booz(self, x, y, z):
        
        #convert to polar
        #r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y,x)
        r = np.sqrt(x**2 + y**2)

        #set up the booz vector guess
        booz_vec = np.empty(3)

        #get guesses for theta and s
        thguess = self.thetaguess(r, phi, z)
        booz_vec[0] = self.sguess(r, phi, z, thguess)
        booz_vec[1] = thguess
        booz_vec[2] = phi

        #this function takes a numpy array booz_coords
        #and returns a float representing the difference
        def solve_function(booz_coords):
            
            booz_coords[0] = abs(booz_coords[0])
            xg, yg, zg = self.booz2xyz(booz_coords[0], booz_coords[1],
                                       booz_coords[2])
            ans = 0
            ans += (x-xg)**2
            ans += (y-yg)**2
            ans += (z-zg)**2
            return np.sqrt(ans)
        
        # set bounds for s, theta and zeta
        bounds = ((0.0,1.0),(booz_vec[1]-np.pi/2,booz_vec[1]+np.pi/2),
                  (booz_vec[2]-np.pi/2,booz_vec[2]+np.pi/2))

        sol = minimize(solve_function, booz_vec, method='L-BFGS-B',tol = 1.E-8,
                       bounds=bounds)

        s = sol.x[0]
        mins = 1.0/(self.nr*3)
        maxs = 1.0-mins
        if s < mins:
            print("warning: s value of ",s," is too low, answer may be incorrect")
        if s > maxs:
            print("warning: s value of ",s," is too high, answer may be incorrect")

        return sol.x
                            
    
        
        
    def interp_bmn(self, s, fourier='b'):
        
        
        #self.dbdpsi = np.empty(self.mnmodes)
        for i in range(self.mnmodes):
            if fourier=='b':
                bspl = interp.UnivariateSpline(self.sr, self.bmnc[:,i])
                self.interpb_at = s
                self.binterp[i] = bspl(s)
            elif fourier=='r':
                bspl = interp.UnivariateSpline(self.sr, self.rmnc[:,i])
                self.interpr_at = s
                self.rinterp[i] = bspl(s)
            elif fourier=='z':
                bspl = interp.UnivariateSpline(self.sr, self.zmns[:,i])
                self.interpz_at = s
                self.zinterp[i] = bspl(s)
            elif fourier=='p':
                bspl = interp.UnivariateSpline(self.sr, self.pmns[:,i])
                self.interpp_at = s
                self.pinterp[i] = bspl(s)
            else:
                print('wrong value passed to interp_bmn')


            
    # calculate the b,r or z value at a given point, default is b
    def field_at_point(self, s,theta,zeta,fourier='b'):
        #make sure we've interpolated at the desired value
        if fourier=='b' and self.interpb_at != s:
            self.interp_bmn(s, fourier='b')
        elif fourier =='r' and self.interpr_at != s:
            self.interp_bmn(s, fourier='r')
        elif fourier =='z' and self.interpz_at != s:
            self.interp_bmn(s, fourier='z')
        elif fourier =='p' and self.interpp_at != s:
            self.interp_bmn(s, fourier='p')
            
        
        angle = self.xm*theta - self.xn*zeta
        if fourier == 'b':
            v = np.sum(self.binterp*np.cos(angle))
        elif fourier == 'r':
            v = np.sum(self.rinterp*np.cos(angle))
        elif fourier == 'z':
            v = np.sum(self.zinterp*np.sin(angle))
        elif fourier == 'p':
            v = np.sum(self.pinterp*np.sin(angle))
        
        return v

    def currents_and_derivs(self, s):
        ipspl = interp.UnivariateSpline(self.s, self.Ip)
        ip = np.empty(2)
        ip[0] = ipspl(s)
        ip[1] = ipspl.derivatives(s)
        itspl = interp.UniveraiteSpline(self.s, self.It)
        it = np.empty(2)
        it[0] = itspl(s)
        it[1] = itspl.derivatives(s)
        return ip,it

    # return b as (modb, dbdpsi, dbdtheta, dbdzeta)
    def field_and_derivs(self, s,theta,zeta):
        if self.interp_at != s:
            self.interp_bmn(s)
        b = np.zeros(4)
        for i in range(self.mnmodes):
            #if self.xn[i] > 5 or self.xm[i] > 5:
            #    continue
            angle = self.xm[i]*theta - self.xn[i]*zeta
            b[0] += self.binterp[i] * np.cos(angle)
            #b[1] += self.dbdpsi[i] * np.cos(angle)
            b[2] += -self.xm[i] * self.binterp[i] * np.sin(angle)
            b[3] += self.xn[i] * self.binterp[i] * np.sin(angle)
        return b

    def dpsidt(self,s,theta,zeta):
        #simple calculation doesn't need this
        #ip, it = self.currents_and_derivs(s)
        b = self.field_and_derivs(s, theta, zeta)
        # we need to know rho_parallel, or do the bounce average
        #gamma = self.charge*(
        return b[2]

    def make_modb_contour(self, s, ntheta, nzeta, plot = True, show = False):
        theta = np.linspace(0,2*np.pi,ntheta)
        zeta = np.linspace(0,2*np.pi,nzeta)
        b = np.empty([ntheta,nzeta])
        for i in range(nzeta):           
            for j in range(ntheta):
                b[j,i] = self.field_at_point(s, theta[j], zeta[i])
            #print zeta[i], theta[j], b[j,i]    
                
        if plot:
            plt.contourf(zeta, theta, b, 60, cmap='jet')
            #plt.colorbar()
            if show:
                plt.show()
        return [theta, zeta, b]

    def make_dpsidt_contour(self, s, ntheta, nzeta):
        theta = np.linspace(0,2*np.pi,ntheta)
        zeta = np.linspace(0,2*np.pi,nzeta)
        psidot = np.empty([ntheta,nzeta])
        for i in range(nzeta):           
            for j in range(ntheta):
                psidot[j,i] = self.dpsidt(s,theta[j],zeta[i])
            #print zeta[i], theta[j], psidot[j,i]
        plt.contour(zeta, theta, psidot, 20)
        plt.colorbar()
        plt.show()


    #guess for s given a point in r,z,phi and a guess for theta
    def sguess(self, r, phi, z, theta, r0=None, z0=None):
        #if axis is not around, get it
        if r0 is None:
            r0, phi0, z0 = self.booz2rpz(0,0,phi)

        #get r and z at lcfs
        r1, phi1, z1 = self.booz2rpz(1, theta, phi)

        #squared distances for plasma minor radius and our point at theta
        d_pl = (r1 - r0)**2 + (z1 - z0)**2
        d_pt = (r - r0)**2 + (z - z0)**2

        #s guess is normalized radius squared
        return d_pt/d_pl

    #Give a guess for theta by considering the axis and LCFS
    #at zeta = phi
    def thetaguess(self, r, phi, z):
        r0, phi0, z0 = self.booz2rpz(0,0,phi)
        r1, phi1, z1 = self.booz2rpz(1,0,phi)

        #get relative r and z for plasma and our point
        r_pl = r1 - r0
        z_pl = z1 - z0

        r_pt = r - r0
        z_pt = z - z0

        #get theta for plasma and our point
        th_pl = np.arctan2(z_pl, r_pl)
        th_pt = np.arctan2(z_pt, r_pt)

        return th_pt - th_pl

    # plot the largest boozer modes,
    # fs is the surface to compare at
    # n is the number of modes to plot
    # rovera is whether to plot wrt r/a or s
    # ignore0 is whether to ignore the B00 mode
    def plot_largest_modes(self, fs=-1, n=10, rovera=True, ignore0=True,
                           show=True, ax=None, xaxis=None, noqa = False):
        # get sorting index for the desired slice
        bslice = self.bmnc[fs,:]
        bslice = -1*abs(bslice)
        sortvals = np.argsort(bslice)

        #decide whether to plot vs r over a, or s
        if rovera:
            xaxis = np.sqrt(self.sr)
        else:
            xaxis = self.sr

        if ignore0:
            startval = 1
        else:
            startval = 0


        leg = [] #legend
        
        #now plot the 10 largest
        if ax is None:
            for i in range(startval,n+startval):
                if noqa and self.xn[sortvals[i]] == 0:
                    continue
                plt.plot(xaxis, self.bmnc[:,sortvals[i]])
                legs = ('n=' + str(self.xn[sortvals[i]]) +
                        ', m=' + str(self.xm[sortvals[i]]))
                leg.append(legs)
            plt.legend(leg)
            if rovera:
                plt.xlabel('$r/a$')
            else:
                plt.xlabel('$s$')
            plt.ylabel('$B_{mn}$')
        else:
            for i in range(startval,n+startval):
                ax.plot(xaxis, self.bmnc[:,sortvals[i]])
                legs = ('n=' + str(self.xn[sortvals[i]]) +
                        ', m=' + str(self.xm[sortvals[i]]))
                leg.append(legs)
            ax.legend(leg)
            if rovera:
                ax.set_xlabel('$r/a$')
            else:
                ax.set_xlabel('$s$')
            ax.set_ylabel('$B_{mn}$')
                
        if show:
            plt.show()
    

            
#bz = boozer('boozmn_qhgc.nc')
#bz = boozer('boozmn_qhs46_mn8.nc')
#bz.make_modb_contour(0.5,51,51)
#bz.make_dpsidt_contour(0.5, 81, 81)

