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


class boozer:
    def __init__(self, fname):
        self.data = Dataset(fname)
        self.bmnc = np.array(self.data.variables['bmnc_b'][:])
        self.rmnc = np.array(self.data.variables['rmnc_b'][:])
        self.zmns = np.array(self.data.variables['zmns_b'][:])
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
        self.charge = 1.602E-19

    #convert a boozer s, theta, zeta to r,z,phi
    def booz2rzp(self, s, theta, zeta):
        #get the r value
        r = self.field_at_point(s, theta, zeta, fourier='r')
        #get the z value
        z = self.field_at_point(s, theta, zeta, fourier='z')
        #theta doesn't change
        return r, z, zeta

    def booz2xyz(self, s, theta, zeta):
        r,z,zeta = self.booz2rzp(s, theta, zeta)
        x = r*np.cos(zeta)
        y = r*np.sin(zeta)
        return x,y,z

    
        
    def interp_bmn(self, s, fourier='b'):
        
        
        #self.dbdpsi = np.empty(self.mnmodes)
        for i in xrange(self.mnmodes):
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
            else:
                print 'wrong value passed to interp_bmn'


            
    # calculate the b,r or z value at a given point, default is b
    def field_at_point(self, s,theta,zeta,fourier='b'):
        #make sure we've interpolated at the desired value
        if fourier=='b' and self.interpb_at != s:
            self.interp_bmn(s, fourier='b')
        elif fourier =='r' and self.interpr_at != s:
            self.interp_bmn(s, fourier='r')
        elif fourier =='z' and self.interpz_at != s:
            self.interp_bmn(s, fourier='z')
            
        v = 0
        for i in xrange(self.mnmodes):
            #if self.xn[i] > 5 or self.xm[i] > 5:
            #    continue
            angle = self.xm[i]*theta - self.xn[i]*zeta
            if fourier == 'b':
                v += self.binterp[i] * np.cos(angle)
            elif fourier == 'r':
                v += self.rinterp[i] * np.cos(angle)
            elif fourier == 'z':
                v += self.zinterp[i] * np.sin(angle)
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
        for i in xrange(self.mnmodes):
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

    def make_modb_contour(self, s, ntheta, nzeta, plot = True):
        theta = np.linspace(0,2*np.pi,ntheta)
        zeta = np.linspace(0,2*np.pi,nzeta)
        b = np.empty([ntheta,nzeta])
        for i in xrange(nzeta):           
            for j in xrange(ntheta):
                b[j,i] = self.field_at_point(s, theta[j], zeta[i])
            #print zeta[i], theta[j], b[j,i]    
                
        if plot:
            plt.contour(zeta, theta, b, 60)
            plt.colorbar()
            plt.show()
        return [theta, zeta, b]

    def make_dpsidt_contour(self, s, ntheta, nzeta):
        theta = np.linspace(0,2*np.pi,ntheta)
        zeta = np.linspace(0,2*np.pi,nzeta)
        psidot = np.empty([ntheta,nzeta])
        for i in xrange(nzeta):           
            for j in xrange(ntheta):
                psidot[j,i] = self.dpsidt(s,theta[j],zeta[i])
            print zeta[i], theta[j], psidot[j,i]
        plt.contour(zeta, theta, psidot, 20)
        plt.colorbar()
        plt.show()
            
#bz = boozer('boozmn_qhgc.nc')
#bz = boozer('boozmn_qhs46_mn8.nc')
#bz.make_modb_contour(0.5,51,51)
#bz.make_dpsidt_contour(0.5, 81, 81)

