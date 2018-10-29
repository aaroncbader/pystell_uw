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
        self.mnmodes = len(self.xm)
        self.interp_at = -1
        self.charge = 1.602E-19

    def interp_bmn(self, s):
        self.binterp = np.empty(self.mnmodes)
        self.dbdpsi = np.empty(self.mnmodes)
        for i in xrange(self.mnmodes):
            bspl = interp.UnivariateSpline(self.sr, self.bmnc[:,i])
            #self.binterp[i] = interp.griddata(self.sr,self.bmnc[:,i],
            #                                  s,method='cubic')
            self.binterp[i] = bspl(s)
            #print bspl.derivatives(s)
            #self.dbdpsi[i] = bspl.derivatives(s)
        self.interp_at = s
            
            
    def field_at_point(self, s,theta,zeta):
        #make sure we've interpolated at the desired value
        if self.interp_at != s:
            self.interp_bmn(s)
        b = 0
        for i in xrange(self.mnmodes):
            if self.xn[i] > 5 or self.xm[i] > 5:
                continue
            angle = self.xm[i]*theta - self.xn[i]*zeta
            b += self.binterp[i] * np.cos(angle)
        return b

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

    def make_modb_contour(self, s, ntheta, nzeta):
        theta = np.linspace(0,2*np.pi,ntheta)
        zeta = np.linspace(0,2*np.pi,nzeta)
        b = np.empty([ntheta,nzeta])
        for i in xrange(nzeta):           
            for j in xrange(ntheta):
                b[j,i] = self.field_at_point(s, theta[j], zeta[i])
            print zeta[i], theta[j], b[j,i]    
                

        plt.contour(zeta, theta, b, 20)
        plt.colorbar()
        plt.show()

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

