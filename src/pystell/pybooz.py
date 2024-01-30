#Author: Aaron Bader, UW-Madison 2020
#A class to convert vmec to boozer coordinates
#Will convert every flux surface except for the core
#Resolution is passed in the class call along with 
#a vmec_data object (see read_vmec)

from netCDF4 import Dataset
from netCDF4 import stringtochar
import numpy as np
import matplotlib.pyplot as plt
import imp
from matplotlib import cm
import read_vmec

class vmec2booz:
    def __init__(self, vmecdata, nboz, mboz):
        self.nboz = int(nboz)
        self.mboz = int(mboz)
        self.mnboz = nboz+1 + (mboz-1)*(1+(2*nboz))

        #data from the vmec file
        self.nfp = vmecdata.nfp
        self.mpol = vmecdata.mpol
        self.ntor = vmecdata.ntor
        #For now set nyq n and nyq m same as ntor and mpol, since
        #that's true for most wout files, and these values aren't
        #easily extracted from the wout
        self.xmnyq = vmecdata.xmnyq  
        self.xnnyq = vmecdata.xnnyq
        self.mnyq = vmecdata.mnyq
        self.nnyq = vmecdata.nnyq
        self.nmnnyq = vmecdata.nmnnyq
        self.ns = vmecdata.ns
        #self.sfull = vmecdata.s
        self.rmnc = vmecdata.rmnc
        self.zmns = vmecdata.zmns
        self.lmns = vmecdata.lmns
        self.bmodmnc = vmecdata.bmnc
        self.xm = vmecdata.xm
        self.xn = vmecdata.xn
        self.nmn = vmecdata.nmn
        self.mnmax = self.nmn
        self.hiota = vmecdata.hiota

        self.aspect = vmecdata.aspect
        self.rmax = vmecdata.rmax_surf
        self.rmin = vmecdata.rmin_surf
        self.zmax = vmecdata.zmax_surf
        self.betaxis = vmecdata.betaxis
        self.pres = vmecdata.pres
        self.betavol = vmecdata.betavol
        self.psi= vmecdata.psi
        self.psips = vmecdata.psips
        self.bvco = vmecdata.bvco
        self.buco = vmecdata.buco
        self.lasym = 0#eventually allow for this

        #boozer outputs (only symmetric for now)
        self.bmncb = np.zeros([self.mnboz, self.ns])
        self.rmncb = np.zeros([self.mnboz, self.ns])
        self.zmnsb = np.zeros([self.mnboz, self.ns])
        self.pmnsb = np.zeros([self.mnboz, self.ns])
        self.gmncb = np.zeros([self.mnboz, self.ns])


        #This section is adapted from setup_booz.f
        #make the xnb and xmb arrays
        self.xnb = np.empty(self.mnboz)
        self.xmb = np.empty(self.mnboz)

        self.setup_xmnb()
        print(self.xnb)
        print(self.xmb)
        self.nu_boz = 2*(2*mboz+1)
        self.nv_boz = 2*(2*nboz+1)
        self.nu2_b = self.nu_boz//2 + 1
        self.nv2_b = self.nv_boz//2 + 1
        #scale factor for Fourier transform normalization
        self.fac = 2.0/((self.nu2_b-1) * self.nv_boz) #\todo add lasym
        self.scl = np.empty(self.mnboz)
        self.scl[1:] = self.fac
        self.scl[0] = self.fac/2 #the m=0, n=0 mode is this the first one

        print(self.nu2_b, self.nv2_b)

        #ntorsum appears to be the indices of all modes with m=1
        #NOTE: this is wrong!
        self.ntorsum = (nboz+1, 3*nboz + 2)
        print("ntorsum: ",self.ntorsum)
        self.ohs = vmecdata.ns - 1
        self.hs = 1.0/self.ohs
        #self.sfull = np.empty(self.ns)
        #self.sfull[0] = 0.0
        #for i in range(1,self.ns):
        #    self.sfull[i] = np.sqrt(self.hs*i)
        self.sfull = np.sqrt(vmecdata.s)

        self.nu3_b = self.nu2_b #\todo fix for lasym
        self.nunv = self.nu3_b*self.nv_boz

        print(self.nu3_b, self.nv_boz)

        #make theta and zeta grids (in foranl.f)
        #for now just copy the inefficient fortran way
        dzt = 2*np.pi/(self.nv_boz*self.nfp)
        dth = 2*np.pi/(2*(self.nu3_b-1))
        lk = 0
        #print dzt, dth
    
        self.thgrid = np.empty(self.nunv)
        self.ztgrid = np.empty(self.nunv)
        for lt in range(1, self.nu3_b+1):
            for lz in range(1, self.nv_boz+1):
                self.thgrid[lk] = (lt-1)*dth
                self.ztgrid[lk] = (lz-1)*dzt
                lk += 1


        cosm, sinm, cosn, sinn = self.trigfunc(self.thgrid, self.ztgrid, 
                        self.mpol-1, self.ntor, self.nunv)
        self.cosm_b = np.copy(cosm)
        self.sinm_b = np.copy(sinm)
        self.cosn_b = np.copy(cosn)
        self.sinn_b = np.copy(sinn)
        



        cosm, sinm, cosn, sinn = self.trigfunc(self.thgrid, self.ztgrid, 
                        self.mnyq, self.nnyq, self.nunv)
        self.cosm_nyq = np.copy(cosm)
        self.sinm_nyq = np.copy(sinm)
        self.cosn_nyq = np.copy(cosn)
        self.sinn_nyq = np.copy(sinn)
        #By this point we are done with foranl.f                      

        #From here we start with the surfaces again ignore asym options
        #Note that jrad is an index, and python indexes at 0 not 1
        #So jrad will be one less than in booz_xform.f
        for jrad in range(1,self.ns):
            bsubtmnc = vmecdata.bsubumnc[jrad,:]
            bsubzmnc = vmecdata.bsubvmnc[jrad,:]
            pmns = np.empty(self.nmnnyq)
            gpsi = np.empty(self.ns)
            gpsi[:] = 0.0
            Ipsi = np.empty(self.ns)
            Ipsi[:] = 0.0
            
            self.transpmn(pmns, bsubtmnc, bsubzmnc, gpsi, Ipsi, jrad)

                        

            #in booz_xform these are all nunv sized even though
            #that size is larger than nmn, or nmnnyq
            #so there are 0 values in them at the end?
            r1 = np.empty(self.nunv)
            z1 = np.empty(self.nunv)
            rodd = np.empty(self.nunv)
            zodd = np.empty(self.nunv)
            lt = np.empty(self.nunv)
            lz = np.empty(self.nunv)
            wt = np.empty(self.nunv)
            wz = np.empty(self.nunv)
            wp = np.empty(self.nunv)
            lam = np.empty(self.nunv)
            bmod_b = np.empty(self.nunv)
            xjac = np.empty(self.nunv)
            xjac[:] = 0.0
            p1 = np.empty(self.nunv)
            q1 = np.empty(self.nunv)

            r1, z1, lt, lz, lam = self.vcoords_rz(jrad, r1, z1, lt, lz, 
                                  lam, nparity=0)
                        
            rodd, zodd, lt, lz, lam = self.vcoords_rz(jrad, rodd, zodd,
                                  lt, lz, lam, nparity=1)

                


            wt, wz, wp = self.vcoords_w(jrad, pmns, wt, wz, wp, bmod_b)

            jacfac, p1, q1, xjac = self.harfun(gpsi, Ipsi, jrad, 
                                   lt, lz, lam, wt, wz, wp)
            if jrad == 9:
                print("p1: ",p1[:10])
                print("q1: ",q1[:10])
                print("xjac: ",xjac[:10])
 
              
            r12 = np.empty(self.nunv)
            z12 = np.empty(self.nunv)

            r12, z12 = self.booz_rzhalf(r1, z1, rodd, zodd, r12, z12,
                                        jrad, nrep = 1)


            #checked agreement up to here
            self.cosmm = None
            self.cosnn = None
            self.sinmm = None
            self.sinnn = None

            
            #cosmm etc. is returned here but appears to be overwritten below?

            self.boozerfun(bmod_b, r12, z12, p1, q1, xjac, jacfac, jrad)
            self.cosmm[:] = 0.0
            self.cosnn[:] = 0.0
            self.sinmm[:] = 0.0
            self.sinnn[:] = 0.0

            if jrad == 2:
                print("bmncb: ",self.bmncb[:10,jrad])
                print("rmncb: ",self.rmncb[:10,jrad])
                print("zmnsb: ",self.zmnsb[:10,jrad])
                print("pmnsb: ",self.pmnsb[:10,jrad])
                print("gmncb: ",self.gmncb[:10,jrad])

            
            # We store angles corresponding to 0 and pi, have to subtract 1
            # due to the python/fortran index difference
            self.u_b = np.empty(4)
            self.v_b = np.empty(4)
            piv = self.ztgrid[self.nv2_b - 1]
            
            self.u_b[0] = p1[0]
            self.v_b[0] = q1[0]
            self.u_b[2] = p1[self.nv2_b-1]
            self.v_b[2] = piv + q1[self.nv2_b-1]
            i1 = self.nv_boz*(self.nu2_b-1)
            piu = self.thgrid[i1]
            self.u_b[1] = piu+p1[i1]
            self.v_b[1] = q1[i1]
            
            i1 = self.nv2_b-1 + self.nv_boz*(self.nu2_b-1)
            self.u_b[3] = piu+p1[i1]
            self.v_b[3] = piv+q1[i1]
            
            #if jrad == 9:
            #    print("u_b",self.u_b)
            #    print("v_b",self.v_b)

            bmodb = self.modbooz(jrad)
            #if jrad == 9:
            #    print('bmodb',bmodb)


    #Set up the arrays of xnb and xmb, these
    #are the boozer equivalents to the vmec
    #xm, xn, or xm_nyq, xn_nyq arrays
    def setup_xmnb(self):
        n2 = self.nboz
        mnboz0 = 0
        for m in range(self.mboz):
            n1 = -self.nboz
            if m == 0:
                n1 = 0
            for n in range(n1, n2+1):
                if mnboz0 >= self.mnboz:
                    print("problem: mnboz0 > mnboz")
                    return
                self.xnb[mnboz0] = n*self.nfp
                self.xmb[mnboz0] = m
                mnboz0 += 1

    def trigfunc(self, th, zt, mpol, ntor, nznt):

        cosm = np.empty([nznt, mpol+1])
        sinm = np.empty([nznt, mpol+1])
        cosn = np.empty([nznt, ntor+1])
        sinn = np.empty([nznt, ntor+1])
        
        cosm[:,0] = 1
        sinm[:,0] = 0
        cosm[:,1] = np.cos(th)
        sinm[:,1] = np.sin(th)

        for m in range(2,mpol+1):
            cosm[:,m] = cosm[:,m-1]*cosm[:,1] - sinm[:,m-1]*sinm[:,1]
            sinm[:,m] = sinm[:,m-1]*cosm[:,1] + cosm[:,m-1]*sinm[:,1]
  
        cosn[:,0] = 1
        sinn[:,0] = 0
        if ntor > 1:
            cosn[:,1] = np.cos(zt*self.nfp)
            sinn[:,1] = np.sin(zt*self.nfp)

        for n in range(2,ntor+1):
            cosn[:,n] = cosn[:,n-1]*cosn[:,1] - sinn[:,n-1]*sinn[:,1]
            sinn[:,n] = sinn[:,n-1]*cosn[:,1] + cosn[:,n-1]*sinn[:,1]

        return cosm, sinm, cosn, sinn

    def transpmn(self, pmns, bsubtmnc, bsubzmnc, gpsi, Ipsi, jrad):

        for mn in range(self.nmnnyq):
            if int(self.xmnyq[mn]) != 0:
                pmns[mn] = bsubtmnc[mn]/self.xmnyq[mn]
            elif int(self.xnnyq[mn]) != 0:
                pmns[mn] = -bsubzmnc[mn]/self.xnnyq[mn]
            else:
                pmns[mn] = 0
                gpsi[jrad] = bsubzmnc[mn]
                Ipsi[jrad] = bsubtmnc[mn]


    def vcoords_rz(self, jrad, r, z, lt, lz, lam, nparity = 0):
        js = jrad
        js1 = js - 1
        if (js <= 0):
            print('Something wrong js <= 0')
            return
        r = 0.0
        z = 0.0
        if nparity == 0:
            t1 = 1.0
            t2 = 1.0
            lt[:] = 0.0
            lz[:] = 0.0
            lam[:] = 0.0
        elif js > 1: # js > 2 in booz_xform
            t1 = 1.0/self.sfull[js]
            t2 = 1.0/self.sfull[js1]
        else:
            t1 = 1.0/self.sfull[1]
            t2 = 1.0
            #This section modifies the rmnc components of the m=1 modes
            #for some reason???
            #
            #Maybe ask John if he understands what's going on in this section
            # lines 55-61 of vcoords.f
            i1 = self.ntorsum[0]
            i2 = self.ntorsum[1]
            self.rmnc[0, i1:i2] = (2 * self.rmnc[1, i1:i2]/self.sfull[1] - 
                                   self.rmnc[2,i1:i2]/self.sfull[2])
            self.zmns[0, i1:i2] = (2 * self.zmns[1, i1:i2]/self.sfull[1] - 
                                   self.zmns[2,i1:i2]/self.sfull[2])

        t1 = t1/2
        t2 = t2/2

        for mn in range(0,self.mnmax):
            m = int(self.xm[mn])
            if m % 2 != nparity:
                continue
            n = int(abs(self.xn[mn]/self.nfp))
            sgn = np.sign(self.xn[mn])
            tcos = (self.cosm_b[:,m]*self.cosn_b[:,n] +
                    self.sinm_b[:,m]*self.sinn_b[:,n]*sgn)
            tsin = (self.sinm_b[:,m]*self.cosn_b[:,n] -
                    self.cosm_b[:,m]*self.sinn_b[:,n]*sgn)

            #tcos,tsin verified against fortran
            #print 'tcos',tcos[:20]
            #print 'tsin',tsin[:20]
            
            rc = t1*self.rmnc[js,mn]+t2*self.rmnc[js1,mn]
            zs = t1*self.zmns[js,mn]+t2*self.zmns[js1,mn]
            r = r+tcos*rc
            z = z+tsin*zs
            lt += tcos*self.lmns[js,mn] * m
            lz -= tcos*self.lmns[js,mn] * self.xn[mn]
            lam += tsin*self.lmns[js,mn]
        return r,z,lt,lz,lam

    def vcoords_w(self, jrad, pmns, wt, wz, w, bmod):
        wt[:] = 0
        wz[:] = 0
        w[:] = 0
        bmod[:] = 0
        for mn in range(self.nmnnyq):

            m = int(self.xmnyq[mn])
            n = int(abs(self.xnnyq[mn]/self.nfp))
            sgn = np.sign(self.xnnyq[mn])
            tcos = (self.cosm_nyq[:,m]*self.cosn_nyq[:,n] +
                    self.sinm_nyq[:,m]*self.sinn_nyq[:,n] * sgn)
            tsin = (self.sinm_nyq[:,m]*self.cosn_nyq[:,n] -
                    self.cosm_nyq[:,m]*self.sinn_nyq[:,n] * sgn)
            

            w += tsin*pmns[mn]
            wt += tcos*pmns[mn]*self.xmnyq[mn]
            wz -= tcos*pmns[mn]*self.xnnyq[mn]
            bmod += tcos*self.bmodmnc[jrad, mn]
        return wt, wz, w

    def harfun(self, gpsi, ipsi, js, xlt, xlz, xl, 
               wt, wz, w):

        nznt = self.nunv
        bsupu = np.empty(nznt)
        bsupv = np.empty(nznt)
        psupu = np.empty(nznt)
        psupv = np.empty(nznt)

        jacfac = gpsi[js] + self.hiota[js]*ipsi[js]
        if (jacfac == 0):
            print("Something's wrong, jacfac = 0")
        dem = 1.0/jacfac
        gpsi1 = gpsi[js]*dem
        hiota1 = self.hiota[js]*dem
        ipsi1 = ipsi[js]*dem

        vboz = dem*w - ipsi1*xl
        uboz = xl + self.hiota[js]*vboz
        psubv = dem*wz - ipsi1*xlz
        psubu = dem*wt - ipsi1*xlt
        bsupv = 1 + xlt
        bsupu = self.hiota[js] - xlz

        xjac = bsupv * (1+psubv) + bsupu*psubu
        #dem = min(xjac)
        #dem2 = max(xjac)

        return jacfac, uboz, vboz, xjac


    def booz_rzhalf(self, r, z, rodd, zodd, r12, z12, js, nrep=1):
        
        #differs by 1 due to indexing difference in python v. fortran
        shalf = np.sqrt(self.hs * abs(js - 0.5))

        if nrep == 1:
            r12 = r + shalf*rodd
            z12 = z + shalf*zodd
        else:
            r12 = r
            z12 = z

        return r12, z12


    def boozerfun(self, bmod_b, rad, zee, uboz, vboz, xjac, jacfac, jrad):

        uang = self.thgrid + uboz
        vang = self.ztgrid + vboz
        nznt = self.nunv
        nu2 = self.nu2_b
        nv = self.nv_boz

        cosmmt, sinmmt, cosnnt, sinnnt = self.trigfunc(uang, vang, 
                self.mboz, self.nboz, nznt)
        self.cosmm = np.copy(cosmmt)
        self.cosnn = np.copy(cosnnt)
        self.sinmm = np.copy(sinmmt)
        self.sinnn = np.copy(sinnnt)


        i = nv*(nu2-1)
        imax = i+nv
        if jrad == 9:
            print("i,max,",i,imax)
        for m in range(self.mboz+1):
            self.cosmm[:nv,m] = 0.5*self.cosmm[:nv,m]
            self.cosmm[i:imax,m] = 0.5*self.cosmm[i:imax,m]
            self.sinmm[:nv,m] = 0.5*self.sinmm[:nv,m]
            self.sinmm[i:imax,m] = 0.5*self.sinmm[i:imax,m]

        bbjac = jacfac / (bmod_b * bmod_b)

        for mn in range(self.mnboz):
            m = int(self.xmb[mn])
            n = int(abs(self.xnb[mn])/self.nfp)
            sgn = np.sign(self.xnb[mn])
            cost = (self.cosmm[:,m]*self.cosnn[:,n] + 
                    self.sinmm[:,m]*self.sinnn[:,n]*sgn)*xjac
            sint = (self.sinmm[:,m]*self.cosnn[:,n] - 
                    self.cosmm[:,m]*self.sinnn[:,n]*sgn)*xjac


            self.bmncb[mn, jrad] = sum(bmod_b * cost)
            self.rmncb[mn, jrad] = sum(rad * cost)
            self.zmnsb[mn, jrad] = sum(zee * sint)
            self.pmnsb[mn, jrad] = -sum(vboz * sint)
            self.gmncb[mn, jrad] = sum(bbjac * cost)
            

        self.bmncb[:,jrad] *= self.scl
        self.rmncb[:,jrad] *= self.scl
        self.zmnsb[:,jrad] *= self.scl
        self.pmnsb[:,jrad] *= self.scl
        self.gmncb[:,jrad] *= self.scl
#        print 'bmncb',self.bmncb[:5,jrad]
#        print 'rmncb',self.rmncb[:5,jrad]
#        print 'zmnsb',self.zmnsb[:5,jrad]
#        print 'pmnsb',self.pmnsb[:5,jrad]
#        print 'gmncb',self.gmncb[:5,jrad]


    def modbooz(self, jrad):
        bmnc = self.bmncb[:,jrad]
        bmns = self.bmncb[:,jrad]
        bmod = np.zeros(4)

        self.cosmm = np.empty(self.mboz+1)
        self.sinmm = np.empty(self.mboz+1)
        self.cosnn = np.empty(self.nboz+1)
        self.sinnn = np.empty(self.nboz+1)

        for angles in range(4):
            self.cosmm[0] = 1.0
            self.sinmm[0] = 0.0
            self.cosmm[1] = np.cos(self.u_b[angles])
            self.sinmm[1] = np.sin(self.u_b[angles])

            self.cosnn[0] = 1.0
            self.sinnn[0] = 0.0
            if self.nboz >= 1:
                self.cosnn[1] = np.cos(self.v_b[angles]*self.nfp)
                self.sinnn[1] = np.sin(self.v_b[angles]*self.nfp)

            for m in range(2,self.mboz+1):
                self.cosmm[m] = (self.cosmm[m-1]*self.cosmm[1] - 
                                 self.sinmm[m-1]*self.sinmm[1])
                self.sinmm[m] = (self.sinmm[m-1]*self.cosmm[1] + 
                                 self.cosmm[m-1]*self.sinmm[1])

            for n in range(2,self.nboz+1):
                self.cosnn[n] = (self.cosnn[n-1]*self.cosnn[1] - 
                                 self.sinnn[n-1]*self.sinnn[1])
                self.sinnn[n] = (self.sinnn[n-1]*self.cosnn[1] + 
                                 self.cosnn[n-1]*self.sinnn[1])

            for mn in range(self.mnboz):
                m = int(self.xmb[mn])
                n = int(self.xnb[mn])//self.nfp
                sgn = np.sign(self.xnb[mn])
                cost = (self.cosmm[m]*self.cosnn[n] + 
                        self.sinmm[m]*self.sinnn[n]*sgn)

                bmod[angles] += bmnc[mn]*cost
                #if angles == 3 and jrad == 9:
                #    print(mn, cost, bmnc[mn])
                #lasym here
                #print 'stuff',angles,mn,bmnc[mn],cost
                
        return bmod


    def ascii_output(self, fname=None):
        if fname == None:
            fname = 'booz_out.txt'
        wf = open(fname, 'w')
        wf.write('mboz: '+str(self.mboz)+'\n')
        wf.write('nboz: '+str(self.nboz)+ '\n') 
        wf.write('mnboz: '+str(self.mnboz) +'\n')
        wf.write('xmb: ' + str(self.xmb)[1:-1] +'\n')
        wf.write('xnb: ' +str(self.xnb)[1:-1] +'\n')
        wf.write('ns: ' +str(self.ns)[1:-1] +'\n')
        wf.write('s: ' +str(self.sfull)[1:-1] +'\n')
        
        self.printvar(wf, self.bmncb,'bmnc_b')
        self.printvar(wf, self.rmncb,'rmnc_b')
        self.printvar(wf, self.zmnsb,'zmns_b')
        self.printvar(wf, self.pmnsb,'pmns_b')
        self.printvar(wf, self.gmncb,'gmnc_b')

        wf.close()

    def printvar(self, wf, data, title):
        wf.write(title+'\n')
        for j in range(1,self.ns):
            wf.write('jindex: '+str(j)+'\n')
            wf.write(str(data[:,j])[1:-1] + '\n')

    def write_boozmn(self, title):
        wncdf = Dataset(title, "w", format="NETCDF4")
        wncdf.createDimension("radius",self.ns)
        wncdf.createDimension("comput_surfs",self.ns-1)
        wncdf.createDimension("mn_mode",self.mnboz)
        wncdf.createDimension("mn_modes",self.mnboz)
        wncdf.createDimension("pack_rad",self.ns-1)
        wncdf.createDimension("nchars",11)
        wncdf.createDimension("nstrings",1)

        nfp_n = wncdf.createVariable("nfp_b","i4")
        nfp_n[:] = self.nfp
        ns_n = wncdf.createVariable("ns_b","i4")
        ns_n[:] = self.ns
        aspect_n = wncdf.createVariable("aspect_b","f8")
        aspect_n[:] = self.aspect
        rmax_n = wncdf.createVariable("rmax_b","f8")
        rmax_n[:] = self.rmax
        rmin_n = wncdf.createVariable("rmin_b","f8")
        rmin_n[:] = self.rmin
        zmax_n = wncdf.createVariable("zmax_b","f8")
        zmax_n[:] = self.zmax
        betaxis_n = wncdf.createVariable("betaxis_b","f8")
        betaxis_n[:] = self.betaxis
        mboz_n = wncdf.createVariable("mboz_b","i4")
        mboz_n[:] = self.mboz
        nboz_n = wncdf.createVariable("nboz_b","i4")
        nboz_n[:] = self.nboz
        version_n = wncdf.createVariable("version","S1",("nchars"))
        version_n[:] = stringtochar(np.array(['pybooz V1.0']))
        lasym_n = wncdf.createVariable("lasym__logical__","i4")
        lasym_n[:] = self.lasym
        iota_n = wncdf.createVariable("iota_b","f8", ("radius"))
        iota_n[:] = self.hiota
        pres_n = wncdf.createVariable("pres_b","f8", ("radius"))
        pres_n[:] = self.pres
        beta_n = wncdf.createVariable("beta_b","f8", ("radius"))
        beta_n[:] = self.betavol
        phip_n = wncdf.createVariable("phip_b","f8", ("radius"))
        phip_n[:] = self.psips
        phi_n = wncdf.createVariable("phi_b","f8", ("radius"))
        phi_n[:] = self.psi
        bvco_n = wncdf.createVariable("bvco_b","f8", ("radius"))
        bvco_n[:] = self.bvco
        buco_n = wncdf.createVariable("buco_b","f8", ("radius"))
        buco_n[:] = self.buco
        jlist = wncdf.createVariable("jlist","i4", ("comput_surfs"))
        jlist[:] = range(2,self.ns+1)#use index 1 convention for legacy
        xmb_n = wncdf.createVariable("ixm_b","i4", ("mn_mode"))
        xmb_n[:] = self.xmb
        xnb_n = wncdf.createVariable("inm_b","i4", ("mn_mode"))
        xnb_n[:] = self.xnb
        bmnc_n = wncdf.createVariable("bmnc_b","f8", ("pack_rad","mn_mode"))
        bmnc_n[:,:] = self.bmncb.transpose()[1:,:]
        rmnc_n = wncdf.createVariable("rmnc_b","f8", ("pack_rad","mn_mode"))
        rmnc_n[:,:] = self.rmncb.transpose()[1:,:]
        zmns_n = wncdf.createVariable("zmns_b","f8", ("pack_rad","mn_mode"))
        zmns_n[:,:] = self.zmnsb.transpose()[1:,:]
        pmns_n = wncdf.createVariable("pmns_b","f8", ("pack_rad","mn_mode"))
        pmns_n[:,:] = self.pmnsb.transpose()[1:,:]
        gmnc_n = wncdf.createVariable("gmn_b","f8", ("pack_rad","mn_mode"))
        gmnc_n[:,:] = self.gmncb.transpose()[1:,:]


        
        

        wncdf.close()
        
        

