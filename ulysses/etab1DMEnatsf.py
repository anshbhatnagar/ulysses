from cmath import e
from locale import MON_1
from operator import xor
from xmlrpc.client import FastMarshaller
import ulysses
import numpy as np
from scipy import interpolate
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.special import zeta
from scipy.special import k1
from scipy.special import kn
from scipy.optimize import fsolve
from scipy.integrate import quad
from ulysses.numba import jit
import matplotlib.pyplot as plt
import ulysses.numba as nb
from ulysses.ulsbase import my_kn2, my_kn1
from odeintw import odeintw

showLeptoPlot = False
showTemps = False

absErr = 5e-3 #absolute error for Ip, Jp, Kp and derivative integrals
relErr  = 5e-4 #relative error for Ip, Jp, Kp and derivative integrals
cutoff = 100 #cutoff at 'infinity' for Ip, Jp, Kp and derivative integrals

#+++++++++++++++++++++++++++++++++++++++++++++++++#
#             FLRW-Boltzmann Equations            #
#+++++++++++++++++++++++++++++++++++++++++++++++++#

def Ip(x):
    integrand = lambda z: z**2/(1+np.exp(np.sqrt(np.abs(z**2+x**2))))
    return quad(integrand, 0, cutoff, epsabs=absErr, epsrel = relErr)[0]

def dIpdx(x):
    integrand = lambda z: -x*z**2*np.exp(np.sqrt(np.abs(z**2+x**2)))/(np.sqrt(np.abs(z**2+x**2))*(1+np.exp(np.sqrt(np.abs(z**2+x**2))))**2)
    return quad(integrand, 0, cutoff, epsabs=absErr, epsrel = relErr)[0]

def Ipp(Th,zh):
    return Th**3*Ip(zh)

def dIppdTh(Th,MN):
    return 3*Th**2*Ip(MN/Th)-MN*Th*dIpdx(MN/Th)

def IppRoot(Th, MN, x):
    return Ipp(Th,MN/Th)-x

def invIpp(x, MN, guess):
    return fsolve(IppRoot,guess,args=(MN,x))[0]

def dinvIppdx(x, MN, guess):
    return 1/dIppdTh(invIpp(x,MN,guess),MN)

def Jp(x): #J+ function for energy density
    integrand = lambda z: z**2*np.sqrt(np.abs(z**2+x**2))/(1+np.exp(np.sqrt(np.abs(z**2+x**2))))
    return quad(integrand, 0, cutoff, epsabs=absErr, epsrel = relErr)[0]

def dJpdx(x): #derivative of the J+ function
    integrand = lambda z: -x*z**2*(-1+np.exp(np.sqrt(np.abs(z**2+x**2)))*(-1+np.sqrt(np.abs(z**2+x**2))))/((1+np.exp(np.sqrt(np.abs(z**2+x**2))))**2*np.sqrt(np.abs(z**2+x**2)))
    return quad(integrand, 0, cutoff, epsabs=absErr, epsrel = relErr)[0]

def Kp(x): #K+ function for pressure
    integrand = lambda z: z**4/(3*np.sqrt(np.abs(z**2+x**2))*(1+np.exp(np.sqrt(np.abs(z**2+x**2)))))
    return quad(integrand, 0, cutoff, epsabs=absErr, epsrel = relErr)[0]

def dKpdx(x): #derivative of the K+ function
    integrand = lambda z:  -x*z**4*(1+np.exp(np.sqrt(np.abs(z**2+x**2)))*(1+np.sqrt(np.abs(z**2+x**2))))/(3*(1+np.exp(np.sqrt(np.abs(z**2+x**2))))**2*np.sqrt(np.abs(z**2+x**2))**3)
    return quad(integrand, 0, cutoff, epsabs=absErr, epsrel = relErr)[0]


def showPlot(lnsf, ys, etab, Tsm, Th, nN_int, NBL, washout, source):
    if showLeptoPlot:
        plt.plot(lnsf, Th/Tsm, color='r', label=r'$\kappa$')
        plt.plot(lnsf, np.exp(3*lnsf)*np.real(ys[:,0])/nN_int, color='g', label=r'$N_N/N_N(a=1)$')
        plt.plot(lnsf, np.abs(etab)*1e9, color='b', label=r'$|\eta_B|\times 10^{9}$')
        plt.plot(lnsf, np.abs(np.real(NBL))*1e8, label=r'$N_{B-L}\times 10^{8}$')
        plt.ylim(0,2)
        
    if showTemps:
        plt.plot(lnsf, np.log10(Tsm), color='b', label=r'$T_{SM}$')
        #plt.plot(lnsf, np.log10(np.real(ys[:,0])), color='g', label=r'$N_N$')
        plt.plot(lnsf, np.log10(Th), color='r', label=r'$T_H$')
    #plt.title("$\kappa(a=1)=1$, $\log_{10}(m_1)=-1$")
    plt.xlabel(r"$\ln(a)$", fontsize=16)
    plt.legend(loc='upper right', fontsize=16)
    plt.grid()
    #plt.ylabel(r"$N_N/N_N(a=1)$, $\kappa$, $|\eta_B|\times 10^{10}$",  fontsize=16)
    plt.show()

#@jit
def fast_RHS(y0, lna, M1, d, invd, w1, epstt, epsmm, epsee, epstm,epste,epsme,c1t,c1m,c1e, widtht, widthm, N1_eq_SM, nN_int, GCF, gN, gst, V):
    N1      = y0[0] # RHN number density
    Ntt     = y0[1]
    Nmm     = y0[2]
    Nee     = y0[3]
    Ntm     = y0[4]
    Nte     = y0[5]
    Nme     = y0[6]
    Tsm = np.abs(y0[7])
    Th = np.abs(y0[8])

    zh=M1/Th
    zsm=M1/Tsm

    c1tc    = c1t.conjugate()
    c1mc    = c1m.conjugate()
    c1ec    = c1e.conjugate()
    
    Mpl = np.sqrt(1/(8 * np.pi * GCF)) #planck mass

    cut = 20

    if zh > cut:
        N1_eq_hot = gN*np.real(V)*(M1*Th/(2*np.pi))**(3/2)*np.exp(-zh)
    else:
        N1_eq_hot = gN*np.real(V)*Ipp(Th,zh)/(2*np.pi**2)

    f = N1/N1_eq_hot

    #partial derivatives of hot sector equilibrium number density
    if zh>cut:
        dN1_eq_hotdTh=gN*np.real(V)*(M1/(2*np.pi))**(3/2)*np.exp(-zh)*(3/2*np.sqrt(Th)+M1/np.sqrt(Th))

        rhoN = M1*N1
        drhoNdTh = f*M1*dN1_eq_hotdTh

        pN = N1*Th
        dpNdTh=N1

    else:
        dN1_eq_hotdTh=gN*np.real(V)/(2*np.pi**2)*Th*(3*Th*Ip(zh)-M1*dIpdx(zh))

        #sets energy density, pressure and derivatives with respect to constant f of hot sector
        rhoN = gN/(2*np.pi**2)*Th**4*f*Jp(zh)
        drhoNdTh = gN/(2*np.pi**2)*Th**2*(4*Th*f*Jp(zh)-M1*f*dJpdx(zh)) #constant f derivative

        pN = gN/(2*np.pi**2)*Th**4*f*Kp(zh)
        dpNdTh=gN/(2*np.pi**2)*Th**2*(4*Th*f*Kp(zh)-M1*f*dKpdx(zh)) #constant f derivative

    #sets entropy density and derivative for RHN
    sN = (rhoN+pN)/Th
    dsNdTh = (drhoNdTh+dpNdTh-sN)/Th #constant f derivative

    #set energy density, pressure, and entropy density for standard model relatiivistic d.o.f.
    rhoSM = np.pi**2/30.*gst*Tsm**4
    drhoSMdTsm = 2*np.pi**2/15.*gst*Tsm**3

    pSM = 1/3*rhoSM
    dpSMdTsm = 1/3*drhoSMdTsm

    sSM=(rhoSM+pSM)/Tsm
    dsSMdTsm = (drhoSMdTsm+dpSMdTsm-sSM)/Tsm


    #set total energy density, pressure and entropy density for SM
    rho = rhoSM + rhoN

    p = pSM + pN

    s = sSM+sN

    H            =      np.sqrt(rho/3.)/Mpl #Hubble parameter

    dN1dlna      =    (-np.exp(3*lna)*N1*d/H +  (N1_eq_SM)*invd/H) #RHN number density BE - for comoving N1

    dQdlna = -M1*dN1dlna/V #set energy transfer rate

    dNttdlna = -epstt*dN1dlna-0.5*w1/H*(2*c1t*c1tc*Ntt + c1m*c1tc*Ntm + c1e*c1tc*Nte + (c1m*c1tc*Ntm+c1e*c1tc*Nte).conjugate()                  )
    dNmmdlna = -epsmm*dN1dlna-0.5*w1/H*(2*c1m*c1mc*Nmm + c1m*c1tc*Ntm + c1e*c1mc*Nme + (c1m*c1tc*Ntm+c1e*c1mc*Nme).conjugate()                  )
    dNeedlna = -epsee*dN1dlna-0.5*w1/H*(2*c1e*c1ec*Nee + c1e*c1mc*Nme + c1e*c1tc*Nte + (c1e*c1mc*Nme+c1e*c1tc*Nte).conjugate()                  )
    dNtmdlna = -epstm*dN1dlna-0.5*w1/H*(  c1t*c1mc*Nmm + c1e*c1mc*Nte + c1m*c1mc*Ntm + c1mc*c1t*Ntt + c1t*c1tc*Ntm + c1t*c1ec*(Nme.conjugate()) ) - widtht*Ntm - widthm*Ntm
    dNtedlna = -epste*dN1dlna-0.5*w1/H*(  c1t*c1ec*Nee + c1e*c1ec*Nte + c1m*c1ec*Ntm + c1t*c1ec*Ntt + c1t*c1mc*Nme + c1t*c1tc*Nte               ) - widtht*Nte
    dNmedlna = -epsme*dN1dlna-0.5*w1/H*(  c1m*c1ec*Nee + c1e*c1ec*Nme + c1m*c1ec*Nmm + c1t*c1ec*(Ntm.conjugate())  + c1m*c1mc*Nme + c1m*c1tc*Nte) - widthm*Nme

    dN1dlna = np.exp(-3*lna)*dN1dlna - 3*N1 #rewriting the derivative in terms of the non-comoving N1

    deltaSM = drhoSMdTsm+dpSMdTsm

    denomSM = deltaSM - sSM

    dTsmdlna = (np.exp(-3*lna)*dQdlna-3*sSM*Tsm)/denomSM #SM temperature derivative from second law of thermodynamics

    dThdlna = (np.exp(-3*lna)*dQdlna - 3*(rhoSM+pSM) - drhoSMdTsm*dTsmdlna + dN1dlna*pN/N1)/(pN/N1_eq_hot*dN1_eq_hotdTh+sN-dpNdTh) #hot sector temp derivative via second law + comoving energy conservation

    return [dN1dlna, dNttdlna, dNmmdlna, dNeedlna, dNtmdlna, dNtedlna, dNmedlna, dTsmdlna, dThdlna, dQdlna]

class EtaB_1DMEsf(ulysses.ULSBase):
    """
    Boltzmann equations with one decaying sterile. For detailed discussions of
    equation derivation see arxiv:1104.2750.  Note these kinetic equations do
    not include off diagonal flavour oscillations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pnames=['m', 'M1', 'M2', 'M3', 'delta', 'a21', 'a31', 'x1', 'x2', 'x3', 'y1', 'y2', 'y3', 't12', 't13', 't23', 'kappa']
        self.GCF   = 6.71862e-39      # Gravitational constant in GeV^-2
        self.evolEnd = False

        #-------------------------------------#
        #    g*(T) and g*S(T) interpolation   #
        #-------------------------------------#
        import os
        data_dir = os.path.dirname(ulysses.__file__)

        fg  = os.path.join(data_dir, "etab1BE1Fscalefactor_gstar.txt")
        fgS = os.path.join(data_dir, "etab1BE1Fscalefactor_gstarS.txt")

        Dg  = np.loadtxt(fg)
        DgS = np.loadtxt(fgS)

        self.tck_   = interpolate.splrep(Dg[:,0],  Dg[:,1],  s=0)
        self.tckS_  = interpolate.splrep(DgS[:,0], DgS[:,1], s=0)
        self.evolname="$a$"

    def ipol_gstar(self, T):
        return interpolate.splev(T, self.tck_, der = 0)

    def ipol_gstarS(self, T):
        return interpolate.splev(T, self.tckS_, der = 0)

    def ipol_dgstarSdT(self,T):
        return interpolate.splev(T, self.tckS_, der = 1)

    def shortname(self): return "1BE1Fsf"

    def flavourindices(self): return [1, 2]

    def flavourlabels(self): return ["$T$", "$NBL$"]

    def sigma(self, s,mH,mN,yH):
        hDiff = mH**2-s
        hSum = mH**2+s
        prefactor = yH**4/(16*np.pi*mH**2*s**2*hDiff**2*hSum)
        term1 = mH**2*hDiff*hSum*(-mN**4 + mN**2*(mH**2+5*s) + 2*mH**4 - mH**2*s)*np.log(mH**2/hSum)
        term2 = s*(mN**4 * (2*mH**4 - mH**2*s + s**2) + mN**2 *(mH**6 + 2*mH**4 *s - 7*mH**2 * s**2) + mH**2 *(2*mH**6 - 2*mH**4 * s + mH**2*s**2 + s**3))

        return prefactor*(term1 + term2)

    def sv(self, Th,mH,mN,yH):
        integrand = lambda s: 1./(16*Th**3*mN**2*my_kn2(mN/Th))*self.sigma(s,mH,mN,yH)/np.sqrt(s)*(s-mN**2)**2*my_kn1(np.sqrt(s)/Th)
        return quad(integrand, mN**2, cutoff*mN**2,epsabs=absErr, epsrel = relErr)[0]

    def RHS(self, y0, lna, nN_int, epstt, epsmm, epsee,epstm,epste,epsme,c1t,c1m,c1e,k, V):
        N1 = y0[0]
        Tsm = np.real(y0[7])
        Th = np.real(y0[8])

        zsm = self.M1/Tsm
        zh = self.M1/Th

        _d       = np.real(self.Gamma1* my_kn1(zh) / my_kn2(zh)) #decay rate thermal averaged with hot sector
        _invd = np.real(self.Gamma1* my_kn1(zsm) / my_kn2(zsm)) #decay rate thermal averaged with SM
        _w1      = _invd * 0.25 * my_kn2(zsm) * zsm**2 #washout rate
        nN_eq     = self.N1Eq(zsm) #equilibrium number density of neutrinos

        if(np.log10(np.exp(3*lna)*np.real(N1)/nN_int) < -6):
            self.evolEnd = True

        # thermal widths are set to zero such that we are in the "one-flavoured regime"
        widtht = 485e-10*self.MP/self.M1
        widthm = 1.7e-10*self.MP/self.M1

        if self.evolEnd:
            return [-3*N1, 0, 0, 0, 0, 0, 0, -Tsm, -Th, 0]
        else:
            return fast_RHS(y0, lna, self.M1, _d, _invd, _w1,  epstt, epsmm, epsee, epstm,epste,epsme,c1t,c1m,c1e, widtht, widthm, nN_eq, nN_int, self.GCF, self.gN, self.ipol_gstar(Tsm), V)

    def __call__(self, x):
        r"""
        Operator that returns EtaB for a given parameter point.

        :Arguments:
            * *x* (``dict``) --
              parameter dictionary

        NOTE --- this operator is intended to be used with derived classes where EtaB is implemented
        """
        self.setParams(x)
        self.kappa=x['kappa']
        return self.EtaB
    


    @property
    def EtaB(self): #kappa is initial ratio Th/Tsm
        #Define fixed quantities for BEs
        epstt = np.real(self.epsilon1ab(2,2))
        epsmm = np.real(self.epsilon1ab(1,1))
        epsee = np.real(self.epsilon1ab(0,0))
        epstm =         self.epsilon1ab(2,1)
        epste =         self.epsilon1ab(2,0)
        epsme =         self.epsilon1ab(1,0)

        c1t   =                 self.c1a(2)
        c1m   =                 self.c1a(1)
        c1e   =                 self.c1a(0)

        k       = np.real(self.k1)

        self.evolEnd = False

        self.f=1

        ggamma      = 2.

        self.gN=2. #RHN relativistic degrees of freedom
        self.inTsm      = 100. * self.M1 # initial temp 100x greater than mass of N1
        self.inTh = self.kappa*self.inTsm

        V = np.pi**2/(self.inTsm**3*zeta(3)*ggamma) #volume factor to normalise the number density, keeping it consistent with the equilibrium number density

        #define initial and final ln(a)
        lnain = 0.
        lnaf = 4*np.log(self.inTsm/self.M1)
        global lnarange
        lnarange=lnaf-lnain

        zeta3= zeta(3)

        besselLimit = 2 # limit of z^2 K_2(z) as z-> 0

        N1_eq_hot=1/(2*np.pi**2)*self.gN*self.inTh**3*besselLimit*V #initial RHN number density at temperature Th

        nN_int = self.f*N1_eq_hot

        self.rho_in=np.pi**2/30.*(self.ipol_gstar(self.inTsm)*self.inTsm**4+(7./8.)*self.f*self.gN*self.inTh**4)

        y0      = np.array([nN_int+0j,0+0j,0+0j,0+0j,0+0j,0+0j,0+0j, self.inTsm, self.inTh, 0], dtype=np.complex128) #initial array
        nphi    = (2.*zeta(3)/np.pi**2) * self.inTsm**3
        params  = np.array([nN_int, epstt,epsmm,epsee,epstm,epste,epsme,c1t,c1m,c1e,k,V], dtype=np.complex128)
        
        lnsf = np.linspace(lnain, lnaf, num=100, endpoint=True)

        ys = odeintw(self.RHS, y0, lnsf, args = tuple(params)) #solves BEs

        T=np.abs(ys[:,7])
        Th=np.abs(ys[:,8])

        gstarSrec = self.ipol_gstarS(0.3e-9) # d.o.f. at recombination
        gstarSoff = self.ipol_gstarS(T[-1])  # d.o.f. at the end of leptogenesis
        SMspl       = 28./79.
        zeta3       = zeta(3)
        
        coeffNgamma = ggamma*zeta3/np.pi**2

        Ngamma      = coeffNgamma*(np.exp(lnsf)*T)**3
        coeffsph    =  SMspl * gstarSrec/gstarSoff

        NBL=np.real(ys[:,1]+ys[:,2]+ys[:,3])
        #self.ys = np.empty((len(T), 5))
        #self.ys[:,0]=lnsf
        #self.ys[:,1]=ys[:,0]
        #self.ys[:,2]=ys[:,1]
        #self.ys[:,3]=ys[:,2]

        etab = coeffsph*NBL*nphi/Ngamma

        zsm=self.M1/T

        zh=self.M1/Th

        d       = np.real(self.Gamma1* kn(1,zh) / kn(2,zh)) #decay rate thermal averaged with hot sector

        if showLeptoPlot^showTemps:
            invd = np.real(self.Gamma1* kn(1,zsm) / kn(2,zsm)) #decay rate thermal averaged with SM
            invd[np.isnan(invd)] = 0
            w=invd * 0.25 * kn(2,zsm) * zsm**2

            eps=(epstt + epsmm + epsee)

            neq=3/8*zsm**2*kn(2,zsm)

            washout= np.abs(w * NBL)
            
            source=np.abs(eps *(-ys[:,0]*d +  neq*invd))

            showPlot(lnsf, ys, etab, T, Th, nN_int, NBL, washout, source)

        nH = ys[:,0]

        yH = np.amax(np.abs(np.transpose(self.h)[0]))

        eqBool = False

        for i in range(0,100):
            Mpl = np.sqrt(1/(8 * np.pi * self.GCF)) #planck mass

            cut = 20

            if zh[i] > cut:
                N1_eq_hot = self.gN*np.real(V)*(self.M1*Th[i]/(2*np.pi))**(3/2)*np.exp(-zh[i])
            else:
                N1_eq_hot = self.gN*np.real(V)*Ipp(Th[i],zh[i])/(2*np.pi**2)

            f = nH[i]/N1_eq_hot

            rhoN = self.gN/(2*np.pi**2)*Th[i]**4*f*Jp(zh[i])
            rhoSM = np.pi**2/30.*self.ipol_gstar(T[i])*T[i]**4
            rho = rhoSM + rhoN

            Hubble = np.real(np.sqrt(rho/3.)/Mpl) #Hubble parameter

            leptogst = 3/4*2*6

            nSM = np.real(zeta3/(np.pi**2)*leptogst*T[i]**3)

            n = nH[i]

            GammaTherm = np.real(self.sv(np.real(Th[i]),125,self.M1,yH)*nSM)

            if ((GammaTherm > d[i]) and (GammaTherm > Hubble) and (np.real(Th[i]/self.M1) > 0.1)):
                eqBool = True
        
        if(eqBool):
            return -np.abs(etab[-1])
        else:
            return np.abs(etab[-1])