from cmath import e
from locale import MON_1
from operator import xor
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

showLeptoPlot = True
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
        plt.plot(lnsf, np.real(ys[:,0])/nN_int, color='g', label=r'$N_N/N_N(a=1)$')
        plt.plot(lnsf, np.abs(etab)*1e10, color='b', label=r'$|\eta_B|\times 10^{10}$')
        plt.plot(lnsf, np.abs(np.real(NBL))*1e8, label=r'$N_{B-L}\times 10^{8}$')
        plt.ylim(0,2)
        
    if showTemps:
        plt.plot(lnsf, np.log10(Tsm), color='b', label=r'$T_{SM}$')
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

    N1_eq_hot = np.exp(3*lna)*gN*np.real(V)*Ipp(Th,zh)/(2*np.pi**2)

    f = N1/N1_eq_hot

    #partial derivatives of hot sector equilibrium number density
    pN1_eq_hotpTh=np.exp(3*lna)*gN*np.real(V)/(2*np.pi**2)*Th*(3*Th*Ip(zh)-M1*dIpdx(zh))
    pN1_eq_hotplna = 3*N1_eq_hot

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

    dN1dlna      =    (-N1*d/H +  (N1_eq_SM)*invd/H) #RHN number density BE

    dQdlna = -M1*dN1dlna/V #set energy transfer rate

    pfplna = dN1dlna/N1_eq_hot-f/N1_eq_hot*pN1_eq_hotplna #partial derivative of f with respect to ln(a)
    pfpTh = -f/N1_eq_hot*pN1_eq_hotpTh #partial derivative of f with respect to Th

    prhoNpTh = drhoNdTh + rhoN/f*pfpTh #partial derivative of neutrino energy density with respect to Th
    ppNpTh = dpNdTh + pN/f*pfpTh #partial derivative of neutrino pressure with respect to Th

    prhoNplna = rhoN/f*pfplna #partial derivative of neutrino energy density with respect to ln(a)
    ppNplna = pN/f*pfplna #partial derivative of neutrino pressure with respect to ln(a)

    deltaSM = drhoSMdTsm+dpSMdTsm

    denomSM = deltaSM - sSM

    dTsmdlna = (np.exp(-3*lna)*dQdlna-3*sSM*Tsm)/denomSM #SM temperature derivative from second law of thermodynamics

    dThdlna = -(drhoSMdTsm*dTsmdlna+3*(rhoSM+rhoN+pSM+pN)+prhoNplna)/prhoNpTh #Hot sector temperature derivative from comoving energy conservation

    dNttdlna = -epstt*dN1dlna-0.5*w1/H*(2*c1t*c1tc*Ntt + c1m*c1tc*Ntm + c1e*c1tc*Nte + (c1m*c1tc*Ntm+c1e*c1tc*Nte).conjugate()                  )
    dNmmdlna = -epsmm*dN1dlna-0.5*w1/H*(2*c1m*c1mc*Nmm + c1m*c1tc*Ntm + c1e*c1mc*Nme + (c1m*c1tc*Ntm+c1e*c1mc*Nme).conjugate()                  )
    dNeedlna = -epsee*dN1dlna-0.5*w1/H*(2*c1e*c1ec*Nee + c1e*c1mc*Nme + c1e*c1tc*Nte + (c1e*c1mc*Nme+c1e*c1tc*Nte).conjugate()                  )
    dNtmdlna = -epstm*dN1dlna-0.5*w1/H*(  c1t*c1mc*Nmm + c1e*c1mc*Nte + c1m*c1mc*Ntm + c1mc*c1t*Ntt + c1t*c1tc*Ntm + c1t*c1ec*(Nme.conjugate()) ) - widtht*Ntm - widthm*Ntm
    dNtedlna = -epste*dN1dlna-0.5*w1/H*(  c1t*c1ec*Nee + c1e*c1ec*Nte + c1m*c1ec*Ntm + c1t*c1ec*Ntt + c1t*c1mc*Nme + c1t*c1tc*Nte               ) - widtht*Nte
    dNmedlna = -epsme*dN1dlna-0.5*w1/H*(  c1m*c1ec*Nee + c1e*c1ec*Nme + c1m*c1ec*Nmm + c1t*c1ec*(Ntm.conjugate())  + c1m*c1mc*Nme + c1m*c1tc*Nte) - widthm*Nme

    return [dN1dlna, dNttdlna, dNmmdlna, dNeedlna, dNtmdlna, dNtedlna, dNmedlna, dTsmdlna, dThdlna, dQdlna]

class EtaB_1DMEsf(ulysses.ULSBase):
    """
    Boltzmann equations with one decaying sterile. For detailed discussions of
    equation derivation see arxiv:1104.2750.  Note these kinetic equations do
    not include off diagonal flavour oscillations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pnames=['m', 'M1', 'M2', 'M3', 'delta', 'a21', 'a31', 'x1', 'x2', 'x3', 'y1', 'y2', 'y3', 't12', 't13', 't23', 'kappa', 'f']
        self.GCF   = 6.71862e-39      # Gravitational constant in GeV^-2

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

        # thermal widths are set to zero such that we are in the "one-flavoured regime"
        widtht = 485e-10*self.MP/self.M1
        widthm = 1.7e-10*self.MP/self.M1
        
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
        self.f=x['f']
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

        if showLeptoPlot^showTemps:
            zsm=self.M1/T

            zh=self.M1/Th

            d       = np.real(self.Gamma1* kn(1,zh) / kn(2,zh)) #decay rate thermal averaged with hot sector
            invd = np.real(self.Gamma1* kn(1,zsm) / kn(2,zsm)) #decay rate thermal averaged with SM
            invd[np.isnan(invd)] = 0
            w=invd * 0.25 * kn(2,zsm) * zsm**2

            eps=(epstt + epsmm + epsee)

            neq=3/8*zsm**2*kn(2,zsm)

            washout= np.abs(w * NBL)
            
            source=np.abs(eps *(-ys[:,0]*d +  neq*invd))

            showPlot(lnsf, ys, etab, T, Th, nN_int, NBL, washout, source)


        return etab[-1]