from cmath import e
from locale import MON_1
import ulysses
import numpy as np
from scipy import interpolate
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.special import zeta
from scipy.special import k1
from scipy.optimize import fsolve
from scipy.integrate import quad
from ulysses.numba import jit
import matplotlib.pyplot as plt
import ulysses.numba as nb
from ulysses.ulsbase import my_kn2, my_kn1

import ast

import progressbar as pb

relApprox = False
nonRelApprox = True

#+++++++++++++++++++++++++++++++++++++++++++++++++#
#             FLRW-Boltzmann Equations            #
#+++++++++++++++++++++++++++++++++++++++++++++++++#

def Ip(x):
    integrand = lambda z: z**2/(1+np.exp(np.sqrt(np.abs(z**2+x**2))))
    return quad(integrand, 0, 50, epsabs=5e-3)[0]

def dIpdx(x):
    integrand = lambda z: -x*z**2*np.exp(np.sqrt(np.abs(z**2+x**2)))/(np.sqrt(np.abs(z**2+x**2))*(1+np.exp(np.sqrt(np.abs(z**2+x**2))))**2)
    return quad(integrand, 0, 50, epsabs=5e-3)[0]

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
    return quad(integrand, 0, 50, epsabs=5e-3)[0]

def dJpdx(x): #derivative of the J+ function
    integrand = lambda z: -x*z**2*(-1+np.exp(np.sqrt(np.abs(z**2+x**2)))*(-1+np.sqrt(np.abs(z**2+x**2))))/((1+np.exp(np.sqrt(np.abs(z**2+x**2))))**2*np.sqrt(np.abs(z**2+x**2)))
    return quad(integrand, 0, 50, epsabs=5e-3)[0]

def Kp(x): #K+ function for pressure
    integrand = lambda z: z**4/(3*np.sqrt(np.abs(z**2+x**2))*(1+np.exp(np.sqrt(np.abs(z**2+x**2)))))
    return quad(integrand, 0, 50, epsabs=5e-3)[0]

def dKpdx(x): #derivative of the K+ function
    integrand = lambda z:  -x*z**4*(1+np.exp(np.sqrt(np.abs(z**2+x**2)))*(1+np.sqrt(np.abs(z**2+x**2))))/(3*(1+np.exp(np.sqrt(np.abs(z**2+x**2))))**2*np.sqrt(np.abs(z**2+x**2))**3)
    return quad(integrand, 0, 50, epsabs=5e-3)[0]


#@jit
def fast_RHS(y0, lna, M1, gst, gsts, dgstsdTsm, gN, d, invd, w1, eps, rnuRda_eq, GCF, V):
    nN      = y0[0] # RHN number density
    Tsm       = y0[1]  #standard model temperature
    Th = y0[2] #hot sector temperature
    NBL = y0[3] #B-L asymmetry
    Q = y0[4] #total energy transferred between sectors
    
    Mpl = np.sqrt(1/(8 * np.pi * GCF)) #planck mass

    zh=M1/Th

    #set energy density, pressure and derivatives for RHN assuming the relativistic approximation or using the full expressions
    if relApprox:
        rhoN = 7/8*np.pi**2/30.*gN*Th**4
        drhoNdTh = 7/8*2*np.pi**2/15.*gN*Th**3

        pN = 1/3*rhoN
        dpNdTh=1/3*drhoNdTh
    elif nonRelApprox and zh>100:
        dnNdTh=gN*(M1/(2*np.pi))**(3/2)*np.exp(-zh)*(3/2*np.sqrt(Th)+M1/np.sqrt(Th))

        rhoN = M1*nN
        drhoNdTh = M1*dnNdTh

        pN = nN*Th
        dpNdTh=nN
    else:
        rhoN = gN/(2*np.pi**2)*Th**4*Jp(zh)
        drhoNdTh = gN/(2*np.pi**2)*Th**2*(4*Th*Jp(zh)-M1*dJpdx(zh))

        pN = gN/(2*np.pi**2)*Th**4*Kp(zh)
        dpNdTh=gN/(2*np.pi**2)*Th**2*(4*Th*Kp(zh)-M1*dKpdx(zh))

    #sets entropy density and derivative for RHN
    sN = (rhoN+pN)/Th
    dsNdTh = (drhoNdTh+dpNdTh-sN)/Th 

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
    
    dQdlna = nN*d*M1/H #set energy transfer rate

    #denom = dsNdTh*(1.0-(drhoNdTh*dsSMdTsm)/(drhoSMdTsm*dsNdTh)) #denominator for Th derivative

    dnNdlna      =    -nN*d/H +  (rnuRda_eq)*invd/H #RHN number density BE

    x=np.exp(-3*lna)*2*np.pi**2*nN/(gN*V) #parameter for inversion

    if x*V<10**(-10): #checks if x*V is close to zero, so that number density and temperature relation won't be used
        dThdlna=-Th #evolve the temperature as a relativistic relic

    else: #otherwise use the inversion of the number density/temperature relation
        dinvIpp=dinvIppdx(x,M1,Th)

        pThpnN = dinvIpp*np.exp(-3*lna)*2*np.pi**2/(gN*V)

        pThplna=-dinvIpp*np.exp(-3*lna)*6*np.pi**2*nN/(gN*V)

        dThdlna = pThpnN*dnNdlna+pThplna

    #dThdlna = (np.exp(-3*lna)*(1/Tsm-1/Th)*dQdlna+dsSMdTsm/drhoSMdTsm*(3*(rho+p))-3*s)/denom #hot sector temperature derivative

    dTsmdlna = -(3*(rho+p)+drhoNdTh*dThdlna)/drhoSMdTsm #SM temperature derivative
    
    dNBLdlna        =     -eps * dnNdlna -  (w1/(H)) * NBL #B-L asymmetry derivative

    pbar.update((lna/lnarange)*100)
    
    return [dnNdlna, dTsmdlna, dThdlna, dNBLdlna, dQdlna]

class EtaB_1BE1Fsf(ulysses.ULSBase):
    """
    Boltzmann equations with one decaying sterile. For detailed discussions of
    equation derivation see arxiv:1104.2750.  Note these kinetic equations do
    not include off diagonal flavour oscillations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(self.pnames)
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

    def RHS(self, lna, y0, Th, Tsm, epstt, epsmm, epsee, V):
        Th = y0[2] #previous Th
        Tsm = y0[1] #previous Tsm

        gst = self.ipol_gstar(Tsm) #gst based on previous Tsm
        zsm             = self.M1/Tsm
        zh = self.M1/Th

        _d       = np.real(self.Gamma1* my_kn1(zh) / my_kn2(zh)) #decay rate thermal averaged with hot sector
        _invd = np.real(self.Gamma1* my_kn1(zsm) / my_kn2(zsm)) #decay rate thermal averaged with SM
        _w1      = _invd * 0.25 * my_kn2(zsm) * zsm**2 #washout rate
        nN_eq     = self.N1Eq(zsm) #equilibrium number density of neutrinos

        eps = (epstt + epsmm + epsee)
        
        return fast_RHS(y0, lna, self.M1, gst, self.ipol_gstarS(Tsm), self.ipol_dgstarSdT(Tsm), self.gN, _d, _invd, _w1,  eps, nN_eq, self.GCF, V)


    @property
    def EtaB(self): #kappa is initial ratio Th/Tsm
        #Define fixed quantities for BEs
        epstt = np.real(self.epsilon1ab(2,2))
        epsmm = np.real(self.epsilon1ab(1,1))
        epsee = np.real(self.epsilon1ab(0,0))

        ggamma      = 2.

        self.gN=7/8*2. #RHN relativistic degrees of freedom
        Tsm      = 100. * self.M1 # initial temp 100x greater than mass of N1
        Th = self.kappa*Tsm

        V = np.pi**2/(Tsm**3*zeta(3)*ggamma) #volume factor to normalise the number density, keeping it consistent with the equilibrium number density

        #define initial and final ln(a)
        lnain = 0.
        lnaf = 2*np.log(Th/self.M1)
        global lnarange
        lnarange=lnaf-lnain

        nN_int=3./4.*zeta(3)/(np.pi**2)*self.gN*Th**3*V #initial RHN number density at temperature Th

        rRadi   = np.pi**2 * self.ipol_gstar(Tsm) / 30. * Tsm**4 # initial radiation domination rho_RAD = pi^2* gstar(T[0])/30*T^4
        y0      = [nN_int,Tsm,Th, 0., 0.] #initial array
        nphi    = (2.*zeta(3)/np.pi**2) * Tsm**3
        params  = [Th, Tsm, epstt, epsmm, epsee, V]
        
        lnsf = np.linspace(lnain, lnaf, num=100, endpoint=True)

        global pbar
        pbar=pb.ProgressBar().start()

        ys = solve_ivp(self.RHS, [lnain, lnaf], y0, method='BDF', args = params) #solves BEs

        pbar.finish()

        # functions for converting to etaB using the solution to find temp
        T           = ys.y[1]
        lnsf = ys.t
        gstarSrec = self.ipol_gstarS(0.3e-9) # d.o.f. at recombination
        gstarSoff = self.ipol_gstarS(T[-1])  # d.o.f. at the end of leptogenesis
        SMspl       = 28./79.
        zeta3       = zeta(3)
        
        coeffNgamma = ggamma*zeta3/np.pi**2

        Ngamma      = coeffNgamma*(np.exp(lnsf)*T)**3
        coeffsph    =  SMspl * gstarSrec/gstarSoff


        #self.ys = np.empty((len(T), 5))
        #self.ys[:,0]=lnsf
        #self.ys[:,1]=ys[:,0]
        #self.ys[:,2]=ys[:,1]
        #self.ys[:,3]=ys[:,2]
        etab = coeffsph*( ys.y[3])*nphi/Ngamma

        #plt.plot(lnsf, ys.y[2]/ys.y[1], color='r', label=r'$\kappa$')
        #plt.plot(lnsf, ys.y[0], color='g', label=r'$N_N$')
        #plt.plot(lnsf, etab*1e8, color='b', label=r'$|\eta_B|\times 10^{-8}$')
        #plt.plot(lnsf, np.abs(ys.y[4]), color='b', label=r'$Q$')
        #plt.plot(lnsf, np.log(ys.y[1]), color='b', label=r'$T_{SM}$')
        #plt.plot(lnsf, np.log(ys.y[2]), color='r', label=r'$T_H$')
        #plt.xlabel(r"$\ln(a)$", fontsize=16)
        #plt.legend(loc='upper right', fontsize=16)
        #plt.ylabel(r"$N_N$, $\kappa$, $|\eta_B|$",  fontsize=16)
        #plt.show()


        return etab[-1]


def main():
    kappa = np.linspace(1, 10, num=50, endpoint=True)

    model=EtaB_1BE1Fsf()

    with open("../examples/1N1F.dat", "r") as data:
        paramdict = ast.literal_eval(data.read())

    etab = model(paramdict, kappa)

    plt.plot(kappa, etab, color='r', label=r'$\eta_B$')
    plt.xlabel(r"$\kappa$", fontsize=16)
    plt.legend(loc='upper right', fontsize=16)
    plt.ylabel(r"$|\eta_B|$",  fontsize=16)
    plt.show()


#if __name__ == "__main__":
#    main()
