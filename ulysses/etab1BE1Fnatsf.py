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


#+++++++++++++++++++++++++++++++++++++++++++++++++#
#             FLRW-Boltzmann Equations            #
#+++++++++++++++++++++++++++++++++++++++++++++++++#


def Jp(x):
    integrand = lambda z: z**2*np.sqrt(np.abs(z**2+x**2))/(1+np.exp(np.sqrt(np.abs(z**2+x**2))))
    return quad(integrand, 0, 50, epsabs=5e-3)[0]

def Ip(x):
    integrand = lambda z: z**2/(1+np.exp(np.sqrt(np.abs(z**2+x**2))))
    return quad(integrand, 0, 50, epsabs=5e-3)[0]

def Ipp(Th, M1):
    return Th**3*Ip(M1/Th)

def invIpp(y, M1):
    f = lambda x, m: Ipp(x, m)+y
    return fsolve(f, 400*M1, args=(M1))[0]

def relrho(T,gN, M):
    #return gN*Th**4*Jp(M/Th)/(2*np.pi**2)
    return 7/8*np.pi**2/30*gN*T**4

def relT(N,gN,M):
    #return invIpp(2*np.pi**2*N/(a**3*gN),M)
    return (4/3*N/zeta(3)*np.pi**2/gN)**1/3



#@jit
def fast_RHS(y0, lna, M1, gst, gsts, gN, gNs, d, w1, epstt, epsmm, epsee, rnuRda_eq, GCF):
    nN      = y0[0] # RHN number density
    Tsm       = y0[1]  #standard model temperature
    Th = y0[2] #hot sector temperature
    NBL = y0[3] # B-L asymmetry

    # set previous timesteps derivatives
    dTsmdlna_old = dTdlna[0]
    dThdlna_old = dTdlna[1]
    
    
    Mpl = np.sqrt(1/(8 * np.pi * GCF)) #planck mass
    rho = np.pi**2/30.*(gst)*Tsm**4+7/8*np.pi**2/30*gN*Th**4 # total comoving energy density

    s = 2*np.pi/45*(gsts*Tsm**3+gNs*Th**3) # total comoving entropy density

    dssmdTsm = 2*np.pi/15*gsts*Tsm**2 
    dsNdTh = 2*np.pi/15*gNs*Th**2

    H            =      np.sqrt(rho/3.)/Mpl #Hubble parameter

    dnNdlna      =    -(nN -  rnuRda_eq) * d/(H) 
    dTdlna[0] = -(30*rho/(np.pi**2*gst)+gN/gst*(Th**3)*dThdlna_old)/Tsm**3
    dTdlna[1] = -(dssmdTsm*dTsmdlna_old+3*s)/dsNdTh
    dNBLlna        =     -(epstt + epsmm + epsee) * dnNdlna -  (w1/(H)) * NBL
    
    return [dnNdlna, dTdlna[0], dTdlna[1], dNBLlna]

class EtaB_1BE1Fsf(ulysses.ULSBase):
    """
    Boltzmann equations with one decaying sterile. For detailed discussions of
    equation derivation see arxiv:1104.2750.  Note these kinetic equations do
    not include off diagonal flavour oscillations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    def RHS(self, lna, y0, Th, Tsm, epstt, epsmm, epsee):
        Th = y0[2] #previous Th
        Tsm = y0[1] #previous Tsm

        gst = self.ipol_gstar(Tsm) #gst based on previous Tsm
        zsm             = self.M1/Tsm
        zh = self.M1/Th

        kn2 = my_kn2(zh)
        _d       = np.real(self.Gamma1* my_kn1(zh) / kn2)
        _w1      = _d * 0.25 * kn2 * zsm**2
        nN_eq     = self.N1Eq(zsm) #equilibrium number density of neutrinos

        print(lna)
        
        return fast_RHS(y0, lna, self.M1, gst, self.ipol_gstarS(Tsm), self.gN, self.gN, _d, _w1,  epstt, epsmm, epsee, nN_eq, self.GCF)


    @property
    def EtaB(self):
        #Define fixed quantities for BEs
        epstt = np.real(self.epsilon1ab(2,2))
        epsmm = np.real(self.epsilon1ab(1,1))
        epsee = np.real(self.epsilon1ab(0,0))
        kappa = 1.06  # initial ratio Th/Tsm
        self.gN=2. #RHN relativistic degrees of freedom
        Tsm      = 100. * self.M1 # initial temp 100x greater than mass of N1
        Th = kappa*Tsm

        lnain = 0.
        lnaf = 10.

        global dTdlna
        dTdlna = [-Tsm, -Th] #global variable to store previous timesteps dTda

        rRadi   = np.pi**2 * self.ipol_gstar(Tsm) / 30. * Tsm**4 # initial radiation domination rho_RAD = pi^2* gstar(T[0])/30*T^4
        y0      = [3./4.*zeta(3)/(np.pi**2)*self.gN*Th**3,Tsm,Th, 0.] #initial array
        nphi    = (2.*zeta(3)/np.pi**2) * Tsm**3
        params  = [Th, Tsm, epstt, epsmm, epsee]
        
        lnsf = np.linspace(lnain, lnaf, num=100, endpoint=True)


        ys = solve_ivp(self.RHS, [lnain, lnaf], y0, method='DOP853', args = params)

        # functions for converting to etaB using the solution to find temp
        T           = ys.y[1]
        lnsf = ys.t
        gstarSrec = self.ipol_gstarS(0.3e-9) # d.o.f. at recombination
        gstarSoff = self.ipol_gstarS(T[-1])  # d.o.f. at the end of leptogenesis
        SMspl       = 28./79.
        zeta3       = zeta(3)
        ggamma      = 2.
        coeffNgamma = ggamma*zeta3/np.pi**2

        Ngamma      = coeffNgamma*(np.exp(lnsf)*T)**3
        coeffsph    =  SMspl * gstarSrec/gstarSoff


        #self.ys = np.empty((len(T), 5))
        #self.ys[:,0]=lnsf
        #self.ys[:,1]=ys[:,0]
        #self.ys[:,2]=ys[:,1]
        #self.ys[:,3]=ys[:,2]
        etab = coeffsph*( ys.y[3])*nphi/Ngamma

        plt.plot(lnsf, ys.y[2]/ys.y[1], color='r', label=r'$\kappa$')
        plt.plot(lnsf, ys.y[0]*1e-47, color='g', label=r'$N_N\times 10^{47}$')
        plt.plot(lnsf, etab*1e-39, color='b', label=r'$|\eta_B|\times 10^{-39}$')
        plt.xlabel(r"$\ln(a)$", fontsize=16)
        plt.legend(loc='upper right', fontsize=16)
        plt.ylabel(r"$N_N$, $\kappa$, $|\eta_B|$",  fontsize=16)
        plt.show()

        return ys.y[-1][-1]
