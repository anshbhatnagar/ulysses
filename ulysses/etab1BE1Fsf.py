import ulysses
import numpy as np
from scipy import interpolate
from scipy.integrate import odeint
from scipy.special import zeta
from ulysses.numba import jit
import matplotlib.pyplot as plt


#+++++++++++++++++++++++++++++++++++++++++++++++++#
#             FLRW-Boltzmann Equations            #
#+++++++++++++++++++++++++++++++++++++++++++++++++#

@jit
def fast_RHS(y0, a, Tp, gst, rRADi, ain, d, w1, epstt, epsmm, epsee, rnuRda_eq, Del, GCF):
    rnuR      = y0[0] # RHN
    NBL       = y0[1] # B-L asymmetry
    
    Mpl = np.sqrt(1/(8 * np.pi * GCF))
    #rho = np.pi**2/30.*(gst)*Tp**4

    rho = np.pi**2/30.*(gst*Tp**4)
    H            =      np.sqrt(rho/3.)/Mpl #Hubble parameter
    expression1  =      (rnuR -  rnuRda_eq) * d/(a*H)
    drnuRda      =    - expression1
    dNBLa        =      (epstt + epsmm + epsee) * expression1 -  (w1/(a*H)) * NBL
    return [drnuRda, dNBLa]

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

    def RHS(self, y0, a, Tp, epstt, epsmm, epsee, rRADi, ain):
        Tp = self.Ti/(a)
        z             = self.M1/Tp
        from ulysses.ulsbase import my_kn2, my_kn1
        kn2 = my_kn2(z)
        _d       = np.real(self.Gamma1* my_kn1(z) / kn2)
        _w1      = _d * 0.25 * kn2 * z**2
        rnuRda_eq     = self.N1Eq(z)
        Del           =  1. + Tp * self.ipol_dgstarSdT(Tp)/(3. * self.ipol_gstar(Tp)) # Temperature parameter
        return fast_RHS(y0, a, Tp, self.ipol_gstar(Tp), rRADi, ain, _d, _w1,  epstt, epsmm, epsee, rnuRda_eq, Del, self.GCF)

    @property
    def EtaB(self):
        #Define fixed quantities for BEs
        epstt = np.real(self.epsilon1ab(2,2))
        epsmm = np.real(self.epsilon1ab(1,1))
        epsee = np.real(self.epsilon1ab(0,0))
        ggamma      = 2.
        self.gN=7/8*2.
        self.Ti      = 100 * self.M1 # initial temp 100 greater than mass N1
        Tp = self.Ti
        V = np.pi**2/(Tp**3*zeta(3)*ggamma)
        nN_int=3./4.*zeta(3)/(np.pi**2)*self.gN*Tp**3*V
        rRadi   = np.pi**2 * self.ipol_gstar(Tp) / 30. * Tp**4 # initial radiation domination rho_RAD = pi^2* gstar(T[0])/30*T^4
        y0      = np.array([nN_int, 0.])
        nphi    = (2.*zeta(3)/np.pi**2) * Tp**3
        params  = np.array([Tp, epstt, epsmm, epsee, np.real(rRadi), 1.])
        af = np.exp(2*np.log(Tp/self.M1))
        t1 = np.linspace(1., af, num=10000, endpoint=True)

        # solve equation
        ys      = odeint(self.RHS, y0, t1, args = tuple(params))
        # functions for converting to etaB using the solution to find temp
        T           = self.Ti/t1
        gstarSrec = self.ipol_gstarS(0.3e-9) # d.o.f. at recombination
        gstarSoff = self.ipol_gstarS(T[-1])  # d.o.f. at the end of leptogenesis
        SMspl       = 28./79.
        zeta3       = zeta(3)
        coeffNgamma = ggamma*zeta3/np.pi**2
        Ngamma      = coeffNgamma*(t1*T)**3
        coeffsph    =  SMspl * gstarSrec/gstarSoff
        self.ys = np.empty((len(T), 4))
        self.ys[:,0]=t1
        self.ys[:,1]=T
        self.ys[:,2]=ys[:,1]
        self.ys[:,-1] = coeffsph*( ys[:,1])*nphi/Ngamma

        print(Ngamma)

        #plt.plot(np.log(t1), np.abs(ys[:,0]), color='r', label=r'$N_N$')
        #plt.plot(np.log(t1), np.log(np.abs(ys[:,1])), color='g', label=r'$N_{B-L}\times 10^{6}$')
        #plt.plot(np.log(t1), self.ys[:,-1], color='b', label=r'$|\eta_B|\times 10^9$')
        #plt.xlabel(r"$\ln(a)$", fontsize=16)
        #plt.legend(loc='lower right', fontsize=16)
        #plt.ylabel(r"$N_N$, $N_{B-L}$, $|\eta_B|$",  fontsize=16)
        #plt.show()

        return self.ys[-1][-1]
