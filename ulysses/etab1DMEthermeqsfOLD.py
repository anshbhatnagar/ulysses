from cmath import e
from locale import MON_1
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

import ast

import progressbar as pb

showPlotBool=True

#+++++++++++++++++++++++++++++++++++++++++++++++++#
#             FLRW-Boltzmann Equations            #
#+++++++++++++++++++++++++++++++++++++++++++++++++#

def Ip(x):
    integrand = lambda z: z**2/(1+np.exp(np.sqrt(np.abs(z**2+x**2))))
    return quad(integrand, 0, 50, epsabs=5e-3)[0]

def Ipp(Th,zh):
    return Th**3*Ip(zh)

def IppRoot(Th, MN, x):
    return Ipp(Th,MN/Th)-x

def invIpp(x, MN, guess):
    return fsolve(IppRoot,guess,args=(MN,x))[0]

def Jp(x): #J+ function for energy density
    integrand = lambda z: z**2*np.sqrt(np.abs(z**2+x**2))/(1+np.exp(np.sqrt(np.abs(z**2+x**2))))
    return quad(integrand, 0, 50, epsabs=5e-3)[0]

def Kp(x): #K+ function for pressure
    integrand = lambda z: z**4/(3*np.sqrt(np.abs(z**2+x**2))*(1+np.exp(np.sqrt(np.abs(z**2+x**2)))))
    return quad(integrand, 0, 50, epsabs=5e-3)[0]


def showPlot(lnsf, ys, etab, Tsm, Th, nN_int, NBL, washout, source):
    plt.plot(lnsf, Th/Tsm, color='r', label=r'$\kappa$')
    plt.plot(lnsf, np.real(ys[:,0])/nN_int, color='g', label=r'$N_N/N_N(a=1)$')
    plt.plot(lnsf, np.abs(etab)*1e9, color='b', label=r'$|\eta_B|\times 10^{9}$')
    plt.plot(lnsf, np.abs(np.real(NBL))*1e8, label=r'$N_{B-L}\times 10^{8}$')
    #plt.plot(lnsf, washout/source[0], label=r'$|w/s(a=1)|$')
    #plt.plot(lnsf, source/source[0], label=r'$|s/s(a=1)|$')
    #plt.plot(lnsf, np.abs(ys.y[4]), color='b', label=r'$Q$')
    #plt.scatter(lnsf, np.real(ys[:,0]), color='g', label=r'$N_N/N_N(a=1)$')
    #plt.scatter(lnsf, Tsm*np.exp(lnsf), color='b', label=r'$T_{SM}$')
    #plt.scatter(lnsf, Th*np.exp(lnsf), color='r', label=r'$T_H$')
    #plt.title("$\kappa(a=1)=1$, $\log_{10}(m_1)=-1$")
    plt.xlabel(r"$\ln(a)$", fontsize=16)
    plt.legend(loc='upper right', fontsize=16)
    plt.ylim(0,2)
    #plt.ylabel(r"$N_N/N_N(a=1)$, $\kappa$, $|\eta_B|\times 10^{10}$",  fontsize=16)
    plt.show()

#@jit
def fast_RHS(y0, lna, rho, d, invd, w1, epstt, epsmm, epsee, epstm,epste,epsme,c1t,c1m,c1e, widtht, widthm, N1_eq, GCF):
    N1      = y0[0] # RHN number density
    Ntt     = y0[1]
    Nmm     = y0[2]
    Nee     = y0[3]
    Ntm     = y0[4]
    Nte     = y0[5]
    Nme     = y0[6]

    c1tc    = c1t.conjugate()
    c1mc    = c1m.conjugate()
    c1ec    = c1e.conjugate()
    
    Mpl = np.sqrt(1/(8 * np.pi * GCF)) #planck mass

    H            =      np.sqrt(rho/3.)/Mpl #Hubble parameter

    dN1dlna      =    -N1*d/H +  (N1_eq)*invd/H #RHN number density BE


    dNttdlna = -epstt*dN1dlna-0.5*w1/H*(2*c1t*c1tc*Ntt + c1m*c1tc*Ntm + c1e*c1tc*Nte + (c1m*c1tc*Ntm+c1e*c1tc*Nte).conjugate()                  )
    dNmmdlna = -epsmm*dN1dlna-0.5*w1/H*(2*c1m*c1mc*Nmm + c1m*c1tc*Ntm + c1e*c1mc*Nme + (c1m*c1tc*Ntm+c1e*c1mc*Nme).conjugate()                  )
    dNeedlna = -epsee*dN1dlna-0.5*w1/H*(2*c1e*c1ec*Nee + c1e*c1mc*Nme + c1e*c1tc*Nte + (c1e*c1mc*Nme+c1e*c1tc*Nte).conjugate()                  )
    dNtmdlna = -epstm*dN1dlna-0.5*w1/H*(  c1t*c1mc*Nmm + c1e*c1mc*Nte + c1m*c1mc*Ntm + c1mc*c1t*Ntt + c1t*c1tc*Ntm + c1t*c1ec*(Nme.conjugate()) ) - widtht*Ntm - widthm*Ntm
    dNtedlna = -epste*dN1dlna-0.5*w1/H*(  c1t*c1ec*Nee + c1e*c1ec*Nte + c1m*c1ec*Ntm + c1t*c1ec*Ntt + c1t*c1mc*Nme + c1t*c1tc*Nte               ) - widtht*Nte
    dNmedlna = -epsme*dN1dlna-0.5*w1/H*(  c1m*c1ec*Nee + c1e*c1ec*Nme + c1m*c1ec*Nmm + c1t*c1ec*(Ntm.conjugate())  + c1m*c1mc*Nme + c1m*c1tc*Nte) - widthm*Nme

    return [dN1dlna, dNttdlna, dNmmdlna, dNeedlna, dNtmdlna, dNtedlna, dNmedlna]

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

    def calculateTemps(self, N1, lna, nN_int, V):
        gst = self.ipol_gstar(self.inTsm*np.exp(-lna)) #gst estimate

        if self.kappa**3*np.abs(N1)/(nN_int)<10**(-7): #checks if number density is too small to use the inversion
            #set energy density, pressure and derivatives for RHN assuming the relativistic approximation or using the full expressions
            rhoN = self.M1*np.real(N1)
            w = 0
            rhoSM=np.exp(-4*lna)*(self.rho_in-rhoN*np.exp(lna*3*(1+w)))
            Tsm = (30/(np.pi**2*gst)*rhoSM)**(1./4.)
            Th = Tsm #set Th to Tsm as sectors have equilibrated (Th is meaningless at this point anyway and doesn't affect results)
        else:
            x=np.exp(-3*lna)*2*np.pi**2*np.abs(N1)/(self.gN*np.real(V)) #parameter for inversion
            Th = invIpp(x, self.M1, self.inTsm*np.exp(-lna))
            zh = self.M1/Th
            Jplus = Jp(zh)
            rhoN = self.gN/(2*np.pi**2)*Th**4*Jplus
            if Jplus == 0.:
                w=0
            else:
                w=Kp(zh)/Jplus

            rhoSM=np.exp(-4*lna)*(self.rho_in-rhoN*np.exp(lna*3*(1+w)))
            Tsm = (30/(np.pi**2*gst)*rhoSM)**(1./4.)

        rho = rhoSM+rhoN
        
        return Tsm, Th, rho

    def RHS(self, y0, lna, nN_int, epstt, epsmm, epsee,epstm,epste,epsme,c1t,c1m,c1e,k, V):
        N1 = y0[0]
        Tsm, Th, rho = self.calculateTemps(N1, lna, nN_int, V)

        zsm = self.M1/Tsm
        zh = self.M1/Th

        _d       = np.real(self.Gamma1* my_kn1(zh) / my_kn2(zh)) #decay rate thermal averaged with hot sector
        _invd = np.real(self.Gamma1* my_kn1(zsm) / my_kn2(zsm)) #decay rate thermal averaged with SM
        _w1      = _invd * 0.25 * my_kn2(zsm) * zsm**2 #washout rate
        nN_eq     = self.N1Eq(zsm) #equilibrium number density of neutrinos

        # thermal widths are set to zero such that we are in the "one-flavoured regime"
        widtht = 485e-10*self.MP/self.M1
        widthm = 1.7e-10*self.MP/self.M1
        
        return fast_RHS(y0, lna, rho, _d, _invd, _w1,  epstt, epsmm, epsee, epstm,epste,epsme,c1t,c1m,c1e, widtht, widthm, nN_eq, self.GCF)

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

        ggamma      = 2.

        self.gN=7/8*2. #RHN relativistic degrees of freedom
        self.inTsm      = 100. * self.M1 # initial temp 100x greater than mass of N1
        self.inTh = self.kappa*self.inTsm

        V = np.pi**2/(self.inTsm**3*zeta(3)*ggamma) #volume factor to normalise the number density, keeping it consistent with the equilibrium number density

        #define initial and final ln(a)
        lnain = 0.
        lnaf = 2*np.log(self.inTsm/self.M1)
        global lnarange
        lnarange=lnaf-lnain

        nN_int=3./4.*zeta(3)/(np.pi**2)*self.gN*self.inTh**3*V #initial RHN number density at temperature Th

        self.rho_in=np.pi**2/30.*(self.ipol_gstar(self.inTsm) *self.inTsm**4+(7./8.)*self.gN*self.inTh**4)

        y0      = np.array([nN_int+0j,0+0j,0+0j,0+0j,0+0j,0+0j,0+0j], dtype=np.complex128) #initial array
        nphi    = (2.*zeta(3)/np.pi**2) * self.inTsm**3
        params  = np.array([nN_int, epstt,epsmm,epsee,epstm,epste,epsme,c1t,c1m,c1e,k,V], dtype=np.complex128)
        
        lnsf = np.linspace(lnain, lnaf, num=100, endpoint=True)

        global pbar

        ys = odeintw(self.RHS, y0, lnsf, args = tuple(params)) #solves BEs

        self.T_arr=[]
        
        for i in range(0,100):
            N1=ys[i][0]
            lna = lnsf[i]
            Tsm, Th, rho = self.calculateTemps(N1, lna, nN_int, V)
            self.T_arr.append([Tsm,Th])

        self.T_arr=np.array(self.T_arr)

        T=self.T_arr[:,0]
        Th=self.T_arr[:,1]

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

        if showPlotBool:
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