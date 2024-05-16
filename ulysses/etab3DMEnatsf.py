# non-resonant leptogenesis with three decaying sterile neutrino using the density matrix equations. Equations from 1112.4528
import ulysses
import numpy as np
from scipy import interpolate
from scipy.special import zeta
from scipy.special import kn
from scipy.optimize import fsolve
from scipy.integrate import quad
from ulysses.ulsbase import my_kn2, my_kn1
import matplotlib.pyplot as plt
from odeintw import odeintw

from ulysses.numba import jit

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

#@jit
def fast_RHS(y0,lna,M1,M2,M3,gst,gN,GCF,V,eps1tt,eps1mm,eps1ee,eps1tm,eps1te,eps1me,eps2tt,eps2mm,eps2ee,eps2tm,eps2te,eps2me,eps3tt,eps3mm,eps3ee,eps3tm,eps3te,eps3me, C, W, d1,invd1,d2,d3,w1,w2,w3,n1eq,n2eq,n3eq):
    N1, N2, N3, Ntt, Nmm, Nee, Ntm, Nte, Nme, Tsm, Th = y0

    Tsm = np.abs(Tsm)
    Th = np.abs(Th)

    zh=M1/Th
    zsm=M1/Tsm

    N1_eq_SM = n1eq

    c1t,c1m,c1e,c2t,c2m,c2e,c3t,c3m,c3e = C
    widtht,widthm = W
    c1tc    = np.conjugate(c1t)
    c1mc    = np.conjugate(c1m)
    c1ec    = np.conjugate(c1e)

    c2tc    = np.conjugate(c2t)
    c2mc    = np.conjugate(c2m)
    c2ec    = np.conjugate(c2e)

    c3tc    = np.conjugate(c3t)
    c3mc    = np.conjugate(c3m)
    c3ec    = np.conjugate(c3e)

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

    if Tsm > M2:
        gst += 7./8.*gN

    if Tsm > M3:
        gst += 7./8.*gN

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

    #define the different RHSs for each equation
    dN1dlna      =    (-np.exp(3*lna)*N1*d1/H +  (N1_eq_SM)*invd1/H) #RHN number density BE - for comoving N1
    dN2dlna    =      - d2/H * (N2-n2eq)
    dN3dlna    =      - d3/H * (N3-n3eq)

    dQdlna = -M1*dN1dlna/V #set energy transfer rate

    dNttdlna    = (eps1tt * (-dN1dlna) + eps2tt * (-dN2dlna) + eps3tt * (-dN3dlna)
            - 0.5 * w1/H * (2 * c1t * c1tc * Ntt + c1m * c1tc * Ntm + c1e * c1tc * Nte + np.conjugate(c1m * c1tc * Ntm + c1e * c1tc * Nte))
            - 0.5 * w2/H * (2 * c2t * c2tc * Ntt + c2m * c2tc * Ntm + c2e * c2tc * Nte + np.conjugate(c2m * c2tc * Ntm + c2e * c2tc * Nte))
            - 0.5 * w3/H * (2 * c3t * c3tc * Ntt + c3m * c3tc * Ntm + c3e * c3tc * Nte + np.conjugate(c3m * c3tc * Ntm + c3e * c3tc * Nte)))

    dNmmdlna    = (eps1mm * (-dN1dlna) + eps2mm * (-dN2dlna) + eps3mm * (-dN3dlna)
            - 0.5 * w1/H * (2 * c1m * c1mc * Nmm + c1m * c1tc * Ntm + c1e * c1mc * Nme + np.conjugate(c1m * c1tc * Ntm + c1e * c1mc * Nme))
            - 0.5 * w2/H * (2 * c2m * c2mc * Nmm + c2m * c2tc * Ntm + c2e * c2mc * Nme + np.conjugate(c2m * c2tc * Ntm + c2e * c2mc * Nme))
            - 0.5 * w3/H * (2 * c3m * c3mc * Nmm + c3m * c3tc * Ntm + c3e * c3mc * Nme + np.conjugate(c3m * c3tc * Ntm + c3e * c3mc * Nme)))

    dNeedlna    = (eps1ee * (-dN1dlna) + eps2ee * (-dN2dlna) + eps3ee * (-dN3dlna)
            - 0.5 * w1/H * (2 * c1e * c1ec * Nee + c1e * c1mc * Nme + c1e * c1tc * Nte + np.conjugate(c1e * c1mc * Nme + c1e * c1tc * Nte))
            - 0.5 * w2/H * (2 * c2e * c2ec * Nee + c2e * c2mc * Nme + c2e * c2tc * Nte + np.conjugate(c2e * c2mc * Nme + c2e * c2tc * Nte))
            - 0.5 * w3/H * (2 * c3e * c3ec * Nee + c3e * c3mc * Nme + c3e * c3tc * Nte + np.conjugate(c3e * c3mc * Nme + c3e * c3tc * Nte)))

    dNtmdlna    = (eps1tm * (-dN1dlna) + eps2tm * (-dN2dlna) + eps3tm * (-dN3dlna)
            - 0.5/H*((w1 * c1t * c1mc + w2 * c2t * c2mc + w3 * c3t * c3mc) * Nmm
            +        (w1 * c1e * c1mc + w2 * c2e * c2mc + w3 * c3e * c3mc) * Nte
            +        (w1 * c1m * c1mc + w2 * c2m * c2mc + w3 * c3m * c3mc) * Ntm
            +        (w1 * c1mc * c1t + w2 * c2mc * c2t + w3 * c3mc * c3t) * Ntt
            +        (w1 * c1t * c1tc + w2 * c2t * c2tc + w3 * c3t * c3tc + 2 * widtht + 2 * widthm) * Ntm
            +        (w1 * c1t * c1ec + w2 * c2t * c2ec + w3 * c3t * c3ec) * np.conjugate(Nme)))

    dNtedlna    = (eps1te * (-dN1dlna) + eps2te * (-dN2dlna) + eps3te * (-dN3dlna)
            - 0.5/H*((w1 * c1t * c1ec + w2 * c2t * c2ec + w3 * c3t * c3ec) * Nee
            +        (w1 * c1e * c1ec + w2 * c2e * c2ec + w3 * c3e * c3ec) * Nte
            +        (w1 * c1m * c1ec + w2 * c2m * c2ec + w3 * c3m * c3ec) * Ntm
            +        (w1 * c1t * c1ec + w2 * c2t * c2ec + w3 * c3t * c3ec) * Ntt
            +        (w1 * c1t * c1mc + w2 * c2t * c2mc + w3 * c3t * c3mc) * Nme
            +        (w1 * c1t * c1tc + w2 * c2t * c2tc + w3 * c3t * c3tc + 2 * widtht) * Nte))

    dNmedlna    = (eps1me * (-dN1dlna) + eps2me * (-dN2dlna) + eps3me * (-dN3dlna)
            - 0.5/H * ((w1 * c1m * c1ec + w2 * c2m * c2ec + w3 * c3m * c3ec) * Nee
            +        (w1 * c1e * c1ec + w2 * c2e * c2ec + w3 * c3e * c3ec + 2 * widthm) * Nme
            +        (w1 * c1m * c1ec + w2 * c2m * c2ec + w3 * c3m * c3ec) * Nmm
            +        (w1 * c1t * c1ec + w2 * c2t * c2ec + w3 * c3t * c3ec) * np.conjugate(Ntm)
            +        (w1 * c1m * c1mc + w2 * c2m * c2mc + w3 * c3m * c3mc) * Nme
            +        (w1 * c1m * c1tc + w2 * c2m * c2tc + w3 * c3m * c3tc) * Nte))

    dN1dlna = np.exp(-3*lna)*dN1dlna - 3*N1 #rewriting the derivative in terms of the non-comoving N1

    deltaSM = drhoSMdTsm+dpSMdTsm

    denomSM = deltaSM - sSM

    dTsmdlna = (np.exp(-3*lna)*dQdlna-3*sSM*Tsm)/denomSM #SM temperature derivative from second law of thermodynamics

    dThdlna = (np.exp(-3*lna)*dQdlna - 3*(rhoSM+pSM) - drhoSMdTsm*dTsmdlna + dN1dlna*pN/N1)/(pN/N1_eq_hot*dN1_eq_hotdTh+sN-dpNdTh) #hot sector temp derivative via second law + comoving energy conservation

    return [dN1dlna, dN2dlna, dN3dlna, dNttdlna, dNmmdlna, dNeedlna, dNtmdlna, dNtedlna, dNmedlna, dTsmdlna, dThdlna]


class EtaB_3DMEsf(ulysses.ULSBase):
    """
    Density matrix equation (DME) with three decaying steriles. See arxiv:1112.4528.
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

    def shortname(self): return "3DME"
    def flavourindices(self): return [3, 4, 5]
    def flavourlabels(self): return ["$N_{\\tau\\tau}$", "$N_{\mu\mu}$", "$N_{ee}$"]

    def sigma(self, s,mH,mN,yH):
        hDiff = mH**2-s
        hSum = mH**2+s
        prefactor = yH**4/(16*np.pi*mH**2*s**2*hDiff**2*hSum)
        term1 = mH**2*hDiff*hSum*(-mN**4 + mN**2*(mH**2+5*s) + 2*mH**4 - mH**2*s)*np.log(mH**2/hSum)
        term2 = s*(mN**4 * (2*mH**4 - mH**2*s + s**2) + mN**2 *(mH**6 + 2*mH**4 *s - 7*mH**2 * s**2) + mH**2 *(2*mH**6 - 2*mH**4 * s + mH**2*s**2 + s**3))

        return prefactor*(term1 + term2)

    def sv(self,Th,Tsm, mH,mN,yH):
        A = mN**2.*Tsm**2.
        B = lambda s: mN**2.*Tsm*(Tsm-Th)+s*Th*Tsm
        C = lambda s: s-mN**2.
        z = lambda s: np.sqrt(B(s))/(Th*Tsm)
        integrand = lambda s: 1./(16*Tsm**2*mN**2*my_kn2(mN/Th))*self.sigma(s,mH,mN,yH)*C(s)/B(s)*(A*(1+z(s))*np.exp(-z(s))+C(s)*np.sqrt(B(s))*my_kn1(z(s)))
        return quad(integrand, mN**2, cutoff*mN**2,epsabs=absErr, epsrel = relErr)[0]

    def RHS(self, y0, lna, nN_int, V, ETA, _C, K, _W):
        N1, N2, N3, Ntt, Nmm, Nee, Ntm, Nte, Nme, Tsm, Th = y0

        Tsm = np.abs(Tsm)
        Th = np.abs(Th)

        zh=self.M1/Th
        zsm=self.M1/Tsm

        (eps1tt,eps1mm,eps1ee,eps1tm,eps1te,eps1me,eps2tt,eps2mm,eps2ee,eps2tm,eps2te,eps2me,eps3tt,eps3mm,eps3ee,eps3tm,eps3te,eps3me) = ETA
        c1t,c1m,c1e,c2t,c2m,c2e,c3t,c3m,c3e = _C
        from ulysses.numba import List
        C=List()
        [C.append(c) for c in _C]

        k1term,k2term,k3term = K
        widtht,widthm = _W
        W=List()
        [W.append(w) for w in _W]

        self._d1       = np.real(self.Gamma1* my_kn1(zh) / my_kn2(zh)) #decay rate thermal averaged with hot sector
        self._invd1 = np.real(self.Gamma1* my_kn1(zsm) / my_kn2(zsm)) #decay rate thermal averaged with SM
        self._w1      = self._invd1 * 0.25 * my_kn2(zsm) * zsm**2 #washout rate
        self._d2      = np.real(self.D2(k2term, zsm))
        self._w2      = np.real(self.W2(k2term, zsm))
        self._d3      = np.real(self.D3(k3term, zsm))
        self._w3      = np.real(self.W3(k3term, zsm))
        self._n1eq    = self.N1Eq(zsm)
        self._n2eq    = self.N2Eq(zsm)
        self._n3eq    = self.N3Eq(zsm)


        if(np.log10(np.exp(3*lna)*np.real(N1)/nN_int) < -6):
            self.evolEnd = True

        if self.evolEnd:
            return [-3*np.abs(N1), 0, 0, 0, 0, 0, 0, 0, 0, -Tsm, -Th]
        else:
            return fast_RHS(y0,lna,self.M1,self.M2,self.M3,self.ipol_gstar(Tsm),self.gN,self.GCF,V,eps1tt,eps1mm,eps1ee,eps1tm,eps1te,eps1me,eps2tt,eps2mm,eps2ee,eps2tm,eps2te,eps2me,eps3tt,eps3mm,eps3ee,eps3tm,eps3te,eps3me, C, W,
                self._d1,self._invd1,self._d2,self._d3,self._w1,self._w2,self._w3,self._n1eq,self._n2eq,self._n3eq)

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
    def EtaB(self):

        #Define fixed quantities for BEs
        _ETA = [
            np.real(self.epsilon1ab(2,2)),
            np.real(self.epsilon1ab(1,1)),
            np.real(self.epsilon1ab(0,0)),
                    self.epsilon1ab(2,1) ,
                    self.epsilon1ab(2,0) ,
                    self.epsilon1ab(1,0) ,
            np.real(self.epsilon2ab(2,2)),
            np.real(self.epsilon2ab(1,1)),
            np.real(self.epsilon2ab(0,0)),
                    self.epsilon2ab(2,1) ,
                    self.epsilon2ab(2,0) ,
                    self.epsilon2ab(1,0) ,
            np.real(self.epsilon3ab(2,2)),
            np.real(self.epsilon3ab(1,1)),
            np.real(self.epsilon3ab(0,0)),
                    self.epsilon3ab(2,1) ,
                    self.epsilon3ab(2,0) ,
                    self.epsilon3ab(1,0)]

        _C =   [self.c1a(2), self.c1a(1), self.c1a(0),
                self.c2a(2), self.c2a(1), self.c2a(0),
                self.c3a(2), self.c3a(1), self.c3a(0)]

        _K      = [np.real(self.k1), np.real(self.k2), np.real(self.k3)]
        _W      = [ 485e-10*self.MP/self.M1, 1.7e-10*self.MP/self.M1]

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

        N_eq_SM=1/(2*np.pi**2)*self.gN*self.inTsm**3*besselLimit*V #initial RHN number density at temperature Th

        nN_int = self.f*N1_eq_hot

        n23_int = N_eq_SM

        self.rho_in=np.pi**2/30.*(self.ipol_gstar(self.inTsm)*self.inTsm**4+(7./8.)*self.f*self.gN*self.inTh**4)

        y0      = np.array([nN_int+0j,N_eq_SM+0j,N_eq_SM+0j,0+0j,0+0j,0+0j,0+0j,0+0j,0+0j, self.inTsm, self.inTh], dtype=np.complex128) #initial array
        nphi    = (2.*zeta(3)/np.pi**2) * self.inTsm**3

        #ys, _   = odeintw(self.RHS, y0, self.zs, args = tuple([_ETA, _C , _K, _W]), full_output=1)
        params  = np.array([nN_int,V], dtype=np.complex128)
        
        lnsf = np.linspace(lnain, lnaf, num=100, endpoint=True)

        ys = odeintw(self.RHS, y0, lnsf, args = tuple([nN_int,V,_ETA, _C , _K, _W])) #solves BEs

        T=np.abs(ys[:,9])
        Th=np.abs(ys[:,10])

        gstarSrec = self.ipol_gstarS(0.3e-9) # d.o.f. at recombination
        gstarSoff = self.ipol_gstarS(T[-1])  # d.o.f. at the end of leptogenesis
        SMspl       = 28./79.
        zeta3       = zeta(3)
        
        coeffNgamma = ggamma*zeta3/np.pi**2

        Ngamma      = coeffNgamma*(np.exp(lnsf)*T)**3
        coeffsph    =  SMspl * gstarSrec/gstarSoff

        NBL=np.real(ys[:,3]+ys[:,4]+ys[:,5])

        etab = coeffsph*NBL*nphi/Ngamma

        zsm=self.M1/T

        zh=self.M1/Th

        d       = np.real(self.Gamma1* kn(1,zh) / kn(2,zh)) #decay rate thermal averaged with hot sector

        nH = ys[:,0]

        yH = np.amax(np.abs(np.transpose(self.h)[0]))

        H = []

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

            H.append(Hubble)

            leptogst = 3/4*2*6

            nSM = np.real(zeta3/(np.pi**2)*leptogst*T[i]**3)

            l         = self.h
            ldag      = np.conjugate(np.transpose(l))
            lcon      = np.conjugate(l)
            M         = self.DM
            lsquare   = np.dot(ldag,l)

            GammaScatt = np.abs(5*10.**(-3.)*lsquare[0,0]*Th[i])

            n = nH[i]

            GammaTherm = np.real(self.sv(np.real(Th[i]),np.real(T[i]),125,self.M1,yH)*nSM)

            if ((GammaTherm > d[i]) and (GammaTherm > Hubble) and (np.real(Th[i]/self.M1) > 0.1)):
                eqBool = True
            
            if ((GammaScatt > Hubble) and (np.real(Th[i]/self.M1) > 1)):
                eqBool = True

        
        if(eqBool):
            return -np.abs(etab[-1])
        else:
            return np.abs(etab[-1])
