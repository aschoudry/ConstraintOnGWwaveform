# This script computes final mass and kick velocity using global supermomentum law.
import sys
sys.path.insert(0, '../src')
import BOB_functions as BOB
import matplotlib.pyplot as plt
import numpy as np
import sxs
from scipy.interpolate import interp1d
import os
from EOBUtils import *
import scipy.special as sc

## Some good q=1 nospin BBH file nums (1132, 3, 4), (1155, 2, 3),(0002, 5,6), (1122, 4,5), spinning (0160, 3,4)
catalog = sxs.load("catalog")
file_num = '0002'

w_L3 = sxs.load("SXS:BBH:"+str(file_num)+"/Lev/rhOverM", extrapolation_order=2)

metadata = sxs.load("SXS:BBH:"+str(file_num)+"/Lev/metadata.json")

# Shift to some reference time

h22_SXS = np.array(w_L3[:, w_L3.index(2, 2)].data.tolist())
time_SXS = w_L3[:, w_L3.index(2, 2)].t

# Define the triple product of spherical harmonics
def fac(n):
    return np.math.factorial(n)

def gj(kj, lj, mj, sj):

    N= pow(-1.0, kj)*(fac(lj+mj)*fac(lj-mj)*fac(lj+sj)*fac(lj-sj))**(0.5)
    D= fac(kj)*fac(lj+mj-kj)*fac(lj-sj-kj)*fac(sj-mj+kj)
    return N/D

def G(s1,s2,s3,l1,l2,l3,m1,m2,m3):
    K1=pow(-1, s1+s2+s3)*np.sqrt((2*l1+1)*(2*l2+1)*(2*l3+1))/np.sqrt(4.0*np.pi)**3
    K2=2.0*np.pi*np.kron(-m1,m2+m3)
    K3=0.0
    k1i=max(0, m1-s1)
    k2i=max(0, m2-s2)
    k3i=max(0, m3-s3)
    k1f=min(l1+m1, l1-s1)
    k2f=min(l2+m2, l2-s2)
    k3f=min(l3+m3, l3-s3)
    
    k1_vec = np.arange(k1i, k1f+1)
    k2_vec = np.arange(k2i, k2f+1)

    k3_vec = np.arange(k3i, k3f+1)

    for k1 in k1_vec:
        for k2 in k2_vec:
            for k3 in k3_vec:
                p1=2.0*k1+s1-m1
                p2=2.0*k2+s2-m2
                p3=2.0*k3+s3-m3
                a=1.0+(p1+p2+p3)/2.0
                b=1.0+l1+l2+l3 - (p1+p2+p3)/2.0
                Beta_ab = sc.beta(a, b)
                if K2==0.0:
                    K3+=0.0
                else:
                    K3+=2.0*gj(k1, l1, m1, s1)*gj(k2, l2, m2, s2)*gj(k3, l3, m3, s3)*Beta_ab
    
    return K1*K2*K3

def alpha_SXS_lm(l,m, h):
    SXS_modes = h.LM
    alpha_lm = np.zeros(h.time.size,dtype=complex) 
    for i in SXS_modes:
        l1, m1 = i[0], i[1]
        hl1m1 = np.array(h[:, h.index(l1, m1)].data.tolist())
        phase1 = np.array(h.arg_unwrapped[:, h.index(l1, m1)].data.tolist())
        omega1 = np.gradient(phase1,h.t)
        hl1m1_dot = omega1*hl1m1
        for j in SXS_modes:
            l2, m2 = j[0], j[1]
            hl2m2_conj = np.array(h[:, h.index(l2, m2)].data.tolist()).conjugate()
            phase2 = np.array(h.arg_unwrapped[:, h.index(l2, m2)].data.tolist())
            omega2 = np.gradient(phase2,h.t)
            hl2m2_dot_conj=omega2*hl2m2_conj

            alpha = hl1m1_dot*hl2m2_dot_conj*G(2,-2,0,l1,l2,l,m1,-m2,-m)
            alpha_lm += alpha

    return alpha_lm

def Intgrl(t_vec, alpha):
    dt_vec = np.diff(t_vec)
    dt_vec = np.append(dt_vec, dt_vec[-1])
    Intgrl = np.cumsum((alpha)*dt_vec)
    return Intgrl

def Mf_and_af(alpha1, alpha2, nu):
    p0 = 0.04826; p1 = 0.01559; p2 = 0.00485; s4 = -0.1229; s5 = 0.4537; 
    t0 = -2.8904; t2 = -3.5171; t3 = 2.5763; q = 1.0; eta = nu; theta = np.pi/2 
    Mf = 1-p0 - p1*(alpha1+alpha2)-p2*pow(alpha1+alpha2,2)
    ab = (pow(q,2)*alpha1+alpha2)/(pow(q,2)+1)
    alpha = ab + s4*eta*pow(ab,2) + s5*pow(eta,2)*ab + t0*eta*ab + 2*np.sqrt(3)*eta + t2*pow(eta,2) + t3*pow(eta,3)
    OM_QNM = (1.0 - 0.63*pow(1.0 - alpha, 0.3))/(2*Mf)
    return alpha, Mf


# Parameter for TEOB code
nu = 0.25
alpha1=np.linalg.norm(metadata['initial_dimensionless_spin1'])
alpha2=np.linalg.norm(metadata['initial_dimensionless_spin2'])


# Generate data from TEOBRESUM

#!/usr/bin/python3

"""
Script to auto generate parfiles and run them

Given a template parfile with basic setup, 
generates parfiles for all combination of a given subset of parameters
(e.g. to vary binary masses and spins). Then it run the code.

SB 10/2018
"""
###############################################################################################################
import os

from EOBUtils import *

if __name__ == "__main__": 


    # Setup -----------------------------------------
    
    # Base dir & parfile
    based = "./"
    basep = "test_NQCIteration.par"

    # Set new values/ranges for parameters (Use lists)
    q = [1.]
    chi1 = [alpha1]
    chi2 = [alpha2]
    # Pack them into a dictionary
    # NOTE: keys must match those in parfile otherwise ignored
    n = {'q': q,
         'chi1': chi1,
         'chi2': chi2}
    
    # ------------------------------------------
    # DO NOT CHANGE BELOW HERE
    # ------------------------------------------

    # Generate parfiles ----------------------------
    
    # Read the base parfile
    d = read_parfile_dict(basep)

    # Generate combinations
    x, keys = combine_parameters(n)
    
    # Write parfiles
    parfile = []
    basen, ext = os.path.splitext(basep)
    for s in range(len(x)):
        for i in range(len(keys)):
            d[keys[i]] = str(x[s][i])
        # Output to file
        print(d)
        parfile.append(based+"/"+basen+"_"+str(s)+ext)
        write_parfile_dict(parfile[-1], d)
        print("Written {}".format(s))

    # Run  ----------------------------

    for p in parfile:
        run(p)
        os.remove(p)

def M_SXS(M0, h):
    SXS_modes = h.LM
    alpha = np.zeros(h.time.size,dtype=complex) 
    for i in SXS_modes:
        l1, m1 = i[0], i[1]
        hl1m1 = np.array(h[:, h.index(l1, m1)].data.tolist())
        phase1 = np.array(h.arg_unwrapped[:, h.index(l1, m1)].data.tolist())
        omega1 = np.gradient(phase1,h.t)
        hl1m1_dot = omega1*hl1m1
        hl1m1_dot_sqr = abs(hl1m1_dot)**2
        alpha += hl1m1_dot_sqr
    
    Int_alpha = Intgrl(h.t, alpha)

    Mf = M0 - Int_alpha/(16.0*np.pi)

    return Mf

def Vx_SXS(M0,h):
    SXS_modes = [[2,2],[2,1],[2,0],[2,-1],[2,-2]] #h.LM
    alpha_lm = np.zeros(h.time.size,dtype=complex) 
    for i in SXS_modes:
        l1, m1 = i[0], i[1]
        hl1m1 = np.array(h[:, h.index(l1, m1)].data.tolist())
        phase1 = np.array(h.arg_unwrapped[:, h.index(l1, m1)].data.tolist())
        omega1 = np.gradient(phase1,h.t)
        hl1m1_dot = 1j*omega1*hl1m1
        for j in SXS_modes:
            l2, m2 = j[0], j[1]
            hl2m2_conj = np.array(h[:, h.index(l2, m2)].data.tolist()).conjugate()
            phase2 = np.array(h.arg_unwrapped[:, h.index(l2, m2)].data.tolist())
            omega2 = np.gradient(phase2,h.t)
            hl2m2_dot_conj=1j*omega2*hl2m2_conj

            alpha = (-1.0)**(1+m2)*hl1m1_dot*hl2m2_dot_conj*(G(2,-2,0,l1,l2,1,m1,-m2,1) - G(2,-2,0,l1,l2,1,m1,-m2,-1))
            alpha_lm += alpha
        
    Int_alpha_lm = Intgrl(h.t, alpha_lm)
    vx = -(1.0/8.0/np.sqrt(6.0*np.pi))*Int_alpha_lm/M_SXS(M0, h)
    return vx 

def Vy_SXS(M0,h):
    SXS_modes = [[2,2],[2,1],[2,0],[2,-1],[2,-2]] #h.LM
    alpha_lm = np.zeros(h.time.size,dtype=complex) 
    for i in SXS_modes:
        l1, m1 = i[0], i[1]
        hl1m1 = np.array(h[:, h.index(l1, m1)].data.tolist())
        phase1 = np.array(h.arg_unwrapped[:, h.index(l1, m1)].data.tolist())
        omega1 = np.gradient(phase1,h.t)
        hl1m1_dot = 1j*omega1*hl1m1
        for j in SXS_modes:
            l2, m2 = j[0], j[1]
            hl2m2_conj = np.array(h[:, h.index(l2, m2)].data.tolist()).conjugate()
            phase2 = np.array(h.arg_unwrapped[:, h.index(l2, m2)].data.tolist())
            omega2 = np.gradient(phase2,h.t)
            hl2m2_dot_conj=1j*omega2*hl2m2_conj

            alpha = (-1.0)**(1+m2)*hl1m1_dot*hl2m2_dot_conj*(G(2,-2,0,l1,l2,1,m1,-m2,1) + G(2,-2,0,l1,l2,1,m1,-m2,-1))
            alpha_lm += alpha
        
    Int_alpha_lm = Intgrl(h.t, alpha_lm)
    vy = -1j*(1.0/8.0/np.sqrt(6.0*np.pi))*Int_alpha_lm/M_SXS(M0, h)
    return vy 

def Vz_SXS(M0,h):
    SXS_modes = [[2,2],[2,1],[2,0],[2,-1],[2,-2]] #h.LM
    alpha_lm = np.zeros(h.time.size,dtype=complex) 
    for i in SXS_modes:
        l1, m1 = i[0], i[1]
        hl1m1 = np.array(h[:, h.index(l1, m1)].data.tolist())
        phase1 = np.array(h.arg_unwrapped[:, h.index(l1, m1)].data.tolist())
        omega1 = np.gradient(phase1,h.t)
        hl1m1_dot = 1j*omega1*hl1m1
        for j in SXS_modes:
            l2, m2 = j[0], j[1]
            hl2m2_conj = np.array(h[:, h.index(l2, m1)].data.tolist()).conjugate()
            phase2 = np.array(h.arg_unwrapped[:, h.index(l2, m1)].data.tolist())
            omega2 = np.gradient(phase2,h.t)
            hl2m2_dot_conj=1j*omega2*hl2m2_conj

            alpha = (-1.0)**(m1)*hl1m1_dot*hl2m2_dot_conj*(G(2,-2,0,l1,l2,1,m1,-m1,1))
            alpha_lm += alpha
        
    Int_alpha_lm = Intgrl(h.t, alpha_lm)
    vz = -(1.0/8.0/np.sqrt(3.0*np.pi))*Int_alpha_lm/M_SXS(M0, h)
    return vz 

#####################################################################################################################################
## Load TeobResumm data for all hlm modes

def alpha_TeobResumm_lm(l, m, lmax):
    t, hp, hc = np.loadtxt('/home/ashok/constraintongwwaveform/script/output/hlm_interp_l2_m2_reim.txt', unpack=True)
    alpha_lm = np.zeros(len(hp),dtype=complex) 
    l_teob_vec = np.arange(2, lmax+1)

    for l1 in l_teob_vec:
        m1_vec =np.append(np.arange(-l1,0),np.arange(1, l1+1))
        for m1 in m1_vec:
            t, hp, hc = np.loadtxt('/home/ashok/constraintongwwaveform/script/output/hlm_interp_l'+str(l1)+'_m'+str(abs(m1))+'_reim.txt', unpack=True)
            hp_dot = np.gradient(hp,t)
            hc_dot = np.gradient(hc,t)
            if m1<0:
                hl1m1 = pow(-1,l1)*(hp_dot - 1j*hc_dot)
            else:
                hl1m1 = hp_dot + 1j*hc_dot

            for l2 in l_teob_vec:
                m2_vec = np.append(np.arange(-l2,0),np.arange(1, l2+1))
                for m2 in m2_vec:
                    t, hp, hc = np.loadtxt('/home/ashok/constraintongwwaveform/script/output/hlm_interp_l'+str(l2)+'_m'+str(abs(m2))+'_reim.txt', unpack=True)
                    hp_dot = np.gradient(hp,t)
                    hc_dot = np.gradient(hc,t)

                    if m2<0:
                        hl2m2 = pow(-1,l2)*(hp_dot - 1j*hc_dot)
                    else:
                        hl2m2 = hp_dot + 1j*hc_dot

                    hl2m2_conj = hl2m2.conj()
                    alpha = hl1m1*hl2m2_conj*G(2,-2,0,l1,l2,l,m1,-m2,-m)/np.sqrt((l1+2)*(l1+1)*(l1)*(l1-1))/np.sqrt((l2+2)*(l2+1)*(l2)*(l2-1))
                    alpha_lm += alpha
    return alpha_lm

t, hp, hc = np.loadtxt('/home/ashok/constraintongwwaveform/script/output/hlm_interp_l2_m2_reim.txt', unpack=True)
#print(alpha_SXS_lm(1,0, w_L3))
#print(1.0 - Intgrl(w_L3.t, alpha_SXS_lm(1,0, w_L3))*(1.0/(16.0*np.pi))) 
#print(1.0 - Intgrl(t, alpha_TeobResumm_lm(1,0,8))*(1.0/(16.0*np.pi)))

#vx_sxs  = Intgrl(w_L3.t, alpha_SXS_lm(1,-1, w_L3)-alpha_SXS_lm(1,1, w_L3))*(1.0/(16.0*np.pi))
#vy_sxs  = Intgrl(w_L3.t, alpha_SXS_lm(1,-1, w_L3))*(1.0/(16.0*np.pi))
#vz_sxs  = Intgrl(w_L3.t, alpha_SXS_lm(1,0, w_L3))*(1.0/(16.0*np.pi))

#vx_eob  = Intgrl(t, alpha_TeobResumm_lm(1,1,8)-alpha_TeobResumm_lm(1,-1,8))*(1.0/(16.0*np.pi))
#vy_eob = Intgrl(t, alpha_TeobResumm_lm(1,-1,8))*(1.0/(16.0*np.pi))
#vz_eob  =Intgrl(t, alpha_TeobResumm_lm(1,0,8))*(1.0/(16.0*np.pi))

#print('vx_sxs', vx_sxs, 'vy_sxs', vy_sxs, 'vz_sxs', vz_sxs) 
#print('vx_eob', vx_eob, 'vy_eob', vy_eob, 'vz_eob', vz_eob) 

#print(Intgrl(t, alpha_TeobResumm_lm(1,0,8))*(1.0/(16.0*np.pi)))

print(M_SXS(1.0, w_L3))
print(Vx_SXS(1.0, w_L3))
print(Vy_SXS(1.0, w_L3))
print(Vz_SXS(1.0, w_L3))

print(Mf_and_af(alpha1, alpha2, nu))
print('remnant_mass',metadata['remnant_mass'])
print('remnant_velocity',metadata['remnant_velocity'])
#print(alpha_SXS_lm(0,0, w_L3))
#print(alpha_TeobResumm_lm(0,0,8))

#plt.plot(abs(Vx_SXS(1.0, w_L3)))
#plt.show()
