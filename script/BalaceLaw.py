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
#kick '0303'
catalog = sxs.load("catalog")
file_num = '0001'

w_L3 = sxs.load("SXS:BBH:"+str(file_num)+"/Lev/rhOverM", extrapolation_order=2)

metadata = sxs.load("SXS:BBH:"+str(file_num)+"/Lev/metadata.json")

# Shift to some reference time

h22_SXS = np.array(w_L3[:, w_L3.index(2, 2)].data.tolist())
time_SXS = w_L3[:, w_L3.index(2, 2)].t

# Parameter for TEOB code
nu = 0.25
alpha1=np.linalg.norm(metadata['initial_dimensionless_spin1'])
alpha2=np.linalg.norm(metadata['initial_dimensionless_spin2'])

# Compute quasinormal frrequencies given initial spins and symmetric mass ratio for BOB

def Mf_and_af(alpha1, alpha2, nu):
    p0 = 0.04826; p1 = 0.01559; p2 = 0.00485; s4 = -0.1229; s5 = 0.4537; 
    t0 = -2.8904; t2 = -3.5171; t3 = 2.5763; q = 1.0; eta = nu; theta = np.pi/2 
    Mf = 1-p0 - p1*(alpha1+alpha2)-p2*pow(alpha1+alpha2,2)
    ab = (pow(q,2)*alpha1+alpha2)/(pow(q,2)+1)
    alpha = ab + s4*eta*pow(ab,2) + s5*pow(eta,2)*ab + t0*eta*ab + 2*np.sqrt(3)*eta + t2*pow(eta,2) + t3*pow(eta,3)
    OM_QNM = (1.0 - 0.63*pow(1.0 - alpha, 0.3))/(2*Mf)
    return alpha, Mf


print(alpha1, alpha2)
af, Mf = Mf_and_af(alpha1, alpha2, nu)

# BOB waveform Final state variable
M_tot = 1.0
t_scale = M_tot*4.92549094830932e-06 

# quasinormal, tau frequency
om_qnm = 1.5251-1.1568*(1-af)**0.1292
om_qnm = om_qnm/Mf
Q = 0.7+1.4187*(1-af)**(-0.499)
OM_QNM = om_qnm/2
tau = 2*Q/om_qnm

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

#####################################################################################################################################
## Load TeobResumm data for all hlm modes


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


# Mass and recoil velocities using SXS data

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

            alpha = hl1m1_dot*hl2m2_dot_conj*((-1.0)**(1+m2)*G(2,-2,0,l1,l2,1,m1,-m2,1) + (-1.0)**(-1+m2)*G(2,-2,0,l1,l2,1,m1,-m2,-1))
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

            alpha = hl1m1_dot*hl2m2_dot_conj*((-1.0)**(1+m2)*G(2,-2,0,l1,l2,1,m1,-m2,1) +(-1.0)**(-1+m2)*G(2,-2,0,l1,l2,1,m1,-m2,-1))
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

            alpha = (-1.0)**(m1)*hl1m1_dot*hl2m2_dot_conj*(G(2,-2,0,l1,l2,1,m1,-m1,0))
            alpha_lm += alpha
        
    Int_alpha_lm = Intgrl(h.t, alpha_lm)
    vz = -(1.0/8.0/np.sqrt(3.0*np.pi))*Int_alpha_lm/M_SXS(M0, h)
    return vz 

def Psi2_SXS(theta, phi, M0, h):
    vx = Vx_SXS(M0,h)
    vy = Vy_SXS(M0,h)
    vz = Vz_SXS(M0,h)
    M = M_SXS(M0, h)
    v = np.sqrt(abs(vx)**2 + abs(vy)**2 + abs(vz)**2)
    gamma = 1.0/np.sqrt(1-v*v)
    psi = -(M/gamma**3)/(1.0 - vx*np.sin(theta)*np.cos(phi) - vy*np.sin(theta)*np.sin(phi) - vz*np.cos(theta))**3

    return psi

def m2Y22(theta,phi):
    return (1.0/8)*np.sqrt(5.0/np.pi)*(1.0+np.cos(theta))**2 *np.exp(1j*2*phi)

def Y22(theta, phi):
    return (1.0/4.0)*np.sqrt(15.0/2.0/np.pi)*np.sin(theta)**2*np.exp(1j*2*phi)

def Int_h_dot_SXS(theta, phi, h):
    l1=2
    m1=2
    hl1m1 = np.array(h[:, h.index(l1, m1)].data.tolist())
    phase1 = np.array(h.arg_unwrapped[:, h.index(l1, m1)].data.tolist())
    omega1 = np.gradient(phase1,h.t)
    hl1m1_dot = 1j*omega1*hl1m1
    h_dot_sqr = abs(hl1m1_dot*m2Y22(theta,phi))**2
    int_h_dot_sqr = Intgrl(h.t, h_dot_sqr)
    return int_h_dot_sqr

def Balance_law(theta, phi, M0, h):
    l1=2
    m1=2
    hl1m1 = np.array(h[:, h.index(l1, m1)].data.tolist())

    eth_2_delta_h  =  Y22(theta, phi)*hl1m1
    psi2 = Psi2_SXS(theta, phi, M0, h)
    eth_2_delta_h  = eth_2_delta_h  - eth_2_delta_h[0]
    psi2 = psi2 - psi2[0]
    int_h_dot = Int_h_dot_SXS(theta, phi, h)
    int_h_dot = int_h_dot - int_h_dot[0]

    return eth_2_delta_h - psi2 + int_h_dot

##### teobresumm #############################

def M_teob(M0, lmax):
    t, hp, hc = np.loadtxt('/home/ashok/constraintongwwaveform/script/output/hlm_interp_l2_m2_reim.txt', unpack=True)
    l_teob_vec = np.arange(2, lmax+1)
    alpha = np.zeros(len(hp),dtype=complex)  
    
    for l1 in l_teob_vec:
        m1_vec =np.append(np.arange(-l1,0),np.arange(1, l1+1))
        for m1 in m1_vec:
            t, hp, hc = np.loadtxt('/home/ashok/constraintongwwaveform/script/output/hlm_interp_l'+str(l1)+'_m'+str(abs(m1))+'_reim.txt', unpack=True)


            if m1<0:
                hl1m1 = pow(-1,l1)*(hp - 1j*hc)
            else:
                hl1m1 = hp + 1j*hc

            hl1m1 = hl1m1

            phase1 = np.unwrap(np.angle(hl1m1))
            omega1 = np.gradient(phase1, t)
            hl1m1_dot = 1j*omega1*hl1m1


            alpha_lm = abs(hl1m1_dot)**2/16.0
            alpha += alpha_lm

    Int_alpha = Intgrl(t, alpha)
    Mf = M0 - Int_alpha/(16.0*np.pi)
    return t, Mf

def Vx_teob(M0, lmax):
    t, hp, hc = np.loadtxt('/home/ashok/constraintongwwaveform/script/output/hlm_interp_l2_m2_reim.txt', unpack=True)
    l_teob_vec = np.arange(2, lmax+1)
    alpha = np.zeros(len(hp),dtype=complex)  

    for l1 in l_teob_vec:
        m1_vec =np.append(np.arange(-l1,0),np.arange(1, l1+1))
        for m1 in m1_vec:
            t, hp, hc = np.loadtxt('/home/ashok/constraintongwwaveform/script/output/hlm_interp_l'+str(l1)+'_m'+str(abs(m1))+'_reim.txt', unpack=True)


            if m1<0:
                hl1m1 = pow(-1,l1)*(hp - 1j*hc)
            else:
                hl1m1 = hp + 1j*hc

            hl1m1 = hl1m1

            phase1 = np.unwrap(np.angle(hl1m1))
            omega1 = np.gradient(phase1, t)
            hl1m1_dot = np.array(1j*omega1*hl1m1)
            
            for l2 in l_teob_vec:
                m2_vec =np.append(np.arange(-l2,0),np.arange(1, l2+1))
                for m2 in m2_vec:
                    t, hp, hc = np.loadtxt('/home/ashok/constraintongwwaveform/script/output/hlm_interp_l'+str(l2)+'_m'+str(abs(m2))+'_reim.txt', unpack=True)


                    if m2<0:
                        hl2m2 = pow(-1,l2)*(hp - 1j*hc)
                    else:
                        hl2m2 = hp + 1j*hc

                    hl2m2 = hl2m2

                    phase2 = np.unwrap(np.angle(hl2m2))
                    omega2 = np.gradient(phase2, t)
                    hl2m2_dot = np.array(1j*omega2*hl2m2)

                    alpha_lm = hl1m1_dot*np.conjugate(hl2m2_dot)*((-1.0)**(1+m2)*G(2,-2,0,l1,l2,1,m1,-m2,1) + (-1.0)**(-1+m2)*G(2,-2,0,l1,l2,1,m1,-m2,-1))/16.0
                    alpha += alpha_lm
    Int_alpha_lm = Intgrl(t, alpha)
    vx = -(1.0/8.0/np.sqrt(6.0*np.pi))*Int_alpha_lm/M_teob(M0, lmax)[1]
    
    return t, vx


### BOB waveform ##########################################################
# Initial condition for BOB using various waveforms

t_2_h22, hp_2_h22, hc_2_h22 = np.loadtxt('/home/ashok/constraintongwwaveform/script/output/hlm_interp_l2_m2_reim.txt', unpack=True)

t_shift_2_idx = BOB.find_nearest1(hp_2_h22, max(hp_2_h22))
t_2_h22 = t_2_h22 - t_2_h22[t_shift_2_idx]

# Omega and Omega_dot from teobResum

h_tot = hp_2_h22 + 1j*hc_2_h22

amp = abs(h_tot)
phase = -np.unwrap(np.angle(h_tot))

omega = np.gradient(phase,t_2_h22)
omega_dot = np.gradient(omega,t_2_h22)

t_ini_teob = -25.0 # time at which initial conditions are given

t_ref_idx = BOB.find_nearest1(t_2_h22, t_ini_teob)
t0 = t_2_h22[t_ref_idx]
OMEGA_ref = omega[t_ref_idx]/2.0; OMEGA_dot_ref = omega_dot[t_ref_idx]/2.0; phase0 = phase[t_ref_idx]
tp = BOB.t_peak(t0, OMEGA_ref, OMEGA_dot_ref, OM_QNM, tau)
A0 = amp[t_ref_idx]*4.0*OMEGA_ref*OMEGA_ref
Ap = BOB.A_peak(A0, t0, tp, tau)

phase_BOB = BOB.Phi_BOB(t_2_h22, tp, OMEGA_ref, OMEGA_dot_ref, OM_QNM, tau)
amp_bob = BOB.h_amp_BOB(t_2_h22, tp, Ap, OMEGA_ref, OMEGA_dot_ref, OM_QNM, tau)


hp_BOB =  amp_bob*np.cos(2*phase_BOB)
t_shift_BOB_idx = BOB.find_nearest1(hp_BOB, max(hp_BOB))
t_BOB = t_2_h22
t_BOB = t_BOB-t_BOB[t_shift_BOB_idx]

# Generate a complete waveform BOB+TEOB
BOB_idx_cut = 300

time_teob = t_2_h22[:t_ref_idx]
h22_teob = hp_2_h22[:t_ref_idx]
time_bob = t_BOB[t_ref_idx-BOB_idx_cut:]
h22_bob = hp_BOB[t_ref_idx-BOB_idx_cut:]

#plt.plot(time_bob, h22_bob)
#plt.plot(w_2_2_L3.real.t-time_l3_max, 3.9*w_2_2_L3.real.data, 'k--')
#plt.plot(t_2_h22, hp_2_h22, 'c--')
#plt.xlim(-200.0, 80.0)
plt.show()




# Load Energy Evolution Teobcode
t1, r, phi, Pph, MOmega, ddor, Prstar, MOmega_orb, E = np.loadtxt('/home/ashok/constraintongwwaveform/script/output/dyn_interp.txt', unpack=True)

'''
p = Psi2_SXS(np.pi/2, 0, 1, w_L3)
plt.plot(w_L3.t, p)
plt.xlabel('time')
plt.ylabel('Psi2')
plt.savefig("../plots/psi2_SXS.pdf")
plt.show()
'''

'''
p = Int_h_dot_SXS(np.pi/2, 0)
plt.plot(w_L3.t, p)
plt.xlabel('time')
plt.ylabel('Psi2')
plt.show()

'''

'''
p = Balance_law(np.pi/2, 0., 1,  w_L3)
plt.plot(w_L3.t, p)
plt.xlabel('time')
plt.ylabel('Psi2')
plt.savefig("../plots/Balance_SXS.pdf")
plt.show()
'''

#'''
p = M_SXS(1.0,w_L3)
p=p-p[0]

plt.plot(w_L3.t-4700, p+0.001)
plt.xlabel('time')
plt.ylabel('Psi2')
#plt.savefig("../plots/Balance_SXS.pdf")
#plt.show()
#'''


p = M_teob(1.0, 5)[1]
t = M_teob(1.0, 5)[0]

p = p[200:]
p=p-p[0]
t = t[200:]

E = E[200:]
E=E-E[0]
t1 = t1[200:]

plt.plot(t, p)
plt.plot(t1,  E, 'k--')

plt.xlabel('time')
plt.ylabel('Psi2')
#plt.savefig("../plots/Balance_SXS.pdf")
plt.show()

p = Vx_teob(1.0, 2)[1]
t = Vx_teob(1.0, 2)[0]

plt.plot(t, p)

plt.xlabel('time')
plt.ylabel('Vx')
#plt.savefig("../plots/Balance_SXS.pdf")
plt.show()


