import numpy as np 
import matplotlib.pyplot as plt
from pycbc import waveform
from pycbc.waveform import get_td_waveform
from scipy.integrate import odeint
from scipy import integrate

def find_nearest1(array,value):
    idx,val = min(enumerate(array), key=lambda x: abs(x[1]-value))
    return idx

def find_nearest1(array,value):
    idx,val = min(enumerate(array), key=lambda x: abs(x[1]-value))
    return idx

def OMEGA_p_pow4(OM0, OM0_dot, OMQNM, tau):
    OMQNMp4 = OMQNM**4; OM0p4 = OM0**4; OM0p3 = OM0**3
    tanh_arg = -np.log(((OMQNMp4 - OM0p4)/(2.0*tau* OM0p3 *OM0_dot))-1.0)/2.0
    OM_p_pow4 = (OM0p4-OMQNMp4*np.tanh(tanh_arg))/(1.0-np.tanh(tanh_arg))
    return OM_p_pow4

def OMEGA_m_pow4(OM0, OM0_dot, OMQNM, tau):
    OMQNMp4 = OMQNM**4; OM0p4 = OM0**4; OM0p3 = OM0**3
    tanh_arg = -np.log(((OMQNMp4 - OM0p4)/(2.0*tau* OM0p3 *OM0_dot))-1.0)/2.0
    OM_m_pow4 = (OMQNMp4-OM0p4)/(1.0-np.tanh(tanh_arg))
    return OM_m_pow4

def kapp(OM0, OM0_dot, OMQNM, tau):
    OMQNMp4 = OMQNM**4; OM0p4 = OM0**4; OM0p3 = OM0**3
    tanh_arg = -np.log(((OMQNMp4 - OM0p4)/(2.0*tau*OM0p3*OM0_dot))-1.0)/2.0
    OM_m_pow4 = (OMQNMp4-OM0p4)/(1.0-np.tanh(tanh_arg))
    kp_p = (OM0**4 + OM_m_pow4*(1.0 - np.tanh(tanh_arg)))**0.25
    kp_m = (OM0**4 - OM_m_pow4*(1.0 + np.tanh(tanh_arg)))**0.25
    return kp_p, kp_m

def t_ph(tp, OM0, OM0_dot, OMQNM, tau):
    OM_p_pow4 = OMEGA_p_pow4(OM0, OM0_dot, OMQNM, tau)
    OM_m_pow4 = OMEGA_m_pow4(OM0, OM0_dot, OMQNM, tau)
    tph = tp + (tau/2.0)*np.arctanh((OM_p_pow4/OM_m_pow4)-np.sqrt((OM_p_pow4/OM_m_pow4)**2-1.0))
    return tph

def t_peak(t0, OM0, OM0_dot, OMQNM, tau):
    OMQNMp4 = OMQNM**4; OM0p4 = OM0**4; OM0p3 = OM0**3
    tanh_arg = -np.log(((OMQNMp4 - OM0p4)/(2.0*tau* OM0p3 *OM0_dot))-1.0)/2.0
    tp = t0 - tau*tanh_arg
    return tp

def Omega_BOB(t, tp, OM0, OM0_dot, OMQNM, tau):
    OM_p_pow4 = OMEGA_p_pow4(OM0, OM0_dot, OMQNM, tau)
    OM_m_pow4 = OMEGA_m_pow4(OM0, OM0_dot, OMQNM, tau)
    OM = (OM_p_pow4 + OM_m_pow4*np.tanh((t-tp)/tau))**0.25
    return OM

def Phi_BOB(t, tp, OM0, OM0_dot, OMQNM, tau):
    OM_BOB = Omega_BOB(t, tp, OM0, OM0_dot, OMQNM, tau)
    kp = kapp(OM0, OM0_dot, OMQNM, tau)[0]
    km = kapp(OM0, OM0_dot, OMQNM, tau)[1]
    arctan_p = kp*tau*(np.arctan2(OM_BOB,kp)-np.arctan2(OM0,kp))
    arctan_m = km*tau*(np.arctan2(OM_BOB,km)-np.arctan2(OM0,km))
    arctanh_p = kp*tau*(np.arctanh(OM_BOB/kp)-np.arctanh(OM0/kp))
    arctanh_m = km*tau*0.5*np.log( (OM_BOB/km + 1)*(1 - OM0/km)/(1 - OM_BOB/km)/(1 + OM0/km))
    return arctan_p + arctanh_p - arctan_m - arctanh_m

def A_peak(A0, t0, tp, tau):
    return A0*np.cosh((t0-tp)/tau)

def Psi4_BOB_amp(t, Ap, tau, tp):
    return Ap/np.cosh((t-tp)/tau)

def h_amp_BOB(t, tp, Ap, OM0, OM0_dot, OMQNM, tau):
    psi4_BOB = Psi4_BOB_amp(t, Ap, tau, tp)
    OM_BOB = Omega_BOB(t, tp, OM0, OM0_dot, OMQNM, tau)
    return psi4_BOB/(4.0*OM_BOB**2)

def Waveform_BOB(t, tp, Ap, OM0, OM0_dot, OMQNM, tau, phase0):
    amp = h_amp_BOB(t, tp, Ap, OM0, OM0_dot, OMQNM, tau)
    phase = Phi_BOB(t, tp, OM0, OM0_dot, OMQNM, tau)- Phi_BOB(t, tp, OM0, OM0_dot, OMQNM, tau)[0] +  phase0
    return amp*np.exp(2j*phase)


# Initial condition using pycbc waveform
m1=23.0; m2=23.0
hp, hc = waveform.get_td_waveform(approximant="SEOBNRv4", mass1=m1, mass2=m2, delta_t=1.0/(8*4096), f_lower=30)

hp, hc = hp.trim_zeros(), hc.trim_zeros()
amp = waveform.utils.amplitude_from_polarizations(hp, hc)
phase = waveform.utils.phase_from_polarizations(hp, hc)
t = hp.sample_times

t_shift_idx=find_nearest1(hp, max(hp))
t=t-t[t_shift_idx]


t_1, hp_1, hc_1, amp_1, phase_1 = np.loadtxt('/home/ashok/teobresumsNoNRtn/teobresums/C/bbh_q1_noSpin_NRtnd_a5_a6_HrznFlx_auto_nqcOutput/waveform.txt', unpack=True)
t_2, hp_2, hc_2, amp_2, phase_2 = np.loadtxt('/home/ashok/teobresumsNoNRtn/teobresums/C/bbh_q1_noSpin_No_a5_a6_No_HrznFlx_No_nqcOutput/waveform.txt', unpack=True)

## Shifting time axix such that all peak are at t=0

t_shift_idx_t1=find_nearest1(hp_1, max(hp_1))
t_1=t_1-t_1[t_shift_idx_t1]
t_shift_idx_t2=find_nearest1(hp_2, max(hp_2))
t_2=t_2-t_2[t_shift_idx_t2]

###########################

plt.plot(t, hp, 'k', label='pycbc wf')
plt.plot(t_1, hp_1, 'r--', label='teobRsum NR tuned')
plt.plot(t_2, hp_2, 'g', label=r'$a_{5}=0$, $a_{6}=0$, $F_{H}=0$ and no NCQ')
plt.xlim(-0.025, 0.015)
plt.xlabel('time')
plt.ylabel(r'$h_{+}$')

plt.legend()
plt.savefig('/home/ashok/constraintongwwaveform/plots/hp_pycbc_teob_comparision.pdf')
plt.show()

# Initial condition for BOB using various waveforms
t_1_h22, hp_1_h22, hc_1_h22 = np.loadtxt('/home/ashok/teobresumsNoNRtn/teobresums/C/bbh_q1_noSpin_NRtnd_a5_a6_HrznFlx_auto_nqcOutput/hlm_ringdown_l2_m2_reim.txt', unpack=True)
t_2_h22, hp_2_h22, hc_2_h22 = np.loadtxt('/home/ashok/teobresumsNoNRtn/teobresums/C/bbh_q1_noSpin_No_a5_a6_No_HrznFlx_No_nqcOutput/hlm_ringdown_l2_m2_reim.txt', unpack=True)

t_shift_1_idx = find_nearest1(hp_1_h22, max(hp_1_h22))
t_1_h22 = t_1_h22 - t_1_h22[t_shift_1_idx]

t_shift_2_idx = find_nearest1(hp_2_h22, max(hp_2_h22))
t_2_h22 = t_2_h22 - t_2_h22[t_shift_2_idx]


## Generate BOB waveform
# Final state variable
af = 0.6864427
Mf = 0.9517857
M_tot = m1+m2
t_scale = M_tot*4.92549094830932e-06 

# quasinormal, tau frequency
om_qnm = 1.5251-1.1568*(1-af)**0.1292
om_qnm = om_qnm/Mf
Q = 0.7+1.4187*(1-af)**(-0.499)
OM_QNM = om_qnm/2
tau = 2*Q/om_qnm

# Omega and Omega_dot from teobResum

h_tot = hp_2_h22 + 1j*hc_2_h22
amp = abs(h_tot)
phase = -np.unwrap(np.angle(h_tot))

omega = np.gradient(phase,t_2_h22)
omega_dot = np.gradient(omega,t_2_h22)

t_ref_idx = find_nearest1(t_2_h22, -1.5)
t0 = t_2_h22[t_ref_idx]
OMEGA_ref = omega[t_ref_idx]/2.0; OMEGA_dot_ref = omega_dot[t_ref_idx]/2.0; phase0 = phase[t_ref_idx]
tp = t_peak(t0, OMEGA_ref, OMEGA_dot_ref, OM_QNM, tau)
A0 = amp[t_ref_idx]*4.0*OMEGA_ref*OMEGA_ref
Ap = A_peak(A0, t0, tp, tau)

phase_BOB = Phi_BOB(t_2_h22, tp, OMEGA_ref, OMEGA_dot_ref, OM_QNM, tau)
amp_bob = h_amp_BOB(t_2_h22, tp, Ap, OMEGA_ref, OMEGA_dot_ref, OM_QNM, tau)


hp_BOB =  amp_bob*np.cos(2*phase_BOB)
t_shift_BOB_idx = find_nearest1(hp_BOB, max(hp_BOB))
t_BOB = t_2_h22

plt.plot(t_1_h22, hp_1_h22, label=r'teobRsum NR tuned $h_{22}$')
plt.plot(t_2_h22, hp_2_h22, 'k', label=r'$h_{22}$ with $a_{5}=0$, $a_{6}=0$, $F_{H}=0$ and no NCQ')
plt.plot(t_BOB+1.6, hp_BOB, 'r--', label=r'$h_{22}$ BOB')
plt.xlim(-100, 75)
plt.ylim(-2.0, 2.0)
plt.xlabel('time')
plt.ylabel(r'$h_{22}$')

plt.legend()
plt.savefig('/home/ashok/constraintongwwaveform/plots/h22_pycbc_teob_comparision.pdf')

plt.show()

# Balance Law
# Plot left hand side of super momentum balance equation
def Sigma_dot_sqr_BOB(y, t):
    return abs(np.sqrt(5.0/np.pi)/8.0*h_amp_BOB(t, tp, Ap, OMEGA_ref, OMEGA_dot_ref, OM_QNM, tau)*2*Omega_BOB(t, tp, OMEGA_ref, OMEGA_dot_ref, OM_QNM, tau))**2

def P_r_BOB(x_rvs, af):
    E = Omega_BOB(x_rvs, tp, OMEGA_ref, OMEGA_dot_ref, OM_QNM, tau)
    Sqrt_R = np.sqrt((E*(9.0+af**2) - 2*af)**2 - (3+af**2)*(2-af*E)**2)
    Delta = 3.0+af**2
    return Sqrt_R/Delta

t_ini = -20

x_in_idx = find_nearest1(t_BOB, t_ini)
x_fin_idx = find_nearest1(t_BOB, 50)

x = t_BOB[x_in_idx:x_fin_idx]

x_rvs = -np.flip(x)

ys = odeint(Sigma_dot_sqr_BOB, 0, x_rvs)
ys = np.array(ys).flatten()
OM_BOB = Omega_BOB(x_rvs, tp, OMEGA_ref, OMEGA_dot_ref, OM_QNM, tau)
OM_BOB = OM_BOB - OM_BOB[0]

Pr_BOB = P_r_BOB(x_rvs, af)
print(Pr_BOB)
Pr_BOB = Pr_BOB - Pr_BOB[0]

plt.plot(x_rvs, ys, 'c', label=r'$\Psi = -\int_{50}^{-u_2}du|\dot{\sigma^{0}}_{BOB}|^2$')
plt.plot(x_rvs, OM_BOB, 'r',  label=r'$E_{BOB}$')
plt.xlabel('time')
plt.ylabel(r'$\Psi$')
plt.legend()

plt.savefig('/home/ashok/constraintongwwaveform/plots/BOB_balanceLaw_with_teobinitial_Cndtn.pdf')

plt.show()
