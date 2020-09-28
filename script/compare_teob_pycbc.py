import numpy as np 
import matplotlib.pyplot as plt
from pycbc import waveform
from pycbc.waveform import get_td_waveform
from scipy.integrate import odeint
from scipy import integrate

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
hp, hc = waveform.get_td_waveform(approximant="SEOBNRv4", mass1=m1, mass2=m2, delta_t=1.0/(2*4096), f_lower=30)

hp, hc = hp.trim_zeros(), hc.trim_zeros()
amp = waveform.utils.amplitude_from_polarizations(hp, hc)
phase = waveform.utils.phase_from_polarizations(hp, hc)
t = hp.sample_times

t_shift_idx=find_nearest1(hp, max(hp))
t=t-t[t_shift_idx]


#t_teob, hp_teob, hc_teob, amp_teob, phase_teob = np.loadtxt('/home/ashok/teobresums/C/compare_pycbc/waveform.txt', unpack=True)
t_teob, hp_teob, hc_teob, amp_teob, phase_teob = np.loadtxt('/home/ashok/teobresums/C/bbh_q1_s0s0_M46_30Hz_postadiab/waveform.txt', unpack=True)


h_tot_teob = hp_teob + 1j*hc_teob
amp_teob = abs(h_tot_teob)
phase_teob = np.unwrap(np.angle(h_tot_teob))

t_shift_idx=find_nearest1(hp_teob, max(hp_teob))
t_teob=t_teob-t_teob[t_shift_idx]

plt.plot(t, hp)
plt.plot(t_teob, hp_teob, 'r--')
plt.show()

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
OM_QNM = om_qnm/2/t_scale
tau = 2*Q/om_qnm*t_scale

# Omega and Omega_dot from SEOBNRv4
omega_seobnrv4 = np.gradient(phase,t)
omega_dot_seobnrv4 = np.gradient(omega_seobnrv4,t)

t_ref_idx = find_nearest1(t, -5.0*t_scale)
t0 = t[t_ref_idx]
OMEGA_ref = omega_seobnrv4[t_ref_idx]/2.0; OMEGA_dot_ref = omega_dot_seobnrv4[t_ref_idx]/2.0; phase0 = phase[t_ref_idx]
tp = t_peak(t0, OMEGA_ref, OMEGA_dot_ref, OM_QNM, tau)
A0 = amp[t_ref_idx]*4.0*OMEGA_ref*OMEGA_ref
Ap = A_peak(A0, t0, tp, tau)

hp_BOB =  Waveform_BOB(t, tp, Ap, OMEGA_ref, OMEGA_dot_ref, OM_QNM, tau, phase0)
t_shift_BOB_idx = find_nearest1(hp_BOB, max(hp_BOB))
t_BOB = t-t[t_shift_BOB_idx]

plt.plot(t, hp)
plt.plot(t_BOB, hp_BOB, 'r--')
plt.show()

# Omega and Omega_dot from SEOBNRv4
omega_seobnrv4 = np.gradient(phase_teob,t_teob)
omega_dot_seobnrv4 = np.gradient(omega_seobnrv4,t_teob)

t_ref_idx = find_nearest1(t_teob, -5.0*t_scale)
t0 = t_teob[t_ref_idx]
OMEGA_ref = omega_seobnrv4[t_ref_idx]/2.0; OMEGA_dot_ref = omega_dot_seobnrv4[t_ref_idx]/2.0; phase0 = phase[t_ref_idx]
tp = t_peak(t0, OMEGA_ref, OMEGA_dot_ref, OM_QNM, tau)
A0 = amp_teob[t_ref_idx]*4.0*OMEGA_ref*OMEGA_ref
Ap = A_peak(A0, t0, tp, tau)

hp_BOB =  Waveform_BOB(t_teob, tp, Ap, OMEGA_ref, OMEGA_dot_ref, OM_QNM, tau, phase0)
t_shift_BOB_idx = find_nearest1(hp_BOB, max(hp_BOB))
t_BOB = t_teob-t_teob[t_shift_BOB_idx]

plt.plot(t, hp)
plt.plot(t_BOB, hp_BOB, 'r--')
plt.show()

# Super momentum Balance law
## Apply supermomentum balace law
GM = 1.32712440018e20
c = 299792458

def Sigma_dot_sqr_BOB(y, t):
    return abs(np.sqrt(5.0/np.pi)/8.0*h_amp_BOB(t, tp, Ap, OMEGA_ref, OMEGA_dot_ref, OM_QNM, tau)*2*Omega_BOB(t, tp, OMEGA_ref, OMEGA_dot_ref, OM_QNM, tau))**2

x_in_idx = find_nearest1(t_BOB, -5*t_scale)
x_fin_idx = find_nearest1(t_BOB, 0)

x = t_BOB[x_in_idx:x_fin_idx]
ys = odeint(Sigma_dot_sqr_BOB, 0, x)
ys = np.array(ys).flatten()
OM_BOB = Omega_BOB(x, tp, OMEGA_ref, OMEGA_dot_ref, OM_QNM, tau)
OM_BOB = OM_BOB - OM_BOB[0]

x_bob = x


plt.plot(x, ys, 'c', label='int_sigmaDot')
#plt.plot(x, 2.0*OM_BOB*GM/c**2, 'b', label='Omega')
plt.legend()
plt.show()

# Loda data for EOB dynamics
EOB_dynamic_filepath = '/home/ashok/teobresums/C/bbh_q1_s0s0_M46_30Hz_postadiab/dyn.txt'
EOB_h22_filepath = '/home/ashok/teobresums/C/bbh_q1_s0s0_M46_30Hz_postadiab/hlm_ringdown_l2_m2_reim.txt'
EOB_h22_amp_filepath = '/home/ashok/teobresums/C/bbh_q1_s0s0_M46_30Hz_postadiab/hlm_ringdown_l2_m2.txt'

t1, r, phi, Pph, MOmega, ddor, Prstar, MOmega_orb, E = np.loadtxt(EOB_dynamic_filepath, unpack=True)
t_h22, h22_re, h22_Im = np.loadtxt(EOB_h22_filepath, unpack=True)
t_h22_amp, h22_amp, h22_phase = np.loadtxt(EOB_h22_amp_filepath, unpack=True)

amp = abs(h22_re + 1j*h22_Im)
phase = -np.unwrap(np.angle(h22_re + 1j*h22_Im))
omega = np.gradient(phase, t_h22)
omega_dot = np.gradient(omega, t_h22)


t_shift_eob_idx = find_nearest1(h22_re, max(h22_re))
t_h22 = t_h22 - t_h22[t_shift_eob_idx]


#Input parameter for BOB h22

OM_QNM = om_qnm/2
tau = 2*Q/om_qnm

t_ref_idx = find_nearest1(t_h22, -1.5)
t0 = t_h22[t_ref_idx]
OMEGA_ref = omega[t_ref_idx]/2.0
OMEGA_dot_ref = omega_dot[t_ref_idx]/2.0; phase0 = phase[t_ref_idx]
tp = t_peak(t0, OMEGA_ref, OMEGA_dot_ref, OM_QNM, tau)
A0 = amp[t_ref_idx]*4.0*OMEGA_ref*OMEGA_ref
Ap = A_peak(A0, t0, tp, tau)

phase_BOB = Phi_BOB(t_h22, tp, OMEGA_ref, OMEGA_dot_ref, OM_QNM, tau)
amp_bob = h_amp_BOB(t_h22, tp, Ap, OMEGA_ref, OMEGA_dot_ref, OM_QNM, tau)

hp_BOB =  amp_bob*np.cos(2*phase_BOB)
t_shift_BOB_idx = find_nearest1(hp_BOB, max(hp_BOB))
t_BOB = t_h22


plt.plot(t_h22, h22_re)
plt.plot(t_BOB+1.5, hp_BOB, '--',label='BOB')
#plt.xlim(-60, 80)
plt.show()


# Plot left hand side of super momentum balance equation
def Sigma_dot_sqr_BOB(y, t):
    return abs(np.sqrt(5.0/np.pi)/8.0*h_amp_BOB(t, tp, Ap, OMEGA_ref, OMEGA_dot_ref, OM_QNM, tau)*2*Omega_BOB(t, tp, OMEGA_ref, OMEGA_dot_ref, OM_QNM, tau))**2

t_ini = -50

x_in_idx = find_nearest1(t_BOB, t_ini)
x_fin_idx = find_nearest1(t_BOB, 0.2)

x = t_BOB[x_in_idx:x_fin_idx]

x_rvs = -np.flip(x)

ys = odeint(Sigma_dot_sqr_BOB, 0, x_rvs)
ys = np.array(ys).flatten()
OM_BOB = Omega_BOB(x, tp, OMEGA_ref, OMEGA_dot_ref, OM_QNM, tau)
OM_BOB = OM_BOB - OM_BOB[0]

t1=t1-t1[-1]
x_eob_in_idx = find_nearest1(t1, t_ini)
x_eob_fin_idx = find_nearest1(t1, 0.1)
x_eob_rvs = -np.flip(t1[x_eob_in_idx:x_eob_fin_idx])
E_rvs = np.flip(E[x_eob_in_idx:x_eob_fin_idx])
Prstar_rvs = np.flip(Prstar[x_eob_in_idx:x_eob_fin_idx])

E_rvs = E_rvs - E_rvs[0]
Prstar_rvs = Prstar_rvs - Prstar_rvs[0]

plt.plot(x_eob_rvs, 4.0*np.sqrt(2.0)*E_rvs, 'r')
plt.plot(x_eob_rvs, 4.0*np.sqrt(2.0)*E_rvs + 0.25*Prstar_rvs, 'r--')
plt.plot(x_rvs, ys, 'c', label='int_sigmaDot')
plt.legend()
#plt.xlim(-40,0)
plt.show()
