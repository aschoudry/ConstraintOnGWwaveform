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


# Final state variable
af = 0.6864427
Mf = 0.9517857
m1 = 23; m2 = 23
M_tot = m1+m2
t_scale = M_tot*4.92549094830932e-06 

# quasinormal, tau frequency
om_qnm = 1.5251-1.1568*(1-af)**0.1292
om_qnm = om_qnm/Mf
Q = 0.7+1.4187*(1-af)**(-0.499)
OM_QNM = om_qnm/2
tau = 2*Q/om_qnm

# generate data from pycbc

# Initial condition using pycbc waveform
hp, hc = waveform.get_td_waveform(approximant="SEOBNRv4", mass1=m1, mass2=m2, delta_t=1.0/(8*4096), f_lower=15)

hp, hc = hp.trim_zeros(), hc.trim_zeros()
amp = waveform.utils.amplitude_from_polarizations(hp, hc)
phase = waveform.utils.phase_from_polarizations(hp, hc)
t = hp.sample_times/(t_scale)

# omega and omega_dot
omega = np.gradient(phase, t)
omega_dot = np.gradient(omega, t)

t_ref_idx = find_nearest1(t, -5.0)

t0 = t[t_ref_idx]
OMEGA_ref = omega[t_ref_idx]/2.0; OMEGA_dot_ref = omega_dot[t_ref_idx]/2.0; phase0 = phase[t_ref_idx+7]
tp = t_peak(t0, OMEGA_ref, OMEGA_dot_ref, OM_QNM, tau)
A0 = amp[t_ref_idx]*4.0*OMEGA_ref*OMEGA_ref
Ap = A_peak(A0, t0, tp, tau)

OM_BOB = Omega_BOB(t, tp, OMEGA_ref, OMEGA_dot_ref, OM_QNM, tau)
phase_BOB = Phi_BOB(t, tp, OMEGA_ref, OMEGA_dot_ref, OM_QNM, tau)
amp_bob = h_amp_BOB(t, tp, Ap, OMEGA_ref, OMEGA_dot_ref, OM_QNM, tau)

idx_pre = 300

plt.plot(t[t_ref_idx-idx_pre:], Waveform_BOB(t, tp, Ap, OMEGA_ref, OMEGA_dot_ref, OM_QNM, tau, phase0)[t_ref_idx-idx_pre:], label='BOB')
plt.plot(t, hp, 'r--', label='SEOBNRv4')
plt.legend()
plt.xlim(-60, 80)
plt.show()

# Load data from teobResumm output
waveform_filepath_hp_hc = '/home/ashok/teobresums/Python/outdir/waveform.txt'
waveform_filepath_h22 = '/home/ashok/teobresums/Python/outdir/hlm_ringdown_l2_m2_reim.txt'

## Load data from teobResum output
t_teob, hp_teob, hc_teob, amp_teob, phase_teob  = np.loadtxt(waveform_filepath_hp_hc, unpack=True)
t, hp, hc  = np.loadtxt(waveform_filepath_h22, unpack=True)

t_shift_idx_teob = find_nearest1(hp_teob, max(hp_teob))
t_shift_idx = find_nearest1(hp, max(hp))

t = t - t[t_shift_idx]
t_teob = t_teob - t_teob[t_shift_idx_teob]


h_tot = (hp + 1j* hc)

phase=np.unwrap(np.angle(h_tot))
amp = abs(h_tot)


# omega and omega_dot
omega = np.gradient(phase, t)
omega_dot = np.gradient(omega, t)


t_ref_idx = find_nearest1(t, -5.0)

t0 = t[t_ref_idx]
OMEGA_ref = omega[t_ref_idx]/2.0; OMEGA_dot_ref = omega_dot[t_ref_idx]/2.0; phase0 = phase[t_ref_idx+7]
tp = t_peak(t0, OMEGA_ref, OMEGA_dot_ref, OM_QNM, tau)
A0 = amp[t_ref_idx]*4.0*OMEGA_ref*OMEGA_ref
Ap = A_peak(A0, t0, tp, tau)

OM_BOB = Omega_BOB(t, tp, OMEGA_ref, OMEGA_dot_ref, OM_QNM, tau)
phase_BOB = Phi_BOB(t, tp, OMEGA_ref, OMEGA_dot_ref, OM_QNM, tau)
amp_bob = h_amp_BOB(t, tp, Ap, OMEGA_ref, OMEGA_dot_ref, OM_QNM, tau)


plt.plot(t, amp_bob*np.cos(2*phase_BOB), label='BOB')
plt.plot(t+2.5, hp, 'r--', label='teobRessum')
plt.xlim(-60, 80)
plt.legend()
plt.show()


## Apply supermomentum balace law
def Sigma_dot_sqr_BOB(y, t):
    return abs(np.sqrt(5.0/np.pi)/8.0*h_amp_BOB(t, tp, Ap, OMEGA_ref, OMEGA_dot_ref, OM_QNM, tau)*2*Omega_BOB(t, tp, OMEGA_ref, OMEGA_dot_ref, OM_QNM, tau))**2


x_in_idx = find_nearest1(t, -50)
x_fin_idx = find_nearest1(t, 0)

x = t[x_in_idx:x_fin_idx]
ys = odeint(Sigma_dot_sqr_BOB, 0, x)
ys = np.array(ys).flatten()
OM_BOB = Omega_BOB(x, tp, OMEGA_ref, OMEGA_dot_ref, OM_QNM, tau)
OM_BOB = OM_BOB-OM_BOB[-1]
ys = ys-ys[-1]

plt.plot(x, ys, 'c', label='int_sigmaDot')
plt.plot(x, 2*OM_BOB, 'b', label='Omega')
plt.legend()
#plt.xlim(-40, 0)
plt.show()

# Loda data for EOB dynamics
EOB_dynamic_filepath = '/home/ashok/teobresums/Python/outdir/dyn.txt'
t1, r, phi, Pph, MOmega, ddor, Prstar, MOmega_orb, E, = np.loadtxt(EOB_dynamic_filepath, unpack=True)
t2, hp, hc  = np.loadtxt(waveform_filepath_h22, unpack=True)
h_tot = hp + 1j* hc
phase=-np.unwrap(np.angle(h_tot))
amp = abs(h_tot)

t_shift_idx = find_nearest1(hp, max(hp))

t=t2
t = t - t[t_shift_idx]

x_in_idx = find_nearest1(t, -50)
x_fin_idx = find_nearest1(t, 0)

x = t[x_in_idx:x_fin_idx]
E = E[x_in_idx:x_fin_idx]
E=E-E[-1]

plt.plot(x, ys, 'c')
plt.plot(x, 2*OM_BOB, 'b')
plt.plot(x, -E, 'r')
plt.show()


# intergrate sigma dot EOB
ys_EOB = np.array([])


for i in range(2,len(t2)):
    ys = integrate.simps(abs(amp[:i])**2, t2[:i])
    ys_EOB = np.append(ys_EOB, ys)

plt.plot(ys_EOB)
plt.show()


