import sys
sys.path.insert(0, '../src')
import BOB_functions as BOB
import matplotlib.pyplot as plt
import numpy as np
import sxs
from scipy.interpolate import interp1d

extrapolation_order = 4
w_L3 = sxs.load("SXS:BBH:2085/Lev(3)/rhOverM", extrapolation_order=extrapolation_order)
w_L4 = sxs.load("SXS:BBH:2085/Lev(4)/rhOverM", extrapolation_order=extrapolation_order)

# Remove junk radiation
metadata_L3 = sxs.load("SXS:BBH:2085/Lev(3)/metadata.json")
metadata_L3.reference_time
metadata_L4 = sxs.load("SXS:BBH:2085/Lev(4)/metadata.json")
metadata_L4.reference_time

reference_index_L3 = w_L3.index_closest_to(metadata_L3.reference_time)
w_sliced_L3 = w_L3[reference_index_L3:]
reference_index_L4 = w_L4.index_closest_to(metadata_L4.reference_time)
w_sliced_L4 = w_L4[reference_index_L4:]

#Plot h22 waveform at differnet resolution

w_2_2_L4 = w_sliced_L4[:, w_L4.index(2, 2)]
w_2_2_L3 = w_sliced_L3[:, w_L3.index(2, 2)]

print(len(w_2_2_L4.real.t), len(w_2_2_L3.real.t))
plt.plot(w_2_2_L4.real.t, w_2_2_L4.real.data)
plt.plot(w_2_2_L3.real.t, w_2_2_L3.real.data, 'r--')

plt.title(f"Sliced extrapolated Waveform, $N={extrapolation_order}$")
plt.xlabel(r"$(t_{\mathrm{corr}} - r_\ast)/M$")
plt.ylabel(r"$r\, h^{(\ell,m)}/M$")
plt.show()

# NR Amplitude and Phasing error

plt.plot(w_sliced_L4.t, w_sliced_L4[:, w_sliced_L4.index(2, 2)].abs)
plt.plot(w_sliced_L3.t, w_sliced_L3[:, w_sliced_L3.index(2, 2)].abs, 'r--')
plt.title(f"Extrapolated Waveform, $N={extrapolation_order}$")
plt.xlabel(r"$(t_{\mathrm{corr}} - r_\ast)/M$")
plt.ylabel(r"$\left| r\, h^{(2,2)}/M \right|$")
plt.show()

Phase_L4 = -w_sliced_L4.arg_unwrapped[:, w_sliced_L4.index(2, 2)]
Phase_L3 = -w_sliced_L3.arg_unwrapped[:, w_sliced_L3.index(2, 2)]

plt.plot(w_sliced_L4.t, Phase_L4, label='arg unwrapped')
plt.xlabel(r"$(t_{\mathrm{corr}} - r_\ast)/M$")
plt.ylabel(rf"$\mathrm{{arg}} \left[ h^{{{2}, {2}}} \right]$")
plt.legend()
plt.show()

# Load data from teobResumm code 
t_1_h22, hp_1_h22, hc_1_h22 = np.loadtxt('/home/ashok/teobresums/C/data/hlm_ringdown_l2_m2_reim.txt', unpack=True)
t_2_h22, h22_amp, h22_phase = np.loadtxt('/home/ashok/teobresums/C/data/hlm_ringdown_l2_m2.txt', unpack=True)

#t_1_h22, hp_1_h22, hc_1_h22 = np.loadtxt('/home/ashok/teobresumsNoNRtn/teobresums/C/data/hlm_ringdown_l2_m2_reim.txt', unpack=True)
#t_2_h22, h22_amp, h22_phase = np.loadtxt('/home/ashok/teobresumsNoNRtn/teobresums/C/data/hlm_ringdown_l2_m2.txt', unpack=True)

phase_teob = -np.unwrap(np.angle(hp_1_h22 + 1j*hc_1_h22))

plt.plot(w_2_2_L3.real.t-w_2_2_L3.real.t[0], w_2_2_L3.real.data, 'r--')
plt.plot(t_1_h22-t_1_h22[0]+31, hp_1_h22*(max(w_2_2_L3.real.data)/max(hp_1_h22)), label=r'teobRsum NR tuned $h_{22}$')
#plt.xlim(-100, 75)
#plt.ylim(-2.0, 2.0)
plt.xlabel('time')
plt.ylabel(r'$h_{22}$')
plt.show()

# Phase difference between teobResumm and SXS waveform
Phase_errr = Phase_L3-Phase_L4.interpolate(w_sliced_L3.t)
time_h22 = t_1_h22-t_1_h22[0]+31
time_SXS_L4 = w_sliced_L4.t-w_2_2_L3.real.t[0]
time_SXS_L3 = w_sliced_L3.t-w_2_2_L3.real.t[0]


Phase_L4 = Phase_L4-Phase_L4[0]
h22_phase = phase_teob- phase_teob[0]

plt.plot(time_h22, h22_phase)
plt.plot(time_SXS_L4, Phase_L4, 'r--')
plt.show()

idx_sort=30
time_SXS_L4 =time_SXS_L4[idx_sort:]
Phase_L4 = Phase_L4[idx_sort:]

h22_phase_intrp=interp1d(time_h22, h22_phase, kind='cubic')
h22_phase_intrp=h22_phase_intrp(time_SXS_L4)
h22_phase_intrp-h22_phase_intrp-h22_phase_intrp[0]

phase_diff_NR_teob = h22_phase_intrp-Phase_L4
phase_diff_NR_teob=phase_diff_NR_teob-phase_diff_NR_teob[0]

plt.plot(time_SXS_L4, phase_diff_NR_teob, 'k')
plt.fill_between(time_SXS_L3, -abs(Phase_errr), abs(Phase_errr), alpha=0.2)
plt.show()

# BOB waveform
# Final state variable
af = 0.6864427
Mf = 0.9517857
M_tot = 1
t_scale = M_tot*4.92549094830932e-06 

# quasinormal, tau frequency
om_qnm = 1.5251-1.1568*(1-af)**0.1292
om_qnm = om_qnm/Mf
Q = 0.7+1.4187*(1-af)**(-0.499)
OM_QNM = om_qnm/2
tau = 2*Q/om_qnm

# Initial condition for BOB using various waveforms
t_1_h22, hp_1_h22, hc_1_h22 = np.loadtxt('/home/ashok/teobresumsNoNRtn/teobresums/C/bbh_q1_noSpin_NRtnd_a5_a6_HrznFlx_auto_nqcOutput/hlm_ringdown_l2_m2_reim.txt', unpack=True)
t_2_h22, hp_2_h22, hc_2_h22 = np.loadtxt('/home/ashok/teobresumsNoNRtn/teobresums/C/bbh_q1_noSpin_No_a5_a6_No_HrznFlx_No_nqcOutput/hlm_ringdown_l2_m2_reim.txt', unpack=True)

t_shift_1_idx = BOB.find_nearest1(hp_1_h22, max(hp_1_h22))
t_1_h22 = t_1_h22 - t_1_h22[t_shift_1_idx]

t_shift_2_idx = BOB.find_nearest1(hp_2_h22, max(hp_2_h22))
t_2_h22 = t_2_h22 - t_2_h22[t_shift_2_idx]

# Omega and Omega_dot from teobResum

h_tot = hp_2_h22 + 1j*hc_2_h22
amp = abs(h_tot)
phase = -np.unwrap(np.angle(h_tot))

omega = np.gradient(phase,t_2_h22)
omega_dot = np.gradient(omega,t_2_h22)

t_ref_idx = BOB.find_nearest1(t_2_h22, -1.5)
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

plt.plot(t_1_h22, hp_1_h22, label=r'teobRsum NR tuned $h_{22}$')
plt.plot(t_2_h22, hp_2_h22, 'k', label=r'$h_{22}$ with $a_{5}=0$, $a_{6}=0$, $F_{H}=0$ and no NCQ')
plt.plot(t_BOB+1.6, hp_BOB, 'r--', label=r'$h_{22}$ BOB')
plt.xlim(-100, 75)
plt.ylim(-2.0, 2.0)
plt.xlabel('time')
plt.ylabel(r'$h_{22}$')

plt.legend()
plt.show()
