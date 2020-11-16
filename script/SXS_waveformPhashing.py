import sys
sys.path.insert(0, '../src')
import BOB_functions as BOB
import matplotlib.pyplot as plt
import numpy as np
import sxs
from scipy.interpolate import interp1d

## Some good q=1 nospin BBH file nums (1132, 3, 4), (1155, 2, 3)

file_num = 1132
L1=3; L2=4

extrapolation_order = 4
w_L3 = sxs.load("SXS:BBH:"+str(file_num)+"/Lev("+str(L1)+")/rhOverM", extrapolation_order=extrapolation_order)
w_L4 = sxs.load("SXS:BBH:"+str(file_num)+"/Lev("+str(L2)+")/rhOverM", extrapolation_order=extrapolation_order)

# Remove junk radiation
metadata_L3 = sxs.load("SXS:BBH:"+str(file_num)+"/Lev("+str(L1)+")/metadata.json")
metadata_L3.reference_time
metadata_L4 = sxs.load("SXS:BBH:"+str(file_num)+"/Lev("+str(L2)+")/metadata.json")
metadata_L4.reference_time

shift = 200
reference_index_L3 = w_L3.index_closest_to(metadata_L3.reference_time)
w_sliced_L3 = w_L3[reference_index_L3+shift:]
reference_index_L4 = w_L4.index_closest_to(metadata_L4.reference_time)
w_sliced_L4 = w_L4[reference_index_L4+shift:]

#Plot h22 waveform at differnet resolution

w_2_2_L4 = w_sliced_L4[:, w_L4.index(2, 2)]
w_2_2_L3 = w_sliced_L3[:, w_L3.index(2, 2)]

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

# Instantanous frequency
Inst_frq_L4 = Phase_L4.dot
Inst_frq_ini = Inst_frq_L4.data.tolist()[0]

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
omega = np.gradient(phase_teob,t_1_h22)

t_NR_phase0_idx = BOB.find_nearest1(omega, Inst_frq_ini)-1

#### keep data only after f0 in teobNR data
t_1_h22 = t_1_h22[t_NR_phase0_idx:]
hp_1_h22 = hp_1_h22[t_NR_phase0_idx:]
hc_1_h22 = hc_1_h22[t_NR_phase0_idx:]
phase_teob = phase_teob[t_NR_phase0_idx:]
omega = omega[t_NR_phase0_idx:]

####


plt.plot(t_1_h22-t_1_h22[0], omega)
plt.plot(Inst_frq_L4.t - Inst_frq_L4.t[0], Inst_frq_L4.data, 'r--')
plt.show()

#plt.plot(t_1_h22-t_1_h22[0]+31, hp_1_h22*(max(w_2_2_L3.real.data)/max(hp_1_h22)), label=r'teobRsum NR tuned $h_{22}$')
plt.plot(t_1_h22-t_1_h22[0], hp_1_h22, label=r'teobRsum NR tuned $h_{22}$')
plt.plot(w_2_2_L3.real.t-w_2_2_L3.real.t[0], 3.9*w_2_2_L3.real.data, 'r--')

#plt.xlim(-100, 75)
#plt.ylim(-2.0, 2.0)
plt.xlabel('time')
plt.ylabel(r'$h_{22}$')
plt.show()

# Phase difference between teobResumm and SXS waveform

time_h22 = t_1_h22-t_1_h22[0]
time_SXS_L4 = w_sliced_L4.t-w_sliced_L4.t[0]
time_SXS_L3 = w_sliced_L3.t-w_sliced_L3.t[0]

if len(w_2_2_L4.real.t) > len(w_2_2_L3.real.t):
    Phase_errr = Phase_L3-Phase_L4.interpolate(w_sliced_L3.t)
    time_SXS = time_SXS_L3
else:
    Phase_errr = Phase_L4-Phase_L3.interpolate(w_sliced_L4.t)
    time_SXS = time_SXS_L4

Phase_L4 = Phase_L4-Phase_L4[0]
h22_phase = phase_teob- phase_teob[0]

plt.plot(time_h22, h22_phase)
plt.plot(time_SXS_L4, Phase_L4, 'r--')
plt.show()

idx_strt=300
idx_end=-3000
time_SXS_L4 =time_SXS_L4[idx_strt:idx_end]
Phase_L4 = Phase_L4[idx_strt:idx_end]

h22_phase_intrp=interp1d(time_h22, h22_phase, kind='cubic')
h22_phase_intrp=h22_phase_intrp(time_SXS_L4)
h22_phase_intrp-h22_phase_intrp-h22_phase_intrp[0]

phase_diff_NR_teob = h22_phase_intrp-Phase_L4
phase_diff_NR_teob=phase_diff_NR_teob-phase_diff_NR_teob[0]

plt.plot(time_SXS_L4, phase_diff_NR_teob, 'k')
#plt.axvline(w_L4.max_norm_time(), c="black", ls="dotted")
plt.fill_between(time_SXS, -abs(Phase_errr), abs(Phase_errr), alpha=0.2)
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

t_ref_idx = BOB.find_nearest1(t_2_h22, -1.0)
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
plt.plot(t_2_h22, hp_2_h22, 'k--', label=r'$h_{22}$ with $a_{5}=0$, $a_{6}=0$, $F_{H}=0$ and no NCQ')
plt.plot(t_BOB, hp_BOB, 'r--', label=r'$h_{22}$ BOB')
#plt.xlim(-100, 75)
plt.ylim(-2.0, 2.0)
plt.xlabel('time')
plt.ylabel(r'$h_{22}$')

plt.legend()
plt.show()



## Compare phase difference between all waveforms
SXS_t = w_2_2_L4.real.t
SXS_h22re = 3.9*w_2_2_L4.real.data

idx_NR_lst = -3000
SXS_t =SXS_t[:idx_NR_lst]
SXS_h22re = SXS_h22re[:idx_NR_lst]
SXS_t-=SXS_t[-1]

idx_teob_lst = -230
teob_t= t_2_h22[:idx_teob_lst]
teob_h22re = hp_2_h22[:idx_teob_lst]
teob_t-=teob_t[-1]


plt.plot(SXS_t, SXS_h22re)
plt.plot(teob_t + 17.0,teob_h22re, 'k--')
#plt.xlim(-200,0)
plt.show()


## Memory terms from SXS code

#memory = sxs.waveforms.memory.J_E(w_L4 , start_time=1000.0)

#plt.plot(memory)
#plt.show()
