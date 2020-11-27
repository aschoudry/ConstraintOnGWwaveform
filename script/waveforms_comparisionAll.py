import sys
sys.path.insert(0, '../src')
import BOB_functions as BOB
import matplotlib.pyplot as plt
import numpy as np
import sxs
from scipy.interpolate import interp1d

## Some good q=1 nospin BBH file nums (1132, 3, 4), (1155, 2, 3),(0002, 5,6)

file_num = '0002'
L1=5; L2=6

extrapolation_order = 4
w_L3 = sxs.load("SXS:BBH:"+str(file_num)+"/Lev("+str(L1)+")/rhOverM", extrapolation_order=extrapolation_order)
w_L4 = sxs.load("SXS:BBH:"+str(file_num)+"/Lev("+str(L2)+")/rhOverM", extrapolation_order=extrapolation_order)


# Shift to some reference time
wl3_lst = w_L3[:, w_L3.index(2, 2)].real.data.tolist()
wl4_lst = w_L4[:, w_L4.index(2, 2)].real.data.tolist()

wl3_max_idx =  BOB.find_nearest1(wl3_lst, max(wl3_lst))
wl4_max_idx =  BOB.find_nearest1(wl4_lst, max(wl4_lst))

time_l3_max = w_L3[:, w_L3.index(2, 2)].t[wl3_max_idx]
time_l4_max = w_L4[:, w_L4.index(2, 2)].t[wl4_max_idx]

# time before peak at which plotting should start
ini_cut = 100
end_cut = 60 # remove data from end of the waveform

reference_index_L3_ini = w_L3.index_closest_to(time_l3_max-ini_cut)
reference_index_L3_end = w_L3.index_closest_to(time_l3_max+end_cut)
w_sliced_L3 = w_L3[reference_index_L3_ini:reference_index_L3_end]

reference_index_L4_ini = w_L4.index_closest_to(time_l4_max-ini_cut)
reference_index_L4_end = w_L4.index_closest_to(time_l4_max+end_cut)
w_sliced_L4 = w_L4[reference_index_L4_ini:reference_index_L4_end]

#Plot h22 waveform at differnet resolution

w_2_2_L4 = w_sliced_L4[:, w_sliced_L4.index(2, 2)]
w_2_2_L3 = w_sliced_L3[:, w_sliced_L3.index(2, 2)]

plt.plot(w_2_2_L4.real.t, w_2_2_L4.real.data)
plt.plot(w_2_2_L3.real.t, w_2_2_L3.real.data, 'r--')

plt.title(f"Sliced extrapolated Waveform, $N={extrapolation_order}$")
plt.xlabel(r"$(t_{\mathrm{corr}} - r_\ast)/M$")
plt.ylabel(r"$r\, h^{(\ell,m)}/M$")
plt.show()

# NR phase
Phase_L4 = -w_sliced_L4.arg_unwrapped[:, w_sliced_L4.index(2, 2)]
Phase_L3 = -w_sliced_L3.arg_unwrapped[:, w_sliced_L3.index(2, 2)]

Phase_L4_wrapd = -w_sliced_L4.arg[:, w_sliced_L4.index(2, 2)]
Phase_L3_wrapd = -w_sliced_L3.arg[:, w_sliced_L3.index(2, 2)]


# Instantanous frequency for comparing plots
Inst_frq_L4 = Phase_L4_wrapd.dot
frq_L4 = Phase_L4.dot

Inst_frq_ini = Inst_frq_L4.data.tolist()[0]

# Load data from teobResumm code 
t_1_h22, hp_1_h22, hc_1_h22 = np.loadtxt('/home/ashok/teobresums/C/data/hlm_ringdown_l2_m2_reim.txt', unpack=True)
t_2_h22, h22_amp, h22_phase = np.loadtxt('/home/ashok/teobresums/C/data/hlm_ringdown_l2_m2.txt', unpack=True)


# Keep data only 50M beyond peak
t_teob_max_idx = BOB.find_nearest1(hp_1_h22, max(hp_1_h22))
t_teob_end_idx = BOB.find_nearest1(t_1_h22, t_1_h22[t_teob_max_idx]) + 200

phase_teob = -np.unwrap(np.angle(hp_1_h22 + 1j*hc_1_h22))
omega = np.gradient(phase_teob,t_1_h22)

# Attaching plots where instantmous frequency

phase_teob_wrap = -np.angle(hp_1_h22 + 1j*hc_1_h22)
omega_wrap = np.gradient(phase_teob_wrap, t_1_h22)

t_NR_phase0_idx = BOB.find_nearest1(omega_wrap, Inst_frq_ini)

#### keep data only after f0 in teobNR data
t_1_h22 = t_1_h22[t_NR_phase0_idx:t_teob_end_idx]
hp_1_h22 = hp_1_h22[t_NR_phase0_idx:t_teob_end_idx]
hc_1_h22 = hc_1_h22[t_NR_phase0_idx:t_teob_end_idx]
phase_teob = phase_teob[t_NR_phase0_idx:t_teob_end_idx]
omega = omega[t_NR_phase0_idx:t_teob_end_idx]

######################################################

plt.plot(t_1_h22-t_1_h22[0], -hp_1_h22, label=r'teobRsum NR tuned $h_{22}$')
plt.plot(w_2_2_L3.real.t-w_2_2_L3.real.t[0], 3.9*w_2_2_L3.real.data, 'r--')
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

idx_strt=0
idx_end=-30
time_SXS_L4 =time_SXS_L4[idx_strt:idx_end]
Phase_L4 = Phase_L4[idx_strt:idx_end]

h22_phase_intrp=interp1d(time_h22, h22_phase, kind='cubic')
h22_phase_intrp=h22_phase_intrp(time_SXS_L4)
h22_phase_intrp-h22_phase_intrp
phase_diff_NR_teob = h22_phase_intrp-Phase_L4
phase_diff_NR_teob=phase_diff_NR_teob-phase_diff_NR_teob[0]

# Plot NR-EOB phase difference and NR phase uncertainity

plt.plot(time_SXS_L4, phase_diff_NR_teob, 'k')
plt.fill_between(time_SXS, -abs(Phase_errr), abs(Phase_errr), alpha=0.2)
plt.show()
# Compute quasinormal frrequencies given initial spins and symmetric mass ratio

def Mf_and_Omega_QNM(alpha1, alpha2, nu):
    p0 = 0.04826; p1 = 0.01559; p2 = 0.00485; s4 = -0.1229; s5 = 0.4537; 
    t0 = -2.8904; t2 = -3.5171; t3 = 2.5763; q = 1.0; eta = nu; theta = np.pi/2; 
    Mf = 1-p0 - p1*(alpha1+alpha2)-p2*pow(alpha1+alpha2,2)
    ab = (pow(q,2)*alpha1+alpha2)/(pow(q,2)+1)
    alpha = ab + s4*eta*pow(ab,2) + s5*pow(eta,2)*ab + t0*eta*ab + 2*np.sqrt(3)*eta + t2*pow(eta,2) + t3*pow(eta,3)
    OM_QNM = (1.0 - 0.63*pow(1.0 - alpha, 0.3))/(2*Mf)
    return alpha, Mf

nu = 0.25; alpha1=0.0; alpha2=0.0
af, Mf = Mf_and_Omega_QNM(alpha1, alpha2, nu)


# BOB waveform
# Final state variable
#af = 0.6864427
#Mf = 0.9517857
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

#t_2_h22, hp_2_h22, hc_2_h22 = np.loadtxt('/home/ashok/teobresums/C/data/hlm_ringdown_l2_m2_reim.txt', unpack=True)


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

t_ini_teob = -15.4 # time at which initial conditions are given

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


plt.plot(t_1_h22, hp_1_h22, label=r'teobRsum NR tuned $h_{22}$')
plt.plot(t_2_h22, hp_2_h22, 'k--', label=r'$h_{22}$ with $a_{5}=0$, $a_{6}=0$, $F_{H}=0$ and no NCQ')
plt.plot(t_BOB, hp_BOB, 'r--', label=r'$h_{22}$ BOB')
#plt.xlim(-100, 75)
plt.ylim(-2.0, 2.0)
plt.xlabel('time')
plt.ylabel(r'$h_{22}$')

plt.legend()
plt.show()

# Generate a complete waveform BOB+TEOB
BOB_idx_cut = 300

time_teob = t_2_h22[:t_ref_idx]
h22_teob = hp_2_h22[:t_ref_idx]
time_bob = t_BOB[t_ref_idx-BOB_idx_cut:]
h22_bob = hp_BOB[t_ref_idx-BOB_idx_cut:]

plt.plot(time_bob, h22_bob)
plt.plot(w_2_2_L3.real.t-time_l3_max, 3.9*w_2_2_L3.real.data, 'k--')
plt.plot(t_1_h22, hp_1_h22, 'c--')
plt.show()





