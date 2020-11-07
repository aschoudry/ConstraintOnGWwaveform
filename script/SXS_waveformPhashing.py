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



plt.plot(w_2_2_L3.real.t-w_2_2_L3.real.t[0], w_2_2_L3.real.data, 'r--')
plt.plot(t_1_h22-t_1_h22[0]+31, hp_1_h22*(max(w_2_2_L3.real.data)/max(hp_1_h22)), label=r'teobRsum NR tuned $h_{22}$')
#plt.xlim(-100, 75)
#plt.ylim(-2.0, 2.0)
plt.xlabel('time')
plt.ylabel(r'$h_{22}$')
plt.show()

# Phase difference between teobResumm and SXS waveform
Phase_errr = Phase_L3-Phase_L4.interpolate(w_sliced_L3.t)
time_h22 = t_1_h22-t_1_h22[0]
time_SXS_L4 = w_sliced_L4.t-w_2_2_L3.real.t[0]
time_SXS_L3 = w_sliced_L3.t-w_2_2_L3.real.t[0]

Phase_L4 = Phase_L4-Phase_L4[0]
h22_phase = h22_phase- h22_phase[0]

plt.plot(time_h22, h22_phase)
plt.plot(time_SXS_L4, Phase_L4, 'r--')
plt.show()

h22_phase_intrp=interp1d(time_h22, h22_phase, kind='cubic')
h22_phase_intrp=h22_phase_intrp(time_SXS_L4)
h22_phase_intrp-h22_phase_intrp-h22_phase_intrp[0]


plt.plot(time_SXS_L4, h22_phase_intrp-Phase_L4, 'k')
plt.fill_between(time_SXS_L3, -abs(Phase_errr), abs(Phase_errr), alpha=0.2)
plt.show()

