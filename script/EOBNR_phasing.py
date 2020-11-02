import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import h5py
from scipy.interpolate import interp1d

def find_nearest1(array,value):
    idx,val = min(enumerate(array), key=lambda x: abs(x[1]-value))
    return idx

# Reading the strain data
file_name_h_L4 = "/home/ashok/constraintongwwaveform/data/rhOverM_Asymptotic_GeometricUnits_L4.h5"
file_name_h_L5 = "/home/ashok/constraintongwwaveform/data/rhOverM_Asymptotic_GeometricUnits_L5.h5"

f_h_L4 = h5py.File(file_name_h_L4,'r+')
f_h_L5 = h5py.File(file_name_h_L5,'r+')

#reading data for time and l2m2 mode
data_h_L4 = f_h_L4['Extrapolated_N4.dir']['Y_l2_m2.dat'][:]
data_h_L5 = f_h_L5['Extrapolated_N4.dir']['Y_l2_m2.dat'][:]

time_L4 = np.array([])
time_L5 = np.array([])

h22_real_SXS_L4=np.array([])
h22_imag_SXS_L4=np.array([])
h22_real_SXS_L5=np.array([])
h22_imag_SXS_L5=np.array([])


for i in range(len(data_h_L4)):
    time_L4=np.append(time_L4, data_h_L4[i,0])
    h22_real_SXS_L4=np.append(h22_real_SXS_L4,data_h_L4[i,1])
    h22_imag_SXS_L4=np.append(h22_imag_SXS_L4,data_h_L4[i,2])

for i in range(len(data_h_L5)):
    time_L5=np.append(time_L5, data_h_L5[i,0])
    h22_real_SXS_L5=np.append(h22_real_SXS_L5,data_h_L5[i,1])
    h22_imag_SXS_L5=np.append(h22_imag_SXS_L5,data_h_L5[i,2])


Amp_L4 = abs(h22_real_SXS_L4 + 1j*h22_imag_SXS_L4)
Phase_L4 = -np.unwrap(np.angle(h22_real_SXS_L4 + 1j*h22_imag_SXS_L4))

Amp_L5 = abs(h22_real_SXS_L5 + 1j*h22_imag_SXS_L5)
Phase_L5 = -np.unwrap(np.angle(h22_real_SXS_L5 + 1j*h22_imag_SXS_L5))

#making plots
plt.plot(time_L4, h22_real_SXS_L4, 'g', label="h22 real")
plt.plot(time_L5, h22_real_SXS_L5,'r--', label="h22 real SXS")
plt.xlabel('time')
plt.ylabel('h22 amplitude')
plt.legend()
plt.show()

#plt.savefig("/home/ashok/gravitational_wave_memory_project/plots/h22_plot2.pdf")

plt.plot(time_L4, Phase_L4, 'g', label="h22 real")
plt.plot(time_L5, Phase_L5, 'r--', label="h22 real")
#plt.plot(time_L5, h22_real_SXS_L5,'r--', label="h22 real SXS")
plt.xlabel('time')
plt.ylabel('h22 ')
plt.legend()
plt.show()

Phase_L4_intrp=interp1d(time_L4, Phase_L4, kind='cubic')
Amp_L4_intrp=interp1d(time_L4, Amp_L4, kind='cubic')
Phase_L5_intrp=interp1d(time_L5, Phase_L5, kind='cubic')
Amp_L5_intrp=interp1d(time_L5, Amp_L5, kind='cubic')


Phase_L4_intrp=Phase_L4_intrp(time_L5)
Amp_L4_intrp=Amp_L4_intrp(time_L5)

Phase_L5_intrp=Phase_L5_intrp(time_L5)
Amp_L5_intrp=Amp_L5_intrp(time_L5)

t_shift_NR_L4_idx = find_nearest1(h22_real_SXS_L4, max(h22_real_SXS_L4))
time_L4 = time_L4 - time_L4[t_shift_NR_L4_idx]

t_shift_NR_L5_idx = find_nearest1(h22_real_SXS_L5, max(h22_real_SXS_L5))
time_L5 = time_L5 - time_L5[t_shift_NR_L5_idx]

plt.plot(time_L5, abs(Phase_L5_intrp-Phase_L4_intrp))
plt.fill_between(time_L5, -abs(Phase_L5_intrp-Phase_L4_intrp), abs(Phase_L5_intrp-Phase_L4_intrp), alpha=0.2)
plt.show()

# Loda data for EOB dynamics
EOB_h22_filepath = '/home/ashok/teobresums/C/bbh_q1_s0s0_M46_30Hz_postadiab/hlm_ringdown_l2_m2_reim.txt'
EOB_h22_amp_filepath = '/home/ashok/teobresums/C/bbh_q1_s0s0_M46_30Hz_postadiab/hlm_ringdown_l2_m2.txt'

t_h22, h22_re, h22_Im = np.loadtxt(EOB_h22_filepath, unpack=True)
t_h22_amp, h22_amp, h22_phase = np.loadtxt(EOB_h22_amp_filepath, unpack=True)

amp_teob = abs(h22_re + 1j*h22_Im)
phase_teob = -np.unwrap(np.angle(h22_re + 1j*h22_Im))

t_shift_eob_idx = find_nearest1(h22_re, max(h22_re))
t_h22_teob = t_h22 - t_h22[t_shift_eob_idx]

EOB_h22_filepath_noNqc = '/home/ashok/teobresumsNoNRtn/teobresums/C/bbh_q1_noSpin_NRtnd_a5_a6_No_HrznFlx_No_nqcOutput/hlm_ringdown_l2_m2_reim.txt'
EOB_h22_amp_filepath_noNqc = '/home/ashok/teobresumsNoNRtn/teobresums/C/bbh_q1_noSpin_NRtnd_a5_a6_No_HrznFlx_No_nqcOutput/hlm_ringdown_l2_m2.txt'

t_h22_noNqc, h22_re_noNqc, h22_Im_noNqc = np.loadtxt(EOB_h22_filepath_noNqc, unpack=True)
t_h22_amp_noNqc, h22_amp_noNqc, h22_phase_noNqc = np.loadtxt(EOB_h22_amp_filepath_noNqc, unpack=True)

amp_teob_noNqc = abs(h22_re_noNqc + 1j*h22_Im_noNqc)
phase_teob_noNqc = -np.unwrap(np.angle(h22_re_noNqc + 1j*h22_Im_noNqc))

t_shift_eob_idx_noNqc = find_nearest1(h22_re_noNqc, max(h22_re_noNqc))
t_h22_teob_noNqc = t_h22_noNqc - t_h22_noNqc[t_shift_eob_idx_noNqc]



plt.plot(time_L5, h22_real_SXS_L5, 'g')
plt.plot(t_h22_teob, h22_re*max(h22_real_SXS_L5)/max(h22_re))
plt.plot(t_h22_teob_noNqc, h22_re_noNqc*max(h22_real_SXS_L5)/max(h22_re_noNqc), 'k--')
plt.show()


