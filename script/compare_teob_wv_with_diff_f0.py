import matplotlib.pyplot as plt
import numpy as np

t_1_h22, hp_1_h22, hc_1_h22 = np.loadtxt('/home/ashok/teobresums/C/data_f0_0p0004/hlm_interp_l2_m2_reim.txt', unpack=True)
t_2_h22, hp_2_h22, hc_2_h22 = np.loadtxt('/home/ashok/teobresums/C/data_f0_0p004/hlm_interp_l2_m2_reim.txt', unpack=True)

t_1_h22 = t_1_h22-t_1_h22[-1]
t_2_h22 = t_2_h22-t_2_h22[-1]

t_1_h22 = t_1_h22[-2*len(t_2_h22):]
hp_1_h22 = hp_1_h22[-2*len(t_2_h22):]


plt.plot(t_1_h22, hp_1_h22)
plt.plot(t_2_h22 + 5, hp_2_h22, 'r--')
plt.show()


