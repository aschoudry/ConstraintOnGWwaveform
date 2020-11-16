import sys
sys.path.insert(0, '../src')
import BOB_functions as BOB
import matplotlib.pyplot as plt
import numpy as np
import sxs
from scipy.interpolate import interp1d

## Some good q=1 nospin BBH file nums (1132, 3, 4), (1155, 2, 3),(0002, 5,6)

file_num = 1375
L1=3

extrapolation_order = 4
w_L3 = sxs.load("SXS:BBH:"+str(file_num)+"/Lev("+str(L1)+")/rhOverM", extrapolation_order=extrapolation_order)

# Remove junk radiation
metadata_L3 = sxs.load("SXS:BBH:"+str(file_num)+"/Lev("+str(L1)+")/metadata.json")
metadata_L3.reference_time

shift = 160
reference_index_L3 = w_L3.index_closest_to(metadata_L3.reference_time)
w_sliced_L3 = w_L3[reference_index_L3+shift:]

#Plot h22 waveform at differnet resolution
t_1_h22, hp_1_h22, hc_1_h22 = np.loadtxt('/home/ashok/teobresums/C/dataS10p8S20/hlm_interp_l2_m2_reim.txt', unpack=True)



w_2_2_L3 = w_sliced_L3[:, w_sliced_L3.index(2, 2)]

plt.plot(t_1_h22-t_1_h22[-1], hp_1_h22)
plt.plot(w_2_2_L3.real.t-w_2_2_L3.real.t[-1]-56, max(hp_1_h22)*w_2_2_L3.real.data/max(w_2_2_L3.real.data), 'r--')
plt.xlim(-6500, 50)
plt.title(f"Sliced extrapolated Waveform, $N={extrapolation_order}$")
plt.xlabel(r"$(t_{\mathrm{corr}} - r_\ast)/M$")
plt.ylabel(r"$r\, h^{(\ell,m)}/M$")
plt.show()

