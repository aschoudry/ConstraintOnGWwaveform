import sys
import numpy as np
import pylab
import matplotlib.pyplot as plt
from pycbc import waveform
from pycbc.waveform import get_td_waveform

## Ignore the PN Orders, the main function to get the waveform is get_td_waveform.

pnorder = ['LAL_PNORDER_ONE', 'LAL_PNORDER_TWO','LAL_PNORDER_THREE','LAL_PNORDER_NUM_ORDER']
HP , HC = get_td_waveform(approximant = "SEOBNRv4",mass1 = 23 , mass2 = 23 , delta_t = 1.0/4096 , f_lower = 30,LALPNOrder = pnorder[-1])


H_tot = HP + 1j* HC

H_phase=np.unwrap(np.angle(H_tot))
H_amp = abs(H_tot)/max(abs(H_tot))
H_complex = np.array(H_amp*np.exp(1j*H_phase))
H_time = np.linspace(HP.sample_times[0],HP.sample_times[-1], len(H_amp))


print(max(H_complex.real))
#plt.plot(HP.sample_times/(40*4.9*pow(10, -6)), HP/max(abs(H_tot)))
#plt.plot(HP.sample_times/(40*4.9*pow(10, -6)), H_complex.real , '--k')

#plt.xlim(-200, 50)
#plt.show()


f = open("/home/ashok/constraintongwwaveform/data/EOB_wf.txt","w") 

for i in range(len(HP)):
    f.write("%E %E %E\n" % (HP.sample_times[i]/(46*4.93*pow(10, -6)), H_amp[i], H_phase[i])) 

f.close()


