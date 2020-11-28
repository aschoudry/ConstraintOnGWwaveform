import sys
sys.path.insert(0, '../src')
import BOB_functions as BOB
import matplotlib.pyplot as plt
import numpy as np
import sxs
from scipy.interpolate import interp1d
import os
from EOBUtils import *

## Some good q=1 nospin BBH file nums (1132, 3, 4), (1155, 2, 3),(0002, 5,6), spinning (0160, 3,4)

file_num = '1122'
L1=4; L2=5

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

# Compute quasinormal frrequencies given initial spins and symmetric mass ratio

def Mf_and_af(alpha1, alpha2, nu):
    p0 = 0.04826; p1 = 0.01559; p2 = 0.00485; s4 = -0.1229; s5 = 0.4537; 
    t0 = -2.8904; t2 = -3.5171; t3 = 2.5763; q = 1.0; eta = nu; theta = np.pi/2 
    Mf = 1-p0 - p1*(alpha1+alpha2)-p2*pow(alpha1+alpha2,2)
    ab = (pow(q,2)*alpha1+alpha2)/(pow(q,2)+1)
    alpha = ab + s4*eta*pow(ab,2) + s5*pow(eta,2)*ab + t0*eta*ab + 2*np.sqrt(3)*eta + t2*pow(eta,2) + t3*pow(eta,3)
    OM_QNM = (1.0 - 0.63*pow(1.0 - alpha, 0.3))/(2*Mf)
    return alpha, Mf

nu = 0.25; alpha1=0.4376; alpha2=0.4376
af, Mf = Mf_and_af(alpha1, alpha2, nu)


# BOB waveform
# Final state variable
M_tot = 1.0
t_scale = M_tot*4.92549094830932e-06 

# quasinormal, tau frequency
om_qnm = 1.5251-1.1568*(1-af)**0.1292
om_qnm = om_qnm/Mf
Q = 0.7+1.4187*(1-af)**(-0.499)
OM_QNM = om_qnm/2
tau = 2*Q/om_qnm

# Generate data from TEOBRESUM

#!/usr/bin/python3

"""
Script to auto generate parfiles and run them

Given a template parfile with basic setup, 
generates parfiles for all combination of a given subset of parameters
(e.g. to vary binary masses and spins). Then it run the code.

SB 10/2018
"""
###############################################################################################################
import os

from EOBUtils import *

if __name__ == "__main__": 


    # Setup -----------------------------------------
    
    # Base dir & parfile
    based = "./"
    basep = "test_NQCIteration.par"

    # Set new values/ranges for parameters (Use lists)
    q = [1.]
    chi1 = [alpha1]
    chi2 = [alpha2]
    
    # Pack them into a dictionary
    # NOTE: keys must match those in parfile otherwise ignored
    n = {'q': q,
         'chi1': chi1,
         'chi2': chi2}
    
    # ------------------------------------------
    # DO NOT CHANGE BELOW HERE
    # ------------------------------------------

    # Generate parfiles ----------------------------
    
    # Read the base parfile
    d = read_parfile_dict(basep)

    # Generate combinations
    x, keys = combine_parameters(n)
    
    # Write parfiles
    parfile = []
    basen, ext = os.path.splitext(basep)
    for s in range(len(x)):
        for i in range(len(keys)):
            d[keys[i]] = str(x[s][i])
        # Output to file
        print(d)
        parfile.append(based+"/"+basen+"_"+str(s)+ext)
        write_parfile_dict(parfile[-1], d)
        print("Written {}".format(s))

    # Run  ----------------------------

    for p in parfile:
        run(p)
        os.remove(p)

#####################################################################################################################################
# Initial condition for BOB using various waveforms

t_2_h22, hp_2_h22, hc_2_h22 = np.loadtxt('/home/ashok/constraintongwwaveform/script/output/hlm_interp_l2_m2_reim.txt', unpack=True)

t_shift_2_idx = BOB.find_nearest1(hp_2_h22, max(hp_2_h22))
t_2_h22 = t_2_h22 - t_2_h22[t_shift_2_idx]

# Omega and Omega_dot from teobResum

h_tot = hp_2_h22 + 1j*hc_2_h22

amp = abs(h_tot)
phase = -np.unwrap(np.angle(h_tot))

omega = np.gradient(phase,t_2_h22)
omega_dot = np.gradient(omega,t_2_h22)

t_ini_teob = -23.5 # time at which initial conditions are given

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

# Generate a complete waveform BOB+TEOB
BOB_idx_cut = 300

time_teob = t_2_h22[:t_ref_idx]
h22_teob = hp_2_h22[:t_ref_idx]
time_bob = t_BOB[t_ref_idx-BOB_idx_cut:]
h22_bob = hp_BOB[t_ref_idx-BOB_idx_cut:]

plt.plot(time_bob, h22_bob)
plt.plot(w_2_2_L3.real.t-time_l3_max, 3.9*w_2_2_L3.real.data, 'k--')
plt.plot(t_2_h22, hp_2_h22, 'c--')
plt.xlim(-100.0, 80.0)
plt.show()

"""
Script to auto generate parfiles and run them

Given a template parfile with basic setup, 
generates parfiles for all combination of a given subset of parameters
(e.g. to vary binary masses and spins). Then it run the code.

SB 10/2018
"""






