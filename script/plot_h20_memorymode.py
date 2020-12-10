import sys
sys.path.insert(0, '../src')
import matplotlib.pyplot as plt
import numpy as np
import sxs
import BOB_functions as BOB


# Load SXS data
catalog = sxs.load("catalog")
file_num_S0 = '0001'
file_num_S0p99 = '0178'

L1=5

extrapolation_order = 4
w_L_S0 = sxs.load("SXS:BBH:"+str(file_num_S0)+"/Lev("+str(L1)+")/rhOverM", extrapolation_order=extrapolation_order)
w_L_S0p99 = sxs.load("SXS:BBH:"+str(file_num_S0p99)+"/Lev("+str(L1)+")/rhOverM", extrapolation_order=extrapolation_order)


metadata_S0 = sxs.load("SXS:BBH:"+str(file_num_S0)+"/Lev("+str(L1)+")/metadata.json")
metadata_S0p99 = sxs.load("SXS:BBH:"+str(file_num_S0p99)+"/Lev("+str(L1)+")/metadata.json")

'''
#Load data used in memory paper
#data location
file_location_S0 ='/home/ashok/gravitational_wave_memory_project/data/NonSpinning_differentMassRatio/Memory_data/'
file_location_S0p99 ='/home/ashok/gravitational_wave_memory_project/data/SXSdata/Spinning_binary_with_SpinAligned_27Dec/Memory_data/'

#import data
filename='q1'
datafile_hMemNR_S0='rMPsi4_noSpin_'+filename+'dataClean_hMemNR.dat'
timeNR_S0, hmem_S0, h_mem_plus_S0 = np.loadtxt(file_location_S0+datafile_hMemNR_S0, unpack=True)
hmem_S0*=17.0

datafile_hMemNR_S0p99='rMPsi4_Sz1_0p99_Sz2_0p99_q1p5dataN4Clean_hMemNR.dat'
timeNR_S0p99, hmem_S0p99, h_mem_plus_S0p99 = np.loadtxt(file_location_S0p99+datafile_hMemNR_S0p99, unpack=True)
hmem_S0p99*=17.0


# Spinweighted spherical harmonics
m2Y20 = 3.0/4.0*np.sqrt(5.0/6.0/np.pi)

# Compute memory terms

reference_index_S0 = w_L_S0.index_closest_to(metadata_S0.reference_time)
reference_index_S0p99 = w_L_S0p99.index_closest_to(metadata_S0p99.reference_time)

h_sliced_S0 = w_L_S0[reference_index_S0:]
h_sliced_S0p99 = w_L_S0p99[reference_index_S0p99:]


h_mem_S0 = sxs.waveforms.memory.J_E(h_sliced_S0, start_time=1000)
h_mem_S0p99 = sxs.waveforms.memory.J_E(h_sliced_S0p99, start_time=1000)

time_S0 = h_mem_S0.t 
time_S0p99 = h_mem_S0p99.t 


h_mem_20_S0 = (h_mem_S0)[:,h_mem_S0.index(2, 0)].data.real*m2Y20
h_mem_20_S0p99 = (h_mem_S0p99)[:,h_mem_S0p99.index(2, 0)].data.real*m2Y20

# shift merger time BOB.find_nearest1(hp_2_h22, max(hp_2_h22))
idx_SXS_S0 = BOB.find_nearest1(h_mem_20_S0, 0.5*max(h_mem_20_S0))
idx_SXS_S0p99 = BOB.find_nearest1(h_mem_20_S0p99, 0.5*max(h_mem_20_S0p99))
idx_FVT_S0 = BOB.find_nearest1(hmem_S0, 0.5*max(hmem_S0))
idx_FVT_S0p99 = BOB.find_nearest1(hmem_S0p99, 0.5*max(hmem_S0p99))

time_S0 = time_S0-time_S0[idx_SXS_S0]
time_S0p99 = time_S0p99-time_S0p99[idx_SXS_S0p99]
timeNR_S0 = timeNR_S0 - timeNR_S0[idx_FVT_S0]
timeNR_S0p99 = timeNR_S0p99 - timeNR_S0p99[idx_FVT_S0p99]



plt.plot(time_S0p99, h_mem_20_S0p99,'c', label=r"0.99 SXS code")
plt.plot(time_S0, h_mem_20_S0,'k', label=r"0.00 SXS code")
plt.plot(timeNR_S0, hmem_S0,'r--', label=r"0.00 Favata's epression")
plt.plot(timeNR_S0p99, hmem_S0p99,'b--', label=r"0.99 Favata's epression")
plt.xlim(-2000, 500)
plt.ylabel(r"$J_E^{(2,0)}$")
plt.xlabel(r"$(t_{\mathrm{corr}} - r_\ast)/M$")
plt.legend()
plt.savefig("/home/ashok/constraintongwwaveform/plots/MemoryComparisioSXSpkgVsFavata.pdf")
plt.show()
'''
#Compute Psi2
ADM_Mass_tot = metadata_S0['initial_mass1'] + metadata_S0['initial_mass2']
print(ADM_Mass_tot)

h = w_L_S0
reference_index =w_L_S0.index_closest_to(metadata_S0.reference_time)
h_sliced = w_L_S0[reference_index:]
h_mem = sxs.waveforms.memory.J_E(h_sliced, start_time=1000)

h_mem_20 = (h_mem)[:,h_mem.index(2, 0)].data.real
h_mem_21 = (h_mem)[:,h_mem.index(2, 1)].data.real
h_mem_22 = (h_mem)[:,h_mem.index(2, 2)].data.real


mts_h = sxs.waveforms.memory.MTS(h_sliced)
eth2_h = mts_h.eth.eth
h_dot_h_bar = (mts_h.dot*mts_h.bar)

psi2 = 0.25*(eth2_h+h_dot_h_bar).re 
psi2_JE22 = psi2# + 0.25*h_mem

time = psi2.t 

psi2_20 = (psi2)[:,psi2.index(2, 0)].data.real
psi2_21 = (psi2)[:,psi2.index(2, 1)].data.real
psi2_22 = (psi2)[:,psi2.index(2, 2)].data.real


#Add memory terms to the equations
psi2_JE_20 = psi2_20 + h_mem_20
psi2_JE_21 = psi2_21 + h_mem_21
psi2_JE_22 = psi2_22 + h_mem_22


plt.plot(time, psi2_20, label=r"$\Psi^{(2,0)}$")
plt.plot(time, psi2_21, label=r"$\Psi^{(2,1)}$")
plt.plot(time, psi2_22, label=r"$\Psi^{(2,2)}$")
plt.ylabel(r"$\Psi^{(l,m)}$")
plt.xlabel(r"$(t_{\mathrm{corr}} - r_\ast)/M$")
plt.xlim(9400, 9600)
plt.title(r'$\Re[\Psi] = \frac{1}{4}\{-\Re[\eth^2 h]-\Re[\dot{h}\bar{h}]\}$')
plt.legend()
plt.savefig("/home/ashok/constraintongwwaveform/plots/Psi2_SXSequation_Nomem.pdf")

plt.show()

plt.plot(time, psi2_JE_20, label=r"$\Psi^{(2,0)}$")
plt.plot(time, psi2_JE_21, label=r"$\Psi^{(2,1)}$")
plt.plot(time, psi2_JE_22, label=r"$\Psi^{(2,2)}$")
plt.ylabel(r"$\Psi^{(l,m)}$")
plt.xlabel(r"$(t_{\mathrm{corr}} - r_\ast)/M$")
plt.xlim(9400, 9600)
plt.title(r'$\Re[\Psi] = \frac{1}{4}\{\int^{u}_{0}\|\dot{h}\|^2 du - \Re[\eth^2 h]-\Re[\dot{h}\bar{h}]\}$')
plt.legend()
plt.savefig("/home/ashok/constraintongwwaveform/plots/Psi2_SXSequation_mem.pdf")

plt.show()


