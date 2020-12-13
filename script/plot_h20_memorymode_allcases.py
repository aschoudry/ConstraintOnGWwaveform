import sys
sys.path.insert(0, '../src')
import matplotlib.pyplot as plt
import numpy as np
import sxs
import BOB_functions as BOB


# Load SXS data
catalog = sxs.load("catalog")
dataframe = catalog.table
#BHBH = dataframe[(dataframe["object_types"] == "BHBH") & (dataframe["initial_mass_ratio"] <1.01)]
BHBH = dataframe[(dataframe["object_types"] == "BHBH") & (abs(dataframe["reference_chi_eff"]) <0.000001) & (abs(dataframe["reference_chi1_perp"]) <0.000001) & (abs(dataframe["reference_chi1_perp"]) <0.000001)]


#file_name = BHBH.index[0]

for file_name in BHBH.index[0:-1:5]:
    extrapolation_order = 4
    w_L = sxs.load(file_name+"/Lev/rhOverM", extrapolation_order=extrapolation_order)


    metadata = sxs.load(file_name+"/Lev/metadata.json")

    #Compute Psi2
    ADM_Mass_tot = metadata['initial_mass1'] + metadata['initial_mass2']
    s1x, s1y, s1z = metadata['initial_dimensionless_spin1']
    s2x, s2y, s2z = metadata['initial_dimensionless_spin2']
    q = metadata['initial_mass_ratio']

    reference_index =w_L.index_closest_to(metadata.reference_time)
    h_sliced = w_L[reference_index:]
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

    #Shift time such that peak is at t=0
    idx_shift = BOB.find_nearest1(h_mem_20, 0.5*max(h_mem_20))
    time = time-time[idx_shift]

    psi2_20 = (psi2)[:,psi2.index(2, 0)].data.real
    psi2_21 = (psi2)[:,psi2.index(2, 1)].data.real
    psi2_22 = (psi2)[:,psi2.index(2, 2)].data.real


    #Add memory terms to the equations
    psi2_JE_20 = psi2_20 + h_mem_20
    psi2_JE_21 = psi2_21 + h_mem_21
    psi2_JE_22 = psi2_22 + h_mem_22

    plt.figure()
    plt.plot(time, psi2_20, label=r"$\Psi^{(2,0)}$")
    plt.plot(time, psi2_21, label=r"$\Psi^{(2,1)}$")
    plt.plot(time, psi2_22, label=r"$\Psi^{(2,2)}$")
    plt.ylabel(r"$\Psi^{(l,m)}$")
    plt.xlabel(r"$(t_{\mathrm{corr}} - r_\ast)/M$")
    plt.xlim(-500, 200)
    plt.title(r'$\Re[\Psi] = \frac{1}{4}\{-\Re[\eth^2 h]-\Re[\dot{h}\bar{h}]\}$'+'\n q = '+str(round(q,2))+'spin1 = [' + str(round(s1x,2)) + ',' +str(round(s1y,2))+ ',' +str(round(s1z,2))+']'+', spin2 = [' + str(round(s2x,2)) + ',' +str(round(s2y,2))+ ',' +str(round(s2z,2))+']')
    plt.legend()
    plt.savefig("/home/ashok/constraintongwwaveform/plots/Nonspinning/Psi2_SXSequation_Nomem"+'_spin1=_'+str(round(s1x,2))+'_'+str(round(s1y,2))+'_'+str(round(s1z,2))+'_'+'_spin2=_'+str(round(s2x,2))+'_'+str(round(s2y,2))+'_'+str(round(s2z,2))+'q_'+str(round(q,2))+'zoom_out.png')
    
    plt.close()

    plt.figure()
    plt.plot(time, psi2_JE_20, label=r"$\Psi^{(2,0)}$")
    plt.plot(time, psi2_JE_21, label=r"$\Psi^{(2,1)}$")
    plt.plot(time, psi2_JE_22, label=r"$\Psi^{(2,2)}$")
    plt.ylabel(r"$\Psi^{(l,m)}$")
    plt.xlabel(r"$(t_{\mathrm{corr}} - r_\ast)/M$")
    plt.xlim(-500, 200)
    plt.title(r'$\Re[\Psi] = \frac{1}{4}\{\int^{u}_{0}\|\dot{h}\|^2 du - \Re[\eth^2 h]-\Re[\dot{h}\bar{h}]\}$'+'\n q = '+str(round(q,2))+'spin1 = [' + str(round(s1x,2)) + ',' +str(round(s1y,2))+ ',' +str(round(s1z,2))+']'+', spin2 = [' + str(round(s2x,2)) + ',' +str(round(s2y,2))+ ',' +str(round(s2z,2))+']')
    plt.legend()
    plt.savefig("/home/ashok/constraintongwwaveform/plots/Nonspinning/Psi2_SXSequation_mem"+'_spin1=_'+str(round(s1x,2))+'_'+str(round(s1y,2))+'_'+str(round(s1z,2))+'_'+'_spin2=_'+str(round(s2x,2))+'_'+str(round(s2y,2))+'_'+str(round(s2z,2))+'q_'+str(round(q,2))+'zoom_out.png')
    plt.close()

    plt.figure()
    plt.plot(time, h_mem_20, label=r"$h_{mem}^{(2,0)}$")
    plt.plot(time, h_mem_21, label=r"$h_{mem}^{(2,1)}$")
    plt.plot(time, h_mem_22, label=r"$h_{mem}^{(2,2)}$")
    plt.ylabel(r"$h_{mem}^{(l,m)}$")
    plt.xlabel(r"$(t_{\mathrm{corr}} - r_\ast)/M$")
    plt.xlim(-500,200)
    plt.title(r'$h_{mem}$'+'\n'+str(round(q,2))+' spin1 = [' + str(round(s1x,2)) + ',' +str(round(s1y,2))+ ',' +str(round(s1z,2))+']'+', spin2 = [' + str(round(s2x,2)) + ',' +str(round(s2y,2))+ ',' +str(round(s2z,2))+']')
    plt.legend()
    plt.savefig("/home/ashok/constraintongwwaveform/plots/Nonspinning/h_mem"+'_spin1=_'+str(round(s1x,2))+'_'+str(round(s1y,2))+'_'+str(round(s1z,2))+'_'+'_spin2=_'+str(round(s2x,2))+'_'+str(round(s2y,2))+'_'+str(round(s2z,2))+'q_'+str(round(q,2))+'zoom_out.png')

    plt.close()

