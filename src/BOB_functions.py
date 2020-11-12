import numpy as np

def find_nearest1(array,value):
    idx,val = min(enumerate(array), key=lambda x: abs(x[1]-value))
    return idx

def find_nearest1(array,value):
    idx,val = min(enumerate(array), key=lambda x: abs(x[1]-value))
    return idx

def OMEGA_p_pow4(OM0, OM0_dot, OMQNM, tau):
    OMQNMp4 = OMQNM**4; OM0p4 = OM0**4; OM0p3 = OM0**3
    tanh_arg = -np.log(((OMQNMp4 - OM0p4)/(2.0*tau* OM0p3 *OM0_dot))-1.0)/2.0
    OM_p_pow4 = (OM0p4-OMQNMp4*np.tanh(tanh_arg))/(1.0-np.tanh(tanh_arg))
    return OM_p_pow4

def OMEGA_m_pow4(OM0, OM0_dot, OMQNM, tau):
    OMQNMp4 = OMQNM**4; OM0p4 = OM0**4; OM0p3 = OM0**3
    tanh_arg = -np.log(((OMQNMp4 - OM0p4)/(2.0*tau* OM0p3 *OM0_dot))-1.0)/2.0
    OM_m_pow4 = (OMQNMp4-OM0p4)/(1.0-np.tanh(tanh_arg))
    return OM_m_pow4

def kapp(OM0, OM0_dot, OMQNM, tau):
    OMQNMp4 = OMQNM**4; OM0p4 = OM0**4; OM0p3 = OM0**3
    tanh_arg = -np.log(((OMQNMp4 - OM0p4)/(2.0*tau*OM0p3*OM0_dot))-1.0)/2.0
    OM_m_pow4 = (OMQNMp4-OM0p4)/(1.0-np.tanh(tanh_arg))
    kp_p = (OM0**4 + OM_m_pow4*(1.0 - np.tanh(tanh_arg)))**0.25
    kp_m = (OM0**4 - OM_m_pow4*(1.0 + np.tanh(tanh_arg)))**0.25
    return kp_p, kp_m

def t_ph(tp, OM0, OM0_dot, OMQNM, tau):
    OM_p_pow4 = OMEGA_p_pow4(OM0, OM0_dot, OMQNM, tau)
    OM_m_pow4 = OMEGA_m_pow4(OM0, OM0_dot, OMQNM, tau)
    tph = tp + (tau/2.0)*np.arctanh((OM_p_pow4/OM_m_pow4)-np.sqrt((OM_p_pow4/OM_m_pow4)**2-1.0))
    return tph

def t_peak(t0, OM0, OM0_dot, OMQNM, tau):
    OMQNMp4 = OMQNM**4; OM0p4 = OM0**4; OM0p3 = OM0**3
    tanh_arg = -np.log(((OMQNMp4 - OM0p4)/(2.0*tau* OM0p3 *OM0_dot))-1.0)/2.0
    tp = t0 - tau*tanh_arg
    return tp

def Omega_BOB(t, tp, OM0, OM0_dot, OMQNM, tau):
    OM_p_pow4 = OMEGA_p_pow4(OM0, OM0_dot, OMQNM, tau)
    OM_m_pow4 = OMEGA_m_pow4(OM0, OM0_dot, OMQNM, tau)
    OM = (OM_p_pow4 + OM_m_pow4*np.tanh((t-tp)/tau))**0.25
    return OM

def Phi_BOB(t, tp, OM0, OM0_dot, OMQNM, tau):
    OM_BOB = Omega_BOB(t, tp, OM0, OM0_dot, OMQNM, tau)
    kp = kapp(OM0, OM0_dot, OMQNM, tau)[0]
    km = kapp(OM0, OM0_dot, OMQNM, tau)[1]
    arctan_p = kp*tau*(np.arctan2(OM_BOB,kp)-np.arctan2(OM0,kp))
    arctan_m = km*tau*(np.arctan2(OM_BOB,km)-np.arctan2(OM0,km))
    arctanh_p = kp*tau*(np.arctanh(OM_BOB/kp)-np.arctanh(OM0/kp))
    arctanh_m = km*tau*0.5*np.log( (OM_BOB/km + 1)*(1 - OM0/km)/(1 - OM_BOB/km)/(1 + OM0/km))
    return arctan_p + arctanh_p - arctan_m - arctanh_m

def A_peak(A0, t0, tp, tau):
    return A0*np.cosh((t0-tp)/tau)

def Psi4_BOB_amp(t, Ap, tau, tp):
    return Ap/np.cosh((t-tp)/tau)

def h_amp_BOB(t, tp, Ap, OM0, OM0_dot, OMQNM, tau):
    psi4_BOB = Psi4_BOB_amp(t, Ap, tau, tp)
    OM_BOB = Omega_BOB(t, tp, OM0, OM0_dot, OMQNM, tau)
    return psi4_BOB/(4.0*OM_BOB**2)

def Waveform_BOB(t, tp, Ap, OM0, OM0_dot, OMQNM, tau, phase0):
    amp = h_amp_BOB(t, tp, Ap, OM0, OM0_dot, OMQNM, tau)
    phase = Phi_BOB(t, tp, OM0, OM0_dot, OMQNM, tau)- Phi_BOB(t, tp, OM0, OM0_dot, OMQNM, tau)[0] +  phase0
    return amp*np.exp(2j*phase)

