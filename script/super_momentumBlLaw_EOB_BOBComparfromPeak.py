import sys
sys.path.insert(0, '../src')

from scipy.integrate import odeint 
import numpy as np 
from numpy.core.umath_tests import inner1d
import numpy as np
import HamiltonianCoupledODE_BOB_v2 as ODEs
import matplotlib.pyplot as plt
import PN_radiationReaction as PN_RR
import scipy.optimize as opt

# create a time vector with variable resolution so that time step are smaller as purterber inspirals in. 
def time_vec(r0, rf, ng_orbit, ng_radial):
    t_vec = np.array([])
    r_vec = np.linspace(r0, rf, ng_radial) 
    dt_vec = (1.0/ng_radial)*np.sqrt(4*pow(np.pi,2)*pow(r_vec,3)) 
    dt_vec_full = np.repeat(dt_vec, ng_orbit) 
    t_vec_full = np.cumsum(dt_vec_full) 
    return t_vec_full, dt_vec_full 

# find intex of nearest value in an array
def find_nearest1(array,value):
    idx,val = min(enumerate(array), key=lambda x: abs(x[1]-value))
    return idx

# Spherical harmics for 22 mode
def Y22(Theta, Phi):
    return (1.0/4.0)*np.sqrt(15.0/(2*np.pi))*np.sin(Theta)*np.sin(Theta)*np.exp(1j*2.0*Phi)

# Spin weight -2 spherical harmics for 22 mode
def m2Y22(Theta, Phi):
    return (1.0/8)*np.sqrt(5.0/np.pi)*(1+np.cos(Theta))*(1+np.cos(Theta))*np.exp(1j*2.0*Phi)

# Initial orbital momentum p_phi for a initial circular orbit  
def Circular_orbit_ini_P_phi(r, nu):
    return np.sqrt(ODEs.Ap(r, nu)*(r**2)/(2*ODEs.A(r, nu)/r - ODEs.Ap(r, nu)))

# Psi due to back ground mass evolving (essentially Psi = M_ADM = H_Real)
def Psi_Growth(r, p_r, phi, p_phi, p):
    HEOB = ODEs.Heob(r, p_r, phi, p_phi, p)
    HEFF = ODEs.Heff(r, p_r, phi, p_phi, p)
    M = HEOB*p + HEFF*p
    return -np.sqrt(2)*M

# Psi due to back ground mass evolving (essentially Psi = M_ADM + Pr = H_Real + Pr), where Pr is the radial linenar momentum of purterber
# Equation 1.9 in notes 
def Psi_Growth_Pr(r, p_r, phi, p_phi, p):
    HEOB = ODEs.Heob(r, p_r, phi, p_phi, p)
    HEFF = ODEs.Heff(r, p_r, phi, p_phi, p)
    M = HEOB*p + HEFF*p
    Pr = p_r*p
    a = ODEs.A(r, p)
    b = ODEs.B(r, p)
    return -np.sqrt(2)*M + -np.sqrt(2)*(6.0)*np.sqrt(b/a)*Pr 

def Psi_Growth_Pr_BOB(r, p_r, phi, p_phi, p):
    Pr = p_r*p
    Pr = Pr
    a = ODEs.A(r, p)
    b = ODEs.B(r, p)
    return (6.0/(2.0))*np.sqrt(b/a)*Pr 

# left hand side of the balance equation Intgral(Sigma_dot_sqr*dt)
def Intg_SigmaDot_sqr(t_vec, r, p_r, phi, p_phi, p):
    abs_Sigma_Dot_sqr = (2*ODEs.dphi_by_dt(r, p_r, phi, p_phi, p)*abs(ODEs.Rh22(r, p_r, phi, p_phi, p)))**2
    abs_Sigma_Dot_sqr = abs_Sigma_Dot_sqr
    dt_vec = np.diff(t_vec)
    dt_vec = np.append(dt_vec, dt_vec[-1])
    Int_abs_Sigma_Dot_sqr = -np.cumsum(abs_Sigma_Dot_sqr*dt_vec)*abs(m2Y22(np.pi/2, np.pi/2.0))**2
    Psi = Int_abs_Sigma_Dot_sqr
    return Psi

#### Functions that compute BOB waveform ###########################################################################################
# Compute quasinormal frrequencies given initial spins and symmetric mass ratio
def Omega_QNM(alpha1, alpha2, nu):
    p0 = 0.04826; p1 = 0.01559; p2 = 0.00485; s4 = -0.1229; s5 = 0.4537; 
    t0 = -2.8904; t2 = -3.5171; t3 = 2.5763; q = 1.0; eta = nu; theta = np.pi/2; 
    Mf = 1-p0 - p1*(alpha1+alpha2)-p2*pow(alpha1+alpha2,2)
    ab = (pow(q,2)*alpha1+alpha2)/(pow(q,2)+1)
    alpha = ab + s4*eta*pow(ab,2) + s5*pow(eta,2)*ab + t0*eta*ab + 2*np.sqrt(3)*eta + t2*pow(eta,2) + t3*pow(eta,3)
    OM_QNM = (1.0 - 0.63*pow(1.0 - alpha, 0.3))/(2*Mf)
    return OM_QNM

def Tau_v2(alpha1, alpha2, nu):
    p0 = 0.04826; p1 = 0.01559; p2 = 0.00485; s4 = -0.1229; s5 = 0.4537; 
    t0 = -2.8904; t2 = -3.5171; t3 = 2.5763; q = 1.0; eta = nu; theta = np.pi/2; 
    Mf = 1-p0 - p1*(alpha1+alpha2)-p2*pow(alpha1+alpha2,2)
    ab = (pow(q,2)*alpha1+alpha2)/(pow(q,2)+1)
    alpha = ab + s4*eta*pow(ab,2) + s5*pow(eta,2)*ab + t0*eta*ab + 2*np.sqrt(3)*eta + t2*pow(eta,2) + t3*pow(eta,3)
    Q = 2.0*pow(1.0 - alpha, -0.45) 
    tau = 2.0*Q*Mf/(1.0-0.63*pow(1.0-alpha, 0.3))
    return tau

def Final_Mass(alpha1, alpha2, nu):
    p0 = 0.04826; p1 = 0.01559; p2 = 0.00485; s4 = -0.1229; s5 = 0.4537; 
    t0 = -2.8904; t2 = -3.5171; t3 = 2.5763; q = 1.0; eta = nu; theta = np.pi/2; 
    Mf = 1-p0 - p1*(alpha1+alpha2)-p2*pow(alpha1+alpha2,2)
    return Mf

def Final_Spin(alpha1, alpha2, nu):
    p0 = 0.04826; p1 = 0.01559; p2 = 0.00485; s4 = -0.1229; s5 = 0.4537; 
    t0 = -2.8904; t2 = -3.5171; t3 = 2.5763; q = 1.0; eta = nu; theta = np.pi/2; 
    Mf = 1-p0 - p1*(alpha1+alpha2)-p2*pow(alpha1+alpha2,2)
    ab = (pow(q,2)*alpha1+alpha2)/(pow(q,2)+1)
    alpha = ab + s4*eta*pow(ab,2) + s5*pow(eta,2)*ab + t0*eta*ab + 2*np.sqrt(3)*eta + t2*pow(eta,2) + t3*pow(eta,3)
    return alpha

def A_peak(A0,tau, t0, tp):
    Ap = A0*np.cosh((t0-tp)/tau)
    return Ap

# Ps2 from BOB
def Omega_BOB(omega0, tau, t, t0, tp):
    omqnm = Omega_QNM(0.0, 0.0, 0.25)
    k = (pow(omqnm,4)-pow(omega0,4))/(1- np.tanh((t0-tp)/tau))
    om = (pow(omega0,4) + k*(np.tanh((t-tp)/tau) - np.tanh((t0-tp)/tau)))**(1.0/4)
    return om

def Tp(t0, tau, omega0, omqnm, omega0_dot):
    tp = t0 + (tau/2.0)*np.log(((pow(omqnm,4)-pow(omega0,4))/(2.0*tau*pow(omega0,3)*omega0_dot))-1.0)
    return tp

def Amplitude_h_BOB(omega0, Ap, tau, t, tp):
    Amp_psi4 = Ap/(np.cosh((t-tp)/tau))
    om_bob = Omega_BOB(omega0, tau, t, t0, tp)
    Amp_h = Amp_psi4/(4.0*om_bob*om_bob)
    return Amp_h

def Abs_sigma_dot_BOB(omega0, Ap, tau, t, tp):
    Amp_psi4 = Ap/(np.cosh((t-tp)/tau))
    om_bob = Omega_BOB(omega0, tau, t, t0, tp)
    Abs_sigma_dot = Amp_psi4/(2.0*om_bob)*m2Y22(np.pi/2, np.pi/2.0)
    return Abs_sigma_dot

def Intg_SigmaDot_sqr_BOB(t_vec, Abs_sigma_dot):
    sigma_dot_sqr = Abs_sigma_dot*Abs_sigma_dot
    dt_vec = np.diff(t_vec)
    dt_vec = np.append(dt_vec, dt_vec[-1])
    Int_sigma_dot_sqr = -np.cumsum(abs(sigma_dot_sqr)*dt_vec)
    Int_sigma_dot_sqr = Int_sigma_dot_sqr #- Int_sigma_dot_sqr[0]
    return Int_sigma_dot_sqr

def Binding_Energy_BOB(t_vec, r, Mf, af, p):
    E = (p/pow(r, 3.0/4.0))*(pow(r, 3.0/2.0) - 2.0*Mf*pow(r, 1.0/2.0) + pow(Mf, 1.0/2.0)*af)/np.sqrt(pow(r, 3.0/2.0) - 3.0*Mf*pow(r, 1.0/2.0) + 2*pow(Mf, 1.0/2.0)*af) 
    return -np.sqrt(2)*E

#def ISCO_radius()

#######################################################################################################################
##### Initial setup for binary very far to get rid of oscilation due to starting cirular orbit at close radai ########
r_ini = 50.0                                                                                    # Initial radius for circular orbit
p=0.25; ng_radial =1000; t0 = -50.0; tp =0.0; tau=1.0; Ap=1.0; r_switch=0; rad_reac=0;           # Parameter nu, ng_radial, t0, tp, tau, Ap
M=1; omega0=0                                                                                             # Total mass 
w0 = r_ini, 0.0, 0, Circular_orbit_ini_P_phi(r_ini, p)                                          # Input paramter of solving ODE: w0 = r0, p_r0, phi0, p_phi0 
rf = 2                                                                                          # Radius where quantities in balance law start being evaluated                    

t_vec = time_vec(r_ini, rf, 3000, 1000)[0]                                                      # Create time vector with variable rosultion 

param_values =  p, t0, tp, tau, Ap, r_switch, rad_reac, omega0

## Solve ODEs to get intial condition at some close radai ###########################################################
yvec = ODEs.Coupled_HamiltonianODEs_solver(w0, t_vec, param_values)

r, p_r, phi, p_phi = yvec[:,0], yvec[:,1], yvec[:,2], yvec[:,3]


#####################################################################################################################
##### Make plots for r0=15 to rf=3M #################################################################################
r0=15; rf=2.01
idx_r0 = find_nearest1(r,r0)
idx_rf = find_nearest1(r,rf)
t_vec = t_vec[idx_r0:idx_rf]; r = r[idx_r0:idx_rf]; p_r = p_r[idx_r0:idx_rf]; phi = phi[idx_r0:idx_rf]; p_phi = p_phi[idx_r0:idx_rf]
t_vec = t_vec-t_vec[-1]

# Turn on radiation reaction from BOB at ISCO
r_isco = 6.0; alpha1=0.0; alpha2=0.0; p=0.25; 

idx_isco = find_nearest1(r,r_isco)
omega = ODEs.dphi_by_dt(r, p_r, phi, p_phi, p)
psi22_eob = abs(ODEs.Rh22(r, p_r, phi, p_phi, p))*4.0*omega*omega 

tau = Tau_v2(alpha1, alpha2, p); t0 = t_vec[idx_isco]; A0=psi22_eob[idx_isco]
omega0 = omega[idx_isco]; omqnm = Omega_QNM(alpha1, alpha2, p)

omega0_dot = (omega[idx_isco+1] - omega[idx_isco])/(t_vec[idx_isco+1]- t_vec[idx_isco])

tp = Tp(t0, tau, omega0, omqnm, omega0_dot); Ap=A_peak(A0,tau, t0, tp)
r_switch=r_isco; rad_reac=1
t_vec_BOB=t_vec[idx_isco:]

param_values_BOB = p, t0, tp, tau, Ap, r_switch, rad_reac, omega0

# start solving equations of motion at ISCO using BOB radiation reaction force

w_isco = r_isco, p_r[idx_isco], phi[idx_isco], p_phi[idx_isco]
yvec_BOB = ODEs.Coupled_HamiltonianODEs_solver(w_isco, t_vec_BOB, param_values_BOB)
r_BOB, p_r_BOB, phi_BOB, p_phi_BOB = yvec_BOB[:,0], yvec_BOB[:,1], yvec_BOB[:,2], yvec_BOB[:,3]

# Plot particle trajactory 
plt.plot(r*np.cos(phi), r*np.sin(phi))
plt.plot(r_BOB*np.cos(phi_BOB), r_BOB*np.sin(phi_BOB), 'r--')
plt.show()

# Plot h22 apmlitude
Amp_BOB = Amplitude_h_BOB(omega0, Ap, tau, t_vec_BOB, tp)
plt.plot(t_vec_BOB, Amp_BOB, 'r--')
plt.plot(t_vec,abs(ODEs.Rh22(r, p_r, phi, p_phi, p)))
plt.show()

'''
#Make plots to compare h22 mode for EOB and BOB
h22_BOB = Amplitude_h_BOB(omega0, Ap, tau, t_vec_BOB, tp)*np.cos(2.0*Omega_BOB(omega0, tau, t_vec_BOB, t0, tp))
plt.plot(t_vec_BOB, h22_BOB, 'r--')
plt.plot(t_vec, ODEs.Rh22(r, p_r, phi, p_phi, p).real)
plt.show()
'''


#flip arrays
t_vec_BOB = np.flip(t_vec_BOB)
r_BOB = np.flip(r_BOB)
p_r_BOB = np.flip(p_r_BOB)
phi_BOB = np.flip(phi_BOB)
p_phi_BOB = np.flip(p_phi_BOB)
Mf = Final_Mass(alpha1, alpha2, p)
af = Final_Spin(alpha1, alpha2, p)

# Plot integral Balance law
r_beyond_isco =r[idx_isco:]; p_r_beyond_isco = p_r[idx_isco:]; phi_beyond_isco=phi[idx_isco:]; p_phi_beyond_isco=p_phi[idx_isco:]
#flip arrays
t_vec_BOB = np.flip(t_vec_BOB)
r_beyond_isco = np.flip(r_beyond_isco)
p_r_beyond_isco = np.flip(p_r_beyond_isco)
phi_beyond_isco = np.flip(phi_beyond_isco)
p_phi_beyond_isco = np.flip(p_phi_beyond_isco)


PsiRRResum_Pr = Psi_Growth(r_beyond_isco, p_r_beyond_isco, phi_beyond_isco, p_phi_beyond_isco, p)
PsiRRResum_Pr= PsiRRResum_Pr-PsiRRResum_Pr[0]

#comoute psi BOB
Abs_sigma_dot_BOB = Abs_sigma_dot_BOB(omega0, Ap, tau, t_vec_BOB, tp)
PsiBOB_Pr =Binding_Energy_BOB(t_vec_BOB, r_BOB, Mf, af, p)
PsiBOB_Pr = PsiBOB_Pr - PsiBOB_Pr[0]
t_vec_BOB=t_vec_BOB-t_vec_BOB[0]

omega_bob =  Omega_BOB(omega0, tau, t_vec_BOB, t0, tp)
omega_bob = omega_bob - omega_bob[0]

Intg_SigmaDot_sqr_EOB = Intg_SigmaDot_sqr(t_vec_BOB, r_beyond_isco, p_r_beyond_isco, phi_beyond_isco, p_phi_beyond_isco, p)
Intg_SigmaDot_sqr_EOB = Intg_SigmaDot_sqr_EOB - Intg_SigmaDot_sqr_EOB[0]

Intg_SigmaDot_sqr_BOB = Intg_SigmaDot_sqr_BOB(t_vec_BOB, Abs_sigma_dot_BOB) 
Intg_SigmaDot_sqr_BOB = Intg_SigmaDot_sqr_BOB - Intg_SigmaDot_sqr_BOB[0]

plt.plot(t_vec_BOB, Intg_SigmaDot_sqr_EOB,'r', label = r'$\Psi = -\int_{u_1}^{u_2}du|\dot{\sigma^{0}}_{Rsum}|^2$')
plt.plot(t_vec_BOB, PsiRRResum_Pr,'r--', label = r'$\Psi = -M_{RRsum}$')
plt.plot(t_vec_BOB, Intg_SigmaDot_sqr_BOB,'c', label = r'$\Psi = -\int_{u_1}^{u_2}du|\dot{\sigma^{0}}_{BOB}|^2$')
plt.plot(t_vec_BOB, PsiBOB_Pr-omega_bob ,'c--', label = r'$\Psi = -M_{BOB}$')
plt.xlabel(r'$time$')
plt.ylabel(r'$\Psi$')
plt.legend()
#plt.savefig('../plots/Psi2EOBHamiltonia_Comparision_usingResumModel.pdf')
plt.show()

plt.plot(t_vec_BOB, abs(Intg_SigmaDot_sqr_EOB-PsiRRResum_Pr),'r', label = r'$\Psi = -\int_{u_1}^{u_2}du|\dot{\sigma^{0}}_{Rsum}|^2 - M_{RRsum}$')
plt.plot(t_vec_BOB, abs(Intg_SigmaDot_sqr_BOB-PsiBOB_Pr+omega_bob),'c', label = r'$\Psi = -\int_{u_1}^{u_2}du|\dot{\sigma^{0}}_{BOB}|^2 - M_{BOB}$')
plt.xlabel(r'$time$')
plt.ylabel(r'$\Psi$')
plt.legend()
#plt.savefig('../plots/Psi2EOBHamiltonia_Comparision_usingResumModel.pdf')
plt.show()


####################################################################################################
###### Using first principle approach for final Mass and Spin #####################################
print('Mf=',Mf)
H_isco = ODEs.Heff(r[idx_isco], p_r[idx_isco], phi[idx_isco], p_phi[idx_isco], p)
Mf = (1+ p - p*H_isco)

print('Mf=',Mf)
