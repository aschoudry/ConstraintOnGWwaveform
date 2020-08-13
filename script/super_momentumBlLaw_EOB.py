import sys
sys.path.insert(0, '../src')

from scipy.integrate import odeint 
import numpy as np 
from numpy.core.umath_tests import inner1d
import numpy as np
import HamiltonianCoupledODE_BOB as ODEs
import matplotlib.pyplot as plt
import PN_radiationReaction as PN_RR
import scipy.optimize as opt

# create a time vector with variable resolution so that time step are smaller are purterber inspirals in. 
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
    HEOB = HEOB-HEOB[0]
    HEFF = ODEs.Heff(r, p_r, phi, p_phi, p)
    HEFF = HEFF - HEFF[0]
    M = HEOB*p + HEFF*p
    return M

# Psi due to back ground mass evolving (essentially Psi = M_ADM + Pr = H_Real + Pr), where Pr is the radial linenar momentum of purterber
# Equation 1.9 in notes 
def Psi_Growth_Pr(r, p_r, phi, p_phi, p):
    HEOB = ODEs.Heob(r, p_r, phi, p_phi, p)
    HEOB = HEOB-HEOB[0]
    HEFF = ODEs.Heff(r, p_r, phi, p_phi, p)
    HEFF = HEFF - HEFF[0]
    M = HEOB*p + HEFF*p
    Pr = p_r*p
    Pr = Pr-Pr[0]
    a = ODEs.A(r, p)
    b = ODEs.B(r, p)
    return M + (6.0/(2.0))*np.sqrt(b/a)*Pr 

# left hand side of the balance equation Intgral(Sigma_dot_sqr*dt)
def Intg_SigmaDot_sqr(t_vec, r, p_r, phi, p_phi, p):
    abs_Sigma_Dot_sqr = (2*ODEs.dphi_by_dt(r, p_r, phi, p_phi, p)*abs(ODEs.Rh22(r, p_r, phi, p_phi, p)))**2
    abs_Sigma_Dot_sqr = abs_Sigma_Dot_sqr
    dt_vec = np.diff(t_vec)
    dt_vec = np.append(dt_vec, dt_vec[-1])
    Int_abs_Sigma_Dot_sqr = -np.cumsum(abs_Sigma_Dot_sqr*dt_vec)*abs(m2Y22(np.pi/2, np.pi/2.0))**2
    Psi = Int_abs_Sigma_Dot_sqr - Int_abs_Sigma_Dot_sqr[0]
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
def A_peak(alpha1, alpha2, nu):
    p0 = 0.04826; p1 = 0.01559; p2 = 0.00485; s4 = -0.1229; s5 = 0.4537;  
    Mf = 1-p0 - p1*(alpha1+alpha2)-p2*pow(alpha1+alpha2,2)
    ap = 1.068*pow(1-Mf, 0.8918)
    return ap

# Ps2 from BOB
def Omega_BOB(omega0, tau, t, t0, tp):
    omqnm = Omega_QNM(0.0, 0.0, 0.25)
    k = (pow(omqnm,4)-pow(omega0,4))/(1- np.tanh((t0-tp)/tau))
    om = (pow(omega0,4) + k*(np.tanh((t-tp)/tau) - np.tanh((t0-tp)/tau) ))**(1.0/4)
    return om

def Intg_SigmaDot_sqr_BOB(Ap, tau, t_vec, t0, tp):
    omega0 = 0.0680414
    omega = Omega_BOB(omega0, tau, t_vec, t0, tp)
    abs_psi4=abs(Ap/np.cosh((t_vec-tp)/tau))
    h22dot = abs_psi4/(2*omega)
    sigma_dot_sqr=h22dot*h22dot*abs(m2Y22(np.pi/2, np.pi/2.0))**2

    dt_vec = np.diff(t_vec)
    dt_vec = np.append(dt_vec, dt_vec[-1])
    Int_sigma_dot_sqr = -np.cumsum(abs(sigma_dot_sqr)*dt_vec)
    Int_sigma_dot_sqr = Int_sigma_dot_sqr - Int_sigma_dot_sqr[0]
    return Int_sigma_dot_sqr 
#######################################################################################################################
##### Initial setup for binary very far to get rid of oscilation due to starting cirular orbit at close radai ########
r_ini = 50.0                                                                                    # Initial radius for circular orbit
p=0.25; ng_radial =1000; t0 = -50.0; tp =0.0; tau=1.0; Ap=1.0; r_switch=0; rad_reac=0           # Parameter nu, ng_radial, t0, tp, tau, Ap
M=1                                                                                             # Total mass 
w0 = r_ini, 0.0, 0, Circular_orbit_ini_P_phi(r_ini, p)                                          # Input paramter of solving ODE: w0 = r0, p_r0, phi0, p_phi0 
rf = 3                                                                                          # Radius where quantities in balance law start being evaluated                    

t_vec = time_vec(r_ini, rf, 2000, 1000)[0]                                                      # Create time vector with variable rosultion 

param_values = p, ng_radial, t0, tp, tau, Ap, r_switch, rad_reac

## Solve ODEs to get intial condition at some close radai ###########################################################
yvec = ODEs.Coupled_HamiltonianODEs_solver(w0, t_vec, param_values)

r, p_r, phi, p_phi = yvec[:,0], yvec[:,1], yvec[:,2], yvec[:,3]


#####################################################################################################################
##### Make plots for r0=15 to rf=3M #################################################################################
r0=15; rf=3
idx_r0 = find_nearest1(r,r0)
idx_rf = find_nearest1(r,rf)
t_vec = t_vec[idx_r0:idx_rf]; r = r[idx_r0:idx_rf]; p_r = p_r[idx_r0:idx_rf]; phi = phi[idx_r0:idx_rf]; p_phi = p_phi[idx_r0:idx_rf]
t_vec = t_vec-t_vec[-1]

# Turn on radiation reaction from BOB at ISCO
r_isco = 6.0; alpha1=0.0; alpha2=0.0; p=0.25; 
idx_isco = find_nearest1(r,r_isco)
ng_radial =1000; t0 = -36.78; tp =0.0; tau=Tau_v2(alpha1, alpha2, p); Ap=A_peak(alpha1, alpha2, p); r_switch=r_isco; rad_reac=1;
t_vec_BOB=t_vec[idx_isco:]
param_values_BOB = p, ng_radial, t0, tp, tau, Ap, r_switch, rad_reac

# start solving equations of motion at ISCO using BOB radiation reaction force

w_isco = r_isco, p_r[idx_isco], phi[idx_isco], p_phi[idx_isco]
yvec_BOB = ODEs.Coupled_HamiltonianODEs_solver(w_isco, t_vec_BOB, param_values_BOB)
r_BOB, p_r_BOB, phi_BOB, p_phi_BOB = yvec_BOB[:,0], yvec_BOB[:,1], yvec_BOB[:,2], yvec_BOB[:,3]

PsiBOB_Pr = Psi_Growth_Pr(r_BOB, p_r_BOB, phi_BOB, p_phi_BOB, p)

# Plot particle trajactory 
plt.plot(r*np.cos(phi), r*np.sin(phi))
plt.plot(r_BOB*np.cos(phi_BOB), r_BOB*np.sin(phi_BOB), 'r--')
plt.show()

# Plot integral Balance law
r_beyond_isco =r[idx_isco:]; p_r_beyond_isco = p_r[idx_isco:]; phi_beyond_isco=phi[idx_isco:]; p_phi_beyond_isco=p_phi[idx_isco:]
PsiRRResum_Pr = Psi_Growth_Pr(r_beyond_isco, p_r_beyond_isco, phi_beyond_isco, p_phi_beyond_isco, p)


plt.plot(t_vec_BOB, Intg_SigmaDot_sqr(t_vec_BOB, r_beyond_isco, p_r_beyond_isco, phi_beyond_isco, p_phi_beyond_isco, p),'r', label = r'$\Psi = -\int_{u_1}^{u_2}du|\dot{\sigma^{0}}_{Rsum}|^2$')
plt.plot(t_vec_BOB, PsiRRResum_Pr,'r--', label = r'$\Psi = -M_{RRsum}$')
plt.plot(t_vec_BOB, Intg_SigmaDot_sqr_BOB(Ap, tau, t_vec_BOB, t0, tp),'c', label = r'$\Psi = -\int_{u_1}^{u_2}du|\dot{\sigma^{0}}_{BOB}|^2$')
plt.plot(t_vec_BOB, PsiBOB_Pr,'c--', label = r'$\Psi = -M_{BOB}$')
plt.xlabel(r'$time$')
plt.ylabel(r'$\Psi$')
plt.legend()
plt.savefig('../plots/Psi2EOBHamiltonia_Comparision_usingResumModel.pdf')
plt.show()

plt.semilogy(t_vec_BOB, -Intg_SigmaDot_sqr(t_vec_BOB, r_beyond_isco, p_r_beyond_isco, phi_beyond_isco, p_phi_beyond_isco, p),'r', label = r'$\Psi = -\int_{u_1}^{u_2}du|\dot{\sigma^{0}}_{Rsum}|^2$')
plt.semilogy(t_vec_BOB, -PsiRRResum_Pr,'r--', label = r'$\Psi = -M_{RRsum}$')
plt.semilogy(t_vec_BOB, -Intg_SigmaDot_sqr_BOB(Ap, tau, t_vec_BOB, t0, tp),'c', label = r'$\Psi = -\int_{u_1}^{u_2}du|\dot{\sigma^{0}}_{BOB}|^2$')
plt.semilogy(t_vec_BOB, -PsiBOB_Pr,'c--', label = r'$\Psi = -M_{BOB}$')
plt.xlabel(r'$time$')
plt.ylabel(r'$\Psi$')
plt.legend()
plt.show()

print(tau)
