import sys
sys.path.insert(0, '/home/aschoudhary/constraintongwwaveform/src')

from scipy.integrate import odeint 
import numpy as np 
from numpy.core.umath_tests import inner1d
import numpy as np
import HamiltonianCoupledODE_BOB as ODEs
import matplotlib.pyplot as plt
import PN_radiationReaction as PN_RR
import scipy.optimize as opt

# create a time vector with variable resulution
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


# Psi2 due to presence of particle
def Psi2p(r, nu, M):
    return -nu*M*(1.0 - 2*M/r)/np.sqrt(1-3*M/r)

# Psi due to back ground mass evolving (essentially M_ADM = H_Real)
def Psi2_0_Growth(r, p_r, phi, p_phi, p):
    HEOB = ODEs.Heob(r, p_r, phi, p_phi, p)
    HEOB = HEOB-HEOB[0]
    M = HEOB*p
    M = M - M[0]
    return M

# Radial momentum of the perturber
def Psi2_0_Growth_Pr(r, p_r, phi, p_phi, p):
    HEOB = ODEs.Heob(r, p_r, phi, p_phi, p)
    HEOB = HEOB-HEOB[0]
    M = HEOB*p
    M = M - M[0]
    Pr = (6.0/2/np.sqrt(2))*p_r*p
    Pr = Pr-Pr[0]
    a = ODEs.A(r, p)
    b = ODEs.B(r, p)
    
    return  M + np.sqrt(b/a)*Pr 

# left hand side of the balance equation Intgral(Sigma_dot_sqr*dt)
def Psi2_sqr_fromWaveform(t_vec, r, p_r, phi, p_phi, p):
    dt = t_vec[1]-t_vec[0]
    abs_Sigma_Dot_sqr = (2*ODEs.dphi_by_dt(r, p_r, phi, p_phi, p)*abs(ODEs.Rh22(r, p_r, phi, p_phi, p)))**2
    abs_Sigma_Dot_sqr = abs_Sigma_Dot_sqr - abs_Sigma_Dot_sqr[0]

    t_vec2 = time_vec(r0, 3, 1500, 1000)[1]
    t_vec2 = t_vec2[:len(t_vec)]  

    Int_abs_Sigma_Dot_sqr = -np.cumsum(abs_Sigma_Dot_sqr*t_vec2)*abs(m2Y22(np.pi/2, np.pi/2))**2
    Psi2_diff = Int_abs_Sigma_Dot_sqr
    return Psi2_diff

# Compute quasinormal frrequencies given initial spins and symmetric mass ratio
def Omega_QNM(alpha1, alpha2, nu):
    p0 = 0.04826; p1 = 0.01559; p2 = 0.00485; s4 = -0.1229; s5 = 0.4537; 
    t0 = -2.8904; t2 = -3.5171; t3 = 2.5763; q = 1.0; eta = nu; theta = np.pi/2; 
    Mf = 1-p0 - p1*(alpha1+alpha2)-p2*pow(alpha1+alpha2,2)
    ab = (pow(q,2)*alpha1+alpha2)/(pow(q,2)+1)
    alpha = ab + s4*eta*pow(ab,2) + s5*pow(eta,2)*ab + t0*eta*ab + 2*np.sqrt(3)*eta + t2*pow(eta,2) + t3*pow(eta,3)
    OM_QNM = (1.0 - 0.63*pow(1.0 - alpha, 0.3))/(2*Mf)
    return OM_QNM

def Tau(t0, tp, OMqnm, OM0, OM0_dot, tau_min, tau_max):
    tau = opt.brentq(lambda x: tp-t0-(x/2.0)*np.log((pow(OMqnm,4)-pow(OM0,4))/(2*x*pow(OM0,3)*OM0_dot)-1.0), tau_min, tau_max)
    return tau




##### Initial setup for binary very far to get rid of oscilation due to starting cirular orbit at close radai ########
r_ini = 50.0                                                                    # Initial radius for circular orbit
p=0.25; ng_radial =100; t0 = -10.0; tp =0.0; tau=1.0; Ap=1.0; r_switch=0        # Parameter nu, ng_radial, t0, tp, tau, Ap
M=1                                                                             # Total mass
w0 = r_ini, 0.0, 0, Circular_orbit_ini_P_phi(r_ini, p)                          # Input paramter of solving ODE
r0 = 15                                                                         # Radius where quantities in balance law start being evaluated                    

t_vec = time_vec(r_ini, r0, 1500, 100)[0]

param_values = p, ng_radial, t0, tp, tau, Ap, r_switch

## Solve ODEs to get intial condition at some close radai ###########################################################
yvec = ODEs.Coupled_HamiltonianODEs_solver(w0, t_vec, param_values)

r, p_r, phi, p_phi = yvec[:,0], yvec[:,1], yvec[:,2], yvec[:,3]
idx_r0 = find_nearest1(r,r0)

#####################################################################################################################



## Initial conditions ##
r0, p_r0, phi0, p_phi0 = r[idx_r0], p_r[idx_r0], phi[idx_r0], p_phi[idx_r0]

p=0.25 # parameter nu
M=1

w0 = r0, p_r0, phi0, p_phi0 


t_vec = time_vec(r0, 3, 1500, 1000)[0]


## Solve ODEs
yvec = ODEs.Coupled_HamiltonianODEs_solver(w0, t_vec, param_values)

r, p_r, phi, p_phi = yvec[:,0], yvec[:,1], yvec[:,2], yvec[:,3]


idx = find_nearest1(r,3)
## stop at r=3M
t_vec, r, p_r, phi, p_phi = t_vec[:idx], r[:idx], p_r[:idx], phi[:idx], p_phi[:idx] 
t_vec = t_vec - t_vec[-1]
Omega = ODEs.dphi_by_dt(r, p_r, phi, p_phi, p)

H_EOB = ODEs.Heob(r, p_r, phi, p_phi, p)

ti = t_vec[find_nearest1(r,r0)]
tf = t_vec[find_nearest1(r,3)]

plt.plot(t_vec, Psi2_0_Growth(r, p_r, phi, p_phi, p), label = r'$\Psi = M_{B}$')
plt.plot(t_vec, Psi2_0_Growth_Pr(r, p_r, phi, p_phi, p), label = r'$\Psi = M_{B} + \frac{6}{2 \sqrt{2}}P_{r}$')
plt.plot(t_vec, Psi2_sqr_fromWaveform(t_vec, r, p_r, phi, p_phi, p), label = r'$\Psi = \int_{u_1}^{u_2}du|\dot{\sigma^{0}}|^2$')
#plt.xlim(ti, tf)
plt.ylim(-0.06, 0.01)
plt.xlabel(r'$time$')
plt.ylabel(r'$\Psi$')
plt.legend()
#plt.savefig('/home/aschoudhary/constraintongwwaveform/plots/Psi2EOBHamiltonia_BOB.pdf')
plt.show()

###################################################################################################################
################ How purterber trajactory looks without radiation reaction beyond ISCO ############################
idx_isco = find_nearest1(r,6)
r0_isco, p_r0_isco, phi0_isco, p_phi0_isco = r[idx_isco], p_r[idx_isco], phi[idx_isco], p_phi[idx_isco]
w0_isco = r0_isco, p_r0_isco, phi0_isco, p_phi0_isco 
p=0.25; ng_radial =100; t0 = -10.0; tp =0.0; tau=1.0; Ap=1.0; r_switch_isco=6 

t_vec_isco = time_vec(r0_isco, 3, 1500, 1000)[0]
param_values_isco = p, ng_radial, t0, tp, tau, Ap, r_switch_isco

yvec = ODEs.Coupled_HamiltonianODEs_solver(w0_isco, t_vec_isco, param_values_isco)

r_isco, p_r_isco, phi_isco, p_phi_isco = yvec[:,0], yvec[:,1], yvec[:,2], yvec[:,3]





#######################################################################################################



################ Start using BOB radiation flux from ISCO onwards ########################
#For Swarzchild case ISCO at r = 6M

idx_isco = find_nearest1(r,6)

'''
OM_ISCO = Omega[idx_isco]
OM_qnm = Omega_QNM(0.0, 0.0, 0.25)
OM_dot_ISCO = (Omega[idx_isco+1] - Omega[idx_isco])/(t_vec[idx_isco+1] - t_vec[idx_isco])

# Damping time scale tau for BOB
tau = Tau(t_vec[idx_isco], 0, OM_qnm , OM_ISCO, OM_dot_ISCO, 2, 50)

### Start solving equation using BOB radiation reaction ##########
p=0.25; ng_radial =100; t0 = t_vec[idx_isco]; tp =0.0; Ap=1.0; r_switch= 6.0  
## Initial conditions ##
r0_BOB, p_r0_BOB, phi0_BOB, p_phi0_BOB = r[idx_isco], p_r[idx_isco], phi[idx_isco], p_phi[idx_isco]
w0_BOB = r0_BOB, p_r0_BOB, phi0_BOB, p_phi0_BOB 
t_vec_BOB = time_vec(r0_BOB, 3, 1500, 1000)[0]

#parameter values
param_values = p, ng_radial, t0, tp, tau, Ap, r_switch
yvec = ODEs.Coupled_HamiltonianODEs_solver(w0_BOB, t_vec_BOB, param_values)
'''


plt.figure()
plt.plot(r*np.cos(phi), r*np.sin(phi), label='$r_{0}=$'+str(r0))
plt.plot(r[idx_isco:]*np.cos(phi[idx_isco:]), r[idx_isco:]*np.sin(phi[idx_isco:]), 'r')
plt.plot(r_isco*np.cos(phi_isco), r_isco*np.sin(phi_isco), 'k--')
plt.xlim(-1.5*r0, 1.5*r0)
plt.ylim(-1.5*r0, 1.5*r0)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.legend()
#plt.savefig('/home/aschoudhary/constraintongwwaveform/plots/OrbitEOB'+str(r0)+'.pdf')
plt.show()

plt.figure()
h2 = ODEs.Rh22(r, p_r, phi, p_phi, p)
plt.ylim(-1.0, 1.0)
plt.xlabel(r'$time$')
plt.ylabel(r'$h_{22}$')
plt.plot(t_vec-t_vec[-1], h2.real)
#plt.plot(t_vec[idx_isco:]-t_vec[-1], h2.real[idx_isco:], 'r')
plt.legend()
#plt.savefig('/home/aschoudhary/constraintongwwaveform/plots/h22EOB.pdf')
plt.show()


