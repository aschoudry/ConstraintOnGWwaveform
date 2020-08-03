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

# Ps2 from BOB
def Omega_BOB(omega0, tau, t, t0, tp):
    omqnm = Omega_QNM(0.0, 0.0, 0.25)
    k = (pow(omqnm,4)-pow(omega0,4))/(1- np.tanh((t0-tp)/tau))
    om = (pow(omega0,4) + k*(np.tanh((t-tp)/tau) - np.tanh((t0-tp)/tau) ))**(1.0/4)
    return om
    

def Psi2_sqr_fromWaveform_BOB(omega0, Ap, tau, t, t0, tp):
    omega = Omega_BOB(omega0, tau, t, t0, tp)
    sigma_dd_sqr = Ap/(np.cosh((t-tp)/tau))
    sigma_dot_sqr = pow(sigma_dd_sqr,2)/pow(2*omega,2)
    sigma_dot_sqr = sigma_dot_sqr - sigma_dot_sqr[0]
    return -np.cumsum(sigma_dot_sqr)*(t[1]-t[0])





##### Initial setup for binary very far to get rid of oscilation due to starting cirular orbit at close radai ########
r_ini = 50.0                                                                                  # Initial radius for circular orbit
p=0.25; ng_radial =100; t0 = -10.0; tp =0.0; tau=1.0; Ap=1.0; r_switch=0; rad_reac=0          # Parameter nu, ng_radial, t0, tp, tau, Ap
M=1                                                                                           # Total mass
w0 = r_ini, 0.0, 0, Circular_orbit_ini_P_phi(r_ini, p)                                        # Input paramter of solving ODE
r0 = 15                                                                                       # Radius where quantities in balance law start being evaluated                    

t_vec = time_vec(r_ini, r0, 1500, 100)[0]

param_values = p, ng_radial, t0, tp, tau, Ap, r_switch, rad_reac

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

plt.plot(t_vec, Psi2_0_Growth(r, p_r, phi, p_phi, p), 'r',label = r'$\Psi = M_{B}$')
plt.plot(t_vec + 28.37, Psi2_0_Growth_Pr(r, p_r, phi, p_phi, p),'c', label = r'$\Psi = M_{B} + \frac{6}{2 \sqrt{2}}P_{r}$')  #Slight shifted the plots such that they peak at t=0
plt.plot(t_vec, Psi2_sqr_fromWaveform(t_vec, r, p_r, phi, p_phi, p),'k--', label = r'$\Psi = -\int_{u_1}^{u_2}du|\dot{\sigma^{0}}|^2$')
#plt.xlim(ti, tf)
plt.ylim(-0.06, 0.01)
plt.xlabel(r'$time$')
plt.ylabel(r'$\Psi$')
plt.legend()
plt.savefig('/home/aschoudhary/constraintongwwaveform/plots/Psi2EOBHamiltonia_BOB.pdf')
plt.show()

###################################################################################################################
################ How purterber trajactory looks without radiation reaction beyond ISCO ############################
idx_isco = find_nearest1(r,6)
r0_isco, p_r0_isco, phi0_isco, p_phi0_isco = r[idx_isco], p_r[idx_isco], phi[idx_isco], p_phi[idx_isco]
w0_isco = r0_isco, p_r0_isco, phi0_isco, p_phi0_isco 
p=0.25; ng_radial =100; t0 = -10.0; tp =0.0; tau=1.0; Ap=1.0; r_switch_isco=6; rad_reac_isco=0

t_vec_isco = time_vec(r0_isco, 3, 1500, 1000)[0]
param_values_isco = p, ng_radial, t0, tp, tau, Ap, r_switch_isco, rad_reac_isco

yvec = ODEs.Coupled_HamiltonianODEs_solver(w0_isco, t_vec_isco, param_values_isco)

r_isco, p_r_isco, phi_isco, p_phi_isco = yvec[:,0], yvec[:,1], yvec[:,2], yvec[:,3]

#######################################################################################################

################ Start using BOB radiation flux from ISCO onwards ########################
#For Swarzchild case ISCO at r = 6M

idx_isco = find_nearest1(r,6)

OM_ISCO = Omega[idx_isco]
OM_qnm = Omega_QNM(0.0, 0.0, 0.25)
OM_dot_ISCO = (Omega[idx_isco+1] - Omega[idx_isco])/(t_vec[idx_isco+1] - t_vec[idx_isco])

print('OM isco = ',OM_ISCO)

plt.figure()
plt.plot(r*np.cos(phi), r*np.sin(phi), label='$r_{0}=$'+str(round(r0)))
plt.plot(r[idx_isco:]*np.cos(phi[idx_isco:]), r[idx_isco:]*np.sin(phi[idx_isco:]), 'r')
plt.plot(r_isco*np.cos(phi_isco), r_isco*np.sin(phi_isco), 'k--', label='No RR beyond ISCO')
plt.xlim(-1.5*r0, 1.5*r0)
plt.ylim(-1.5*r0, 1.5*r0)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.legend()
plt.savefig('/home/aschoudhary/constraintongwwaveform/plots/OrbitEOB_rr'+str(round(r0))+'.pdf')
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

################### Make plot for r=6M to r=3M ###############################################
idx_ini = find_nearest1(r,6)
idx_fin = find_nearest1(r,3)

# Damping time scale tau for BOB
tau = Tau(t_vec[idx_isco], 0, OM_qnm , OM_ISCO, OM_dot_ISCO, 2, 50)

### Start solving equation using BOB radiation reaction ##########
p=0.25; ng_radial =1000; t0 = -37.14466605358939; tp =0.0; Ap=4.0*pow(Omega_BOB(OM_ISCO, tau, tp, t0, tp),2); r_switch_isco=6; rad_reac_isco=1
## Initial conditions ##
r0_BOB, p_r0_BOB, phi0_BOB, p_phi0_BOB = r[idx_isco], p_r[idx_isco], phi[idx_isco], p_phi[idx_isco]
w0_BOB = r0_BOB, p_r0_BOB, phi0_BOB, p_phi0_BOB 
t_vec_BOB = t_vec[idx_isco:idx_fin]

print(t_vec[idx_isco], t_vec[idx_fin])
print('OMp = ',Omega_BOB(OM_ISCO, tau, tp, t0, tp) )
param_values = p, ng_radial, t0, tp, tau, Ap, r_switch_isco, rad_reac_isco
yvec_BOB = ODEs.Coupled_HamiltonianODEs_solver(w0_BOB, t_vec_BOB, param_values)

r_BOB, p_r_BOB, phi_BOB, p_phi_BOB = yvec_BOB[:,0], yvec_BOB[:,1], yvec_BOB[:,2], yvec_BOB[:,3]

plt.figure()
plt.plot(r*np.cos(phi), r*np.sin(phi), label='$r_{0}=$'+str(round(r0)))
plt.plot(r_isco*np.cos(phi_isco), r_isco*np.sin(phi_isco), 'k--', label='No RR beyond ISCO')
plt.plot(r[idx_isco:]*np.cos(phi[idx_isco:]), r[idx_isco:]*np.sin(phi[idx_isco:]), 'r', label='Resummed RR beyond ISCO')
plt.plot(r_BOB*np.cos(phi_BOB), r_BOB*np.sin(phi_BOB), 'g--', label='BOB beyond ISCO')
plt.xlim(-1.5*r0, 1.5*r0)
plt.ylim(-1.5*r0, 1.5*r0)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.legend()
plt.savefig('/home/aschoudhary/constraintongwwaveform/plots/OrbitEOB_Vs_BOB_beyodISCO.pdf')
plt.show()

########## Compare Psi2 Plots beyond ISCO ########################################################
# Resum Psi2 beyond ISCO
t_vec_resum = t_vec_BOB
r_resum_beyondISCO =r[idx_ini:idx_fin]
p_r_resum_beyondISCO =p_r[idx_ini:idx_fin]
phi_resum_beyondISCO =phi[idx_ini:idx_fin]
p_phi_resum_beyondISCO =p_phi[idx_ini:idx_fin]


Psi2_0_Growth_BOB_beyondISCO = Psi2_0_Growth_Pr(r_BOB, p_r_BOB, phi_BOB, p_phi_BOB, p)
Psi2_0_Growth_resum_beyondISCO = Psi2_0_Growth_Pr(r_resum_beyondISCO, p_r_resum_beyondISCO, phi_resum_beyondISCO, p_phi_resum_beyondISCO, p)

RR_Int_sigmadot = Psi2_sqr_fromWaveform(t_vec_resum , r_resum_beyondISCO, p_r_resum_beyondISCO, phi_resum_beyondISCO, p_phi_resum_beyondISCO, p)
BOB_Int_sigmadot=Psi2_sqr_fromWaveform_BOB(0.068, Ap, tau, t_vec_BOB, t0, tp)


plt.plot( t_vec_BOB, Psi2_0_Growth_BOB_beyondISCO,'c', label = r'$\Psi = BOB \; rad \; force$')  
plt.plot(t_vec_BOB, BOB_Int_sigmadot,'y--', label = r'$\Psi = -\int_{u_1}^{u_2}du|\dot{\sigma^{0}}|^2 \;BOB$')
plt.plot(t_vec_resum, Psi2_0_Growth_resum_beyondISCO,'k--', label = r'$\Psi = Resum \;RR$')
plt.plot(t_vec_resum, RR_Int_sigmadot,'g--', label = r'$\Psi = -\int_{u_1}^{u_2}du|\dot{\sigma^{0}}|^2 \;RR$')


#plt.xlim(ti, tf)
#plt.ylim(-0.06, 0.01)
plt.xlabel(r'$time$')
plt.ylabel(r'$\Psi$')
plt.legend()
plt.savefig('/home/aschoudhary/constraintongwwaveform/plots/Psi2EOBHamiltonia_Comparision.pdf')
plt.show()

########## Compare Psi2 Plots beyond ISCO using resum waveform for both ########################################################
# Resum Psi2 beyond ISCO
t_vec_resum = t_vec_BOB
r_resum_beyondISCO =r[idx_ini:idx_fin]
p_r_resum_beyondISCO =p_r[idx_ini:idx_fin]
phi_resum_beyondISCO =phi[idx_ini:idx_fin]
p_phi_resum_beyondISCO =p_phi[idx_ini:idx_fin]


Psi2_0_Growth_BOB_beyondISCO = Psi2_0_Growth_Pr(r_BOB, p_r_BOB, phi_BOB, p_phi_BOB, p)
Psi2_0_Growth_resum_beyondISCO = Psi2_0_Growth_Pr(r_resum_beyondISCO, p_r_resum_beyondISCO, phi_resum_beyondISCO, p_phi_resum_beyondISCO, p)

RR_Int_sigmadot = Psi2_sqr_fromWaveform(t_vec_resum , r_resum_beyondISCO, p_r_resum_beyondISCO, phi_resum_beyondISCO, p_phi_resum_beyondISCO, p)
BOB_Int_sigmadot=Psi2_sqr_fromWaveform(t_vec_BOB , r_BOB, p_r_BOB, phi_BOB, p_phi_BOB, p)

plt.plot( t_vec_BOB, Psi2_0_Growth_BOB_beyondISCO,'c', label = r'$\Psi = BOB \; rad \; force$')  
plt.plot(t_vec_BOB, BOB_Int_sigmadot,'y--', label = r'$\Psi = -\int_{u_1}^{u_2}du|\dot{\sigma^{0}}|^2 \;BOB$')
plt.plot(t_vec_resum, Psi2_0_Growth_resum_beyondISCO,'k--', label = r'$\Psi = Resum \;RR$')
plt.plot(t_vec_resum, RR_Int_sigmadot,'g--', label = r'$\Psi = -\int_{u_1}^{u_2}du|\dot{\sigma^{0}}|^2 \;RR$')


#plt.xlim(ti, tf)
#plt.ylim(-0.06, 0.01)
plt.xlabel(r'$time$')
plt.ylabel(r'$\Psi$')
plt.legend()
plt.savefig('/home/aschoudhary/constraintongwwaveform/plots/Psi2EOBHamiltonia_Comparision_usingResumModel.pdf')
plt.show()


