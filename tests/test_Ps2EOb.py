import sys
sys.path.insert(0, '/home/aschoudhary/constraintongwwaveform/src')

import numpy as np
import HamiltonianCoupledODE as ODEs
import matplotlib.pyplot as plt
import PN_radiationReaction as PN_RR

# create a time vector with variable resulution
def time_vec(r0, rf, ng_orbit, ng_radial):
    t_vec = np.array([])
    r_vec = np.linspace(r0, rf, ng_radial) 
    dt_vec = (1.0/ng_radial)*np.sqrt(4*pow(np.pi,2)*pow(r_vec,3)) 
    dt_vec_full = np.repeat(dt_vec, ng_orbit) 
    t_vec_full = np.cumsum(dt_vec_full) 
    return t_vec_full 

# find intex of nearest value in an array
def find_nearest1(array,value):
    idx,val = min(enumerate(array), key=lambda x: abs(x[1]-value))
    return idx


def Y22(Theta, Phi):
    return (1.0/4.0)*np.sqrt(15.0/(2*np.pi))*np.sin(Theta)*np.sin(Theta)*np.exp(1j*2.0*Phi)

def m2Y22(Theta, Phi):
    return (1.0/8)*np.sqrt(5.0/np.pi)*(1+np.cos(Theta))*(1+np.cos(Theta))*np.exp(1j*2.0*Phi)


def CircularPr(r, nu):
    return np.sqrt(ODEs.Ap(r, nu)*(r**2)/(2*ODEs.A(r, nu)/r - ODEs.Ap(r, nu)))

def Psi2p(r, nu, M):
    return -nu*M*(1.0 - 2*M/r)/np.sqrt(1-3*M/r)

def Psi2_0_Growth(r, p_r, phi, p_phi, p):
    HEOB = ODEs.Heob(r, p_r, phi, p_phi, p)
    HEOB = HEOB-HEOB[0]
    M = HEOB*p
    M = M - M[0]
    return M

def Psi2_0_Growth_Pr(r, p_r, phi, p_phi, p):
    HEOB = ODEs.Heob(r, p_r, phi, p_phi, p)
    HEOB = HEOB-HEOB[0]
    M = HEOB*p
    M = M - M[0]
    Pr = (6.0/2/np.sqrt(2))*p_r*p
    Pr = Pr
    a = ODEs.A(r, p)
    b = ODEs.B(r, p)

    return  M + np.sqrt(b/a)*Pr


def Psi2_fromWaveform(t_vec, r, p_r, phi, p_phi, p):
    dt = t_vec[1]-t_vec[0]
    sigma_sigma_Dot =  (abs(ODEs.Rh22(r, p_r, phi, p_phi, p))**2)*2*ODEs.dphi_by_dt(r, p_r, phi, p_phi, p)
    sigma_sigma_Dot = sigma_sigma_Dot - sigma_sigma_Dot[0]
    abs_Sigma_Dot_sqr = (2*ODEs.dphi_by_dt(r, p_r, phi, p_phi, p)*abs(ODEs.Rh22(r, p_r, phi, p_phi, p)))**2
    abs_Sigma_Dot_sqr = abs_Sigma_Dot_sqr - abs_Sigma_Dot_sqr[0]
    Int_abs_Sigma_Dot_sqr = np.cumsum(abs_Sigma_Dot_sqr)*dt
    Int_abs_Sigma_Dot = 0.5*(ODEs.Rh22(r, p_r, phi, p_phi, p)*Y22(np.pi/2, np.pi/2)).real
    Int_abs_Sigma_Dot = Int_abs_Sigma_Dot - Int_abs_Sigma_Dot[0]
    Psi2_diff = (sigma_sigma_Dot - Int_abs_Sigma_Dot_sqr)*abs(m2Y22(np.pi/2, np.pi/2))**2 #- Int_abs_Sigma_Dot
    Psi2_diff = Psi2_diff - Psi2_diff[0]
    return Psi2_diff

def Psi2_sqr_fromWaveform(t_vec, r, p_r, phi, p_phi, p):
    dt = t_vec[1]-t_vec[0]
    abs_Sigma_Dot_sqr = (2*ODEs.dphi_by_dt(r, p_r, phi, p_phi, p)*abs(ODEs.Rh22(r, p_r, phi, p_phi, p)))**2
    abs_Sigma_Dot_sqr = abs_Sigma_Dot_sqr - abs_Sigma_Dot_sqr[0]
    Int_abs_Sigma_Dot_sqr = -np.cumsum(abs_Sigma_Dot_sqr)*dt*abs(m2Y22(np.pi/2, np.pi/2))**2
    Psi2_diff = Int_abs_Sigma_Dot_sqr
    return Psi2_diff

##### Initial setup for binary very far ###############################################
r0 = 50.0
p=0.25 # parameter nu
M=1
w0 = r0, 0.0, 0, CircularPr(r0, p) 

#t_vec = np.arange(0, 525, 0.5)
#t_vec = np.arange(0, 5900, 1)

t_vec = time_vec(r0, 15, 1388, 100)


## Solve ODEs
yvec = ODEs.Coupled_HamiltonianODEs_solver(w0, t_vec, p)

r, p_r, phi, p_phi = yvec[:,0], yvec[:,1], yvec[:,2], yvec[:,3]
print(r[-1], p_r[-1])

########################################################################################

## Initial conditions ##
r0, p_r0, phi0, p_phi0 = r[-1], p_r[-1], phi[-1], p_phi[-1]

p=0.25 # parameter nu
M=1
pr0 = p_phi[-1]

w0 = r0, p_r0, phi0, p_phi0 

#w0 = r[-1], 0.0, 0, p_r[-1]

#t_vec = np.arange(0, 525, 0.5)
#t_vec = np.arange(0, 5900, 1)

t_vec = time_vec(r0, 3, 100, 100)


## Solve ODEs
yvec = ODEs.Coupled_HamiltonianODEs_solver(w0, t_vec, p)

r, p_r, phi, p_phi = yvec[:,0], yvec[:,1], yvec[:,2], yvec[:,3]

print(find_nearest1(r,3), t_vec[find_nearest1(r,3)])
idx = find_nearest1(r,3)
## stop at r=3M
t_vec, r, p_r, phi, p_phi = t_vec[:idx], r[:idx], p_r[:idx], phi[:idx], p_phi[:idx] 
t_vec = t_vec - t_vec[-1]
Omega = ODEs.dphi_by_dt(r, p_r, phi, p_phi, p)

H_EOB = ODEs.Heob(r, p_r, phi, p_phi, p)

plt.plot(t_vec, Psi2_0_Growth(r, p_r, phi, p_phi, p), label = r'$\Psi = M_{B}$')
plt.plot(t_vec, Psi2_0_Growth_Pr(r, p_r, phi, p_phi, p), label = r'$\Psi = M_{B} + \frac{6}{2 \sqrt{2}}P_{r}$')
plt.plot(t_vec, Psi2_sqr_fromWaveform(t_vec, r, p_r, phi, p_phi, p), label = r'$\Psi = \int_{u_1}^{u_2}du|\dot{\sigma^{0}}|^2$')
#plt.xlim(0, 5900)
plt.ylim(-0.06, 0.01)
plt.xlabel(r'$time$')
#plt.ylabel(r'$\psi_{2}^{0}$')
plt.legend()
#plt.savefig('/home/aschoudhary/constraintongwwaveform/plots/Psi2EOBHamiltonia.pdf')
plt.show()


