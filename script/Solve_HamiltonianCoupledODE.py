import sys
sys.path.insert(0, '/home/aschoudhary/constraintongwwaveform/src')

import numpy as np
import HamiltonianCoupledODE as ODEs
import matplotlib.pyplot as plt
import PN_radiationReaction as PN_RR

def CircularPr(r, nu):
    return np.sqrt(ODEs.Ap(r, nu)*(r**2)/(2*ODEs.A(r, nu)/r - ODEs.Ap(r, nu)))
 
## Initial conditions ##
w0 = 10, 0.0, 0, CircularPr(10, 0.25) 
w01 = 60, 0.0, 0, CircularPr(60, 0.25) 
w02 = 80, 0.0, 0, CircularPr(80, 0.25) 

p=0.25 # parameter nu
t_vec = np.arange(0, 1000, 0.5)

## Solve ODEs
yvec = ODEs.Coupled_HamiltonianODEs_solver(w0, t_vec, p)
yvec1 = ODEs.Coupled_HamiltonianODEs_solver(w01, t_vec, p)
yvec2 = ODEs.Coupled_HamiltonianODEs_solver(w02, t_vec, p)


r, p_r, phi, p_phi = yvec[:,0], yvec[:,1], yvec[:,2], yvec[:,3]

r1, p_r1, phi1, p_phi1 = yvec1[:,0], yvec1[:,1], yvec1[:,2], yvec1[:,3]
r2, p_r2, phi2, p_phi2 = yvec2[:,0], yvec2[:,1], yvec2[:,2], yvec2[:,3]


plt.plot(r*np.cos(phi), r*np.sin(phi), 'r')
plt.plot(r1*np.cos(phi1), r1*np.sin(phi1), 'g')
plt.plot(r2*np.cos(phi2), r2*np.sin(phi2), 'k')

plt.xlim(-2*w02[0], 2*w02[0])
plt.ylim(-2*w02[0], 2*w02[0])
plt.show()

Omega = ODEs.dphi_by_dt(r, p_r, phi, p_phi, p)
Omega1 = ODEs.dphi_by_dt(r1, p_r1, phi1, p_phi1, p)
Omega2 = ODEs.dphi_by_dt(r2, p_r2, phi2, p_phi2, p)

print(np.log(Omega[0])/np.log(r[0]))
print(np.log(Omega1[0])/np.log(r1[0]))
print(np.log(Omega2[0])/np.log(r2[0]))

