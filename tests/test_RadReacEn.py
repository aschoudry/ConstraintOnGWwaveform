import sys
sys.path.insert(0, '/home/aschoudhary/constraintongwwaveform/src')

import numpy as np
import HamiltonianCoupledODE as ODEs
import matplotlib.pyplot as plt
import PN_radiationReaction as PN_RR

def CircularPr(r, nu):
    return np.sqrt(ODEs.Ap(r, nu)*(r**2)/(2*ODEs.A(r, nu)/r - ODEs.Ap(r, nu)))
 
## Initial conditions ##
r0 = 5.6

w0 = r0, 0.0, 0, CircularPr(r0, 0.25) 

p=0.25 # parameter nu
t_vec = np.arange(0, 12000, 1)

## Solve ODEs
yvec = ODEs.Coupled_HamiltonianODEs_solver(w0, t_vec, p)

r, p_r, phi, p_phi = yvec[:,0], yvec[:,1], yvec[:,2], yvec[:,3]

plt.plot(r*np.cos(phi), r*np.sin(phi), 'r')
plt.xlim(-2*w0[0], 2*w0[0])
plt.ylim(-2*w0[0], 2*w0[0])
plt.show()

Omega = ODEs.dphi_by_dt(r, p_r, phi, p_phi, p)

print(np.log(Omega[0])/np.log(r[0]))

