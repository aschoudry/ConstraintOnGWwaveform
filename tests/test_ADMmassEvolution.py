import sys
sys.path.insert(0, '/home/aschoudhary/constraintongwwaveform/src')

import numpy as np
import HamiltonianCoupledODE as ODEs
import matplotlib.pyplot as plt
import PN_radiationReaction as PN_RR

def CircularPr(r, nu):
    return np.sqrt(ODEs.Ap(r, nu)*(r**2)/(2*ODEs.A(r, nu)/r - ODEs.Ap(r, nu)))
 
## Initial conditions ##
r0 = 7
p=0.25 # parameter nu

w0 = r0, 0.0, 0, CircularPr(r0, p) 

#t_vec = np.arange(0, 525, 0.5)
t_vec = np.arange(0, 7000, 1)


## Solve ODEs
yvec = ODEs.Coupled_HamiltonianODEs_solver(w0, t_vec, p)

r, p_r, phi, p_phi = yvec[:,0], yvec[:,1], yvec[:,2], yvec[:,3]

plt.figure()
plt.plot(r*np.cos(phi), r*np.sin(phi), label='$r_{0}=$'+str(r0))
plt.xlim(-2*w0[0], 2*w0[0])
plt.ylim(-2*w0[0], 2*w0[0])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.legend()
plt.savefig('/home/aschoudhary/constraintongwwaveform/plots/OrbitEOB'+str(r0)+'.pdf')
plt.show()

Omega = ODEs.dphi_by_dt(r, p_r, phi, p_phi, p)

H_EOB = ODEs.Heob(r, p_r, phi, p_phi, p)

print(np.log(Omega[0])/np.log(r[0]))
print(np.log(Omega[-1])/np.log(r[-1]))

plt.plot(t_vec, Omega)
plt.show()

plt.plot(t_vec-t_vec[-1], H_EOB-H_EOB[0])
plt.show()

plt.figure()
plt.plot(t_vec-t_vec[-1], H_EOB-H_EOB[0], label=r'$EOB \, \psi2$ = ')
plt.xlabel(r'$time$')
plt.ylabel(r'$\psi_{2}$')
plt.xlim(-50, 0)
plt.ylim(-0.065, -0.020)
plt.legend()
plt.savefig('/home/aschoudhary/constraintongwwaveform/plots/Psi2EOB.pdf')
plt.show()

h2 = ODEs.Rh22(r, p_r, phi, p_phi, p)
plt.plot(t_vec, h2)
plt.show()
