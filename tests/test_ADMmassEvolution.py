import sys
sys.path.insert(0, '/home/aschoudhary/constraintongwwaveform/src')

import numpy as np
import HamiltonianCoupledODE as ODEs
import matplotlib.pyplot as plt
import PN_radiationReaction as PN_RR

def CircularPr(r, nu):
    return np.sqrt(ODEs.Ap(r, nu)*(r**2)/(2*ODEs.A(r, nu)/r - ODEs.Ap(r, nu)))

def Psi2p(r, nu, M):
    return -nu*M*(1.0 - 2*M/r)/np.sqrt(1-3*M/r)
 
## Initial conditions ##
r0 = 15.0
p=0.249 # parameter nu
M=1
w0 = r0, 0.0, 0, CircularPr(r0, p) 

#t_vec = np.arange(0, 525, 0.5)
t_vec = np.arange(0, 5900, 1)


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

#plt.figure()
#plt.plot(t_vec-t_vec[-1], H_EOB-H_EOB[0], label=r'$EOB \, \psi2$ = ')
#plt.xlabel(r'$time$')
#plt.ylabel(r'$\psi_{2}$')
#plt.xlim(-50, 0)
#plt.ylim(-0.065, -0.020)
#plt.legend()
#plt.savefig('/home/aschoudhary/constraintongwwaveform/plots/Psi2EOB.pdf')
#plt.show()

plt.figure()
h2 = ODEs.Rh22(r, p_r, phi, p_phi, p)
plt.ylim(-1.0, 1.0)
plt.xlabel(r'$time$')
plt.ylabel(r'$h_{22}$')
plt.plot(t_vec-t_vec[-1], h2.real)
plt.legend()
plt.savefig('/home/aschoudhary/constraintongwwaveform/plots/h22EOB.pdf')
plt.show()

plt.figure()
psi2p = Psi2p(r,p,M )
#plt.ylim(-1.0, 1.0)
plt.xlabel(r'$time$')
plt.ylabel(r'$\psi_{2}^{0(1)}$')
plt.plot(t_vec-t_vec[-1], psi2p)
plt.legend()
#plt.savefig('/home/aschoudhary/constraintongwwaveform/plots/OrbitEOB'+str(r0)+'.pdf')
plt.show()

H2 = abs(ODEs.Rh22(r, p_r, phi, p_phi, p))
H2p2 = abs(ODEs.Rh22(r, p_r, phi, p_phi, p))**2
H2p2OM = abs(ODEs.Rh22(r, p_r, phi, p_phi, p))**2 *ODEs.dphi_by_dt(r, p_r, phi, p_phi, p)

H2 = H2-H2[0]
H2p2 = H2p2-H2p2[0]
H2p2OM = H2p2OM-H2p2OM[0]

print(H2[1:10])
print(H2p2[1:10])
print(H2p2OM[1:10])

heff = abs(ODEs.Heff(r, p_r, phi, p_phi, p))
heff = heff - heff[0]

print(heff[1:10])

plt.plot(t_vec, -H2p2OM)
plt.plot(t_vec, heff)
plt.show()
