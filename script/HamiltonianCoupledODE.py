""" Script to solve coupled ODEs for EOB equations of motion  

Ashok , 2020-7-29
"""
from scipy.integrate import odeint 
import matplotlib.pyplot as plt
import numpy as np 
from numpy.core.umath_tests import inner1d

'Coupled ODEs Solver'
#	intial_conditions(w) = array[r0, phi0, p_r0, p_phi0]
#	domain(t) = time papermeter
# 	parameter_values = array[m, nu]  

def Coupled_HamiltonianODEs_solver(initial_condition, domain, parameter_values):

	w0 = initial_condition
	p = parameter_values
	t_vec = domain
	
	abserr = 1.0e-8                         # error control parameters
	relerr = 1.0e-6

	yvec = odeint(derivs, w0, t_vec, args = (p,), atol=abserr, rtol=relerr)
 	return yvec


'Derivatives'

def derivs(w, v, p):

	r, phi, p_r, p_phi = w
	nu = p  #parameters 
	
        dphi_bydt= dphi_by_dt(r, p_r, phi, p_phi, nu)
        dr_bydt= dr_by_dt(r, p_r, phi, p_phi, nu)
        dp_phi_bydt= dp_phi_by_dt(r, p_r, phi, p_phi, nu)
        dp_r_bydt= dp_r_by_dt(r, p_r, phi, p_phi, nu)

	return  dphi_bydt, dr_bydt, dp_phi_bydt, dp_r_bydt

"""Expression for RHS of ODE. Eqn 3.8 to 3.11 in Notes,
   We is geomatrized units
"""

#####  Potentials and Hamiltonian ########
def A(r, nu):
    u=1.0/r
    return 1- 2*(u) + 2*nu*(u**3)
def Ap(r, nu):
    u=1.0/r
    return (2*u**2) -6.0*u**4

def B(r, nu):
    u=1.0/r
    D=1-6*nu*(u**2) + 2*(3*nu-26)*nu*(u**3)
    return D/A(r, nu)

def Heff(r, p_r, phi, p_phi, nu):
    z3 = 2*nu*(4-3*nu)
    return np.sqrt(p_r**2 + A(r, nu)*(1+ (p_phi/r)**2 + z3*(p_r**4/r**2)))

def Heob(r, p_r, phi, p_phi, nu):
    return (1.0/nu)*np.sqrt(1+ 2*nu*(Heff(r, p_r, phi, p_phi, nu)-1))


##### ODE #################

def dphi_by_dt(r, p_r, phi, p_phi, nu):
    return A(r, nu)*p_phi/(nu*(r**2)*Heff(r, p_r, phi, p_phi, nu)*Heob(r, p_r, phi, p_phi, nu))

def dr_by_dt(r, p_r, phi, p_phi, nu):
    z3 = 2*nu*(4-3*nu)
    a= np.sqrt(A(r, nu)/B(r, nu))*(1.0/(nu*Heff(r, p_r, phi, p_phi, nu)*Heob(r, p_r, phi, p_phi, nu)))
    b=  p_r + z3*(2*A(r,nu)/r**2)*p_r**3
    return a*b

def dp_phi_by_dt(r, p_r, phi, p_phi, nu):
    return 0

def dp_r_by_dt(r, p_r, phi, p_phi, nu):
    z3 = 2*nu*(4-3*nu)
    a = -np.sqrt(A(r, nu)/B(r, nu))*(1.0/(2*nu*Heff(r, p_r, phi, p_phi, nu)*Heob(r, p_r, phi, p_phi, nu)))
    b = (Ap(r, nu) + (p_phi/r)**2 *(Ap(r, nu) - 2*A(r, nu)/r) + z3*(Ap(r, nu)/r**2 - 2*A(r, nu)/r**3)*p_r**4)
    return a*b







