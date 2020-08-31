""" Script to solve coupled ODEs for EOB equations of motion  

Ashok , 2020-7-29
"""
from scipy.integrate import odeint 
import numpy as np 
from numpy.core.umath_tests import inner1d
from scipy.special import gamma

'Coupled ODEs Solver'
#	intial_conditions(w) = array[r0, phi0, p_r0, p_phi0]
#	domain(t) = time papermeter
# 	parameter_values = array[m, nu]  

def Coupled_HamiltonianODEs_solver(initial_condition, domain, parameter_values):

    w0 = initial_condition
    nu = parameter_values
    t_vec = domain
	
    abserr = 1.0e-8                         # error control parameters
    relerr = 1.0e-6

    yvec = odeint(derivs, w0, t_vec, args = (nu,), atol=abserr, rtol=relerr)
    return yvec


'Derivatives'

def derivs(w, t_vec, nu):

    r, p_r, phi, p_phi = w

    dphi_bydt= dphi_by_dt(r, p_r, phi, p_phi, nu)
    dr_bydt= dr_by_dt(r, p_r, phi, p_phi, nu)
    dp_phi_bydt= dp_phi_by_dt(r, p_r, phi, p_phi, nu)
    dp_r_bydt= dp_r_by_dt(r, p_r, phi, p_phi, nu)

    return dr_bydt, dp_r_bydt,  dphi_bydt, dp_phi_bydt
"""Expression for RHS of ODE. Eqn 3.8 to 3.11 in Notes,
   We is geomatrized units
"""

#####  Potentials and Hamiltonian ########
def A(r, nu):
    u=1.0/r
    return 1-2*(u)+2*nu*(u**3)
def Ap(r, nu):
    u=1.0/r
    return (2*u**2)-nu*6.0*u**4

def B(r, nu):
    u=1.0/r
    a= 1-2*(u)+2*nu*(u**3)
    d= 1.0-6.0*nu*pow(u,2)
    return  d/a  

def Bp(r, nu):
    N = 12.0*nu*A(r, nu)/pow(r,3) - (1.0 - 6.0*nu/pow(r,2)*Ap(r, nu))
    D = pow(A(r, nu),2)
    return N/D

def Heff(r, p_r, phi, p_phi, nu):
    return np.sqrt(A(r, nu)*(1+ pow(p_r,2)/B(r, nu) +(p_phi/r)**2))

def Heob(r, p_r, phi, p_phi, nu):
    return (1.0/nu)*np.sqrt(1+ 2*nu*(Heff(r, p_r, phi, p_phi, nu)-1))


##### ODE ##############################################################

def dphi_by_dt(r, p_r, phi, p_phi, nu):
    return A(r, nu)*p_phi/(nu*(r**2)*Heff(r, p_r, phi, p_phi, nu)*Heob(r, p_r, phi, p_phi, nu))

def dr_by_dt(r, p_r, phi, p_phi, nu):
    z3 = 2*nu*(4-3*nu)
    a= (A(r, nu)/B(r, nu))*(1.0/(nu*Heff(r, p_r, phi, p_phi, nu)*Heob(r, p_r, phi, p_phi, nu)))
    b=  p_r 
    return a*b

def dp_phi_by_dt(r, p_r, phi, p_phi, nu):
    omega = dphi_by_dt(r, p_r, phi, p_phi, nu)
    
    forceRR = F_phi(r, p_r,  phi, p_phi, nu)
    return forceRR

def dp_r_by_dt(r, p_r, phi, p_phi, nu):
    z3 = 2*nu*(4-3*nu)
    a = -(1.0/(2*nu*Heff(r, p_r, phi, p_phi, nu)*Heob(r, p_r, phi, p_phi, nu)))
    b = Ap(r, nu) + pow(p_r,2)*(Ap(r, nu)*B(r, nu)-A(r, nu)*Bp(r, nu))/pow(Bp(r, nu),2) \
            - 2*pow(p_phi,2)/pow(r,3) 
    return a*b


###################### EOB Radiation reaction force ##########################

def f_dis(r, p_r,  phi, p_phi, nu):
    omega = dphi_by_dt(r, p_r, phi, p_phi, nu)
    v=pow(omega,1.0/3.0)
    v_pole = (1.0/np.sqrt(3))*np.sqrt((1.0 + nu/3.0)/(1.0 - 35.0*nu/36.0))
    
    F1=0
    F2= -1247.0/336.0 - 35.0*nu/12.0
    F3= 4.0*np.pi
    F4= - 44711.0/9072.0 + 9271.0*nu/504.0 + 65.0*pow(nu,2)/18.0
    F5= - (8191.0/672.0 + 535.0*nu/24.0)*np.pi
    
    f1 = -1.0/v_pole
    f2 = F2 - F1/v_pole
    f3 = F3 - F2/v_pole
    f4 = F4 - F3/v_pole
    f5 = F5 - F4/v_pole
    
    c1 = -f1
    c2 = f1 - f2/f1
    c3 = (f1*f3-f2*f2)/(f1*(f1*f1-f2))
    c4 = -(f1*(f2*f2*f2 + f3*f3 + f1*f1*f4 - f2*(2.0*f1*f3 + f4)))/((f1*f1-f2)*(f1*f3-f2*f2))
    c5 = -((f1*f1-f2)*(-f3*f3*f3 + 2*f2*f3*f4 - f1*f4*f4 - f2*f2*f5 + f1*f3*f5))/((f1*f3-f2*f2)*(f2*f2*f2 \
           + f3*f3 + f1*f1*f4 - f2*(2.0*f1*f3 + f4)))

    f_Dis = 1.0/(1.0 + c1*v/(1.0 + c2*v/(1.0+ c3*v/(1.0 + c4*v/(1.0 + c5*v))))) 
    return f_Dis

def F_phi(r, p_r,  phi, p_phi, nu):
    omega = dphi_by_dt(r, p_r, phi, p_phi, nu)
    v=pow(omega,1.0/3.0)
    v_pole = (1.0/np.sqrt(3))*np.sqrt((1.0 + nu/3.0)/(1.0 - 35.0*nu/36.0))

    FN = -(32.0/5.0)*nu*pow(v,7)*f_dis(r, p_r,  phi, p_phi, nu)
    FD = 1.0 - v/v_pole
    return FN/FD


def h_inspiral(r, p_r,  phi, p_phi, nu):
    omega = dphi_by_dt(r, p_r, phi, p_phi, nu)
    v=pow(omega,1.0/3.0)
    h_in = pow(v,2)*np.cos(2*phi)
    return h_in


