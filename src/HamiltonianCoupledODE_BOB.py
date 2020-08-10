""" Script to solve coupled ODEs for EOB equations of motion  

Ashok , 2020-7-29
"""
from scipy.integrate import odeint 
import numpy as np 
from numpy.core.umath_tests import inner1d
import PN_radiationReaction as PN_RR
from scipy.special import gamma

'Coupled ODEs Solver'
#	intial_conditions(w) = array[r0, phi0, p_r0, p_phi0]
#	domain(t) = time papermeter
# 	parameter_values = array[m, nu]  

def Coupled_HamiltonianODEs_solver(initial_condition, domain, parameter_values):

    w0 = initial_condition
    nu, ng_radial, t0, tp, tau, Ap, r_switch, rad_reac = parameter_values
    t_vec = domain
	
    abserr = 1.0e-8                         # error control parameters
    relerr = 1.0e-6

    yvec = odeint(derivs, w0, t_vec, args = (nu, ng_radial, t0, tp, tau, Ap, r_switch, rad_reac), atol=abserr, rtol=relerr)
    return yvec


'Derivatives'

def derivs(w, t_vec, nu, ng_radial, t0, tp, tau, Ap, r_switch, rad_reac):

    r, p_r, phi, p_phi = w

    dphi_bydt= dphi_by_dt(r, p_r, phi, p_phi, nu)
    dr_bydt= dr_by_dt(r, p_r, phi, p_phi, nu)
    dp_phi_bydt= dp_phi_by_dt(r, p_r, phi, p_phi, nu, ng_radial, t_vec, t0, tp, tau, Ap, r_switch, rad_reac)
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
    return 1+2*u+2*(2-3*nu)*(u**2) 

def Heff(r, p_r, phi, p_phi, nu):
    z3 = 2*nu*(4-3*nu)
    return np.sqrt(p_r**2 + A(r, nu)*(1+ (p_phi/r)**2 + z3*(p_r**4/r**2)))

def Heob(r, p_r, phi, p_phi, nu):
    return (1.0/nu)*np.sqrt(1+ 2*nu*(Heff(r, p_r, phi, p_phi, nu)-1))


##### ODE ##############################################################

def dphi_by_dt(r, p_r, phi, p_phi, nu):
    return A(r, nu)*p_phi/(nu*(r**2)*Heff(r, p_r, phi, p_phi, nu)*Heob(r, p_r, phi, p_phi, nu))

def dr_by_dt(r, p_r, phi, p_phi, nu):
    z3 = 2*nu*(4-3*nu)
    a= np.sqrt(A(r, nu)/B(r, nu))*(1.0/(nu*Heff(r, p_r, phi, p_phi, nu)*Heob(r, p_r, phi, p_phi, nu)))
    b=  p_r + z3*(2*A(r,nu)/r**2)*p_r**3
    return a*b

def dp_phi_by_dt(r, p_r, phi, p_phi, nu, ng_radial, t_vec, t0, tp, tau, Ap, r_switch, rad_reac):
    omega = dphi_by_dt(r, p_r, phi, p_phi, nu)
    
    if r >= r_switch:
        forceRR = f_phi(r, p_r, phi, p_phi, nu)
    else:
        if rad_reac==1:
            forceRR = f_phi_BOB(r, p_r, phi, p_phi, nu, ng_radial, t_vec, t0, tp, tau, Ap, r_switch)
        else:
            forceRR = 0.0

    return forceRR

def dp_r_by_dt(r, p_r, phi, p_phi, nu):
    z3 = 2*nu*(4-3*nu)
    a = -np.sqrt(A(r, nu)/B(r, nu))*(1.0/(2*nu*Heff(r, p_r, phi, p_phi, nu)*Heob(r, p_r, phi, p_phi, nu)))
    b = (Ap(r, nu) + (p_phi/r)**2 *(Ap(r, nu) - 2*A(r, nu)/r) + z3*(Ap(r, nu)/r**2 - 2*A(r, nu)/r**3)*p_r**4)
    return a*b


###################### EOB Radiation reaction force ##########################


def rho(x, nu, PN_order):
    
    gammaE = 0.57725
    eulerlog2 = gammaE + 2.0*np.log(2) + (1.0/2)*np.log(x)

    B0 = 1
    B1 = 55.0*nu/84.0 - 43.0/42
    B2 = 19583*pow(nu,2)/42336.0 - 33025.0*nu/21168.0 - 20555.0/10584.0
    B3 = 10620745.0*pow(nu,3)/39118464.0 - 6292061.0*pow(nu,2)/3259872.0 + 41.0*pow(np.pi,2)*nu/192 - 48993925.0*nu/9779616.0\
            - 428.0*eulerlog2/105.0 + 1556919113.0/122245200.0
    B4 = 9202.0*eulerlog2/2205.0 - 387216563023.0/160190110080.0
    B5 = 439877.0*eulerlog2/55566.0 - 16094530514677.0/533967033600.0

    if PN_order==0:
    	rh = (B0)
    if PN_order==1:
    	rh = (B0 + B1*x) 	
    if PN_order==2:
    	rh = (B0 + B1*x + B2*pow(x,2))
    if PN_order==3:
    	rh = (B0 + B1*x  + B2*pow(x,2)+ B3*pow(x,3))
    if PN_order==4:
    	rh = (B0 + B1*x  + B2*pow(x,2)+ B3*pow(x,3)+ B4*pow(x,4))
    if PN_order==5:
    	rh = (B0 + B1*x  + B2*pow(x,2)+ B3*pow(x,3)+ B4*pow(x,4)+ B5*pow(x,5))
	
    return abs(rh)

def delta22(r, p_r, phi, p_phi, nu):
    omega = dphi_by_dt(r, p_r, phi, p_phi, nu)
    Hreal = Heob(r, p_r, phi, p_phi, nu)

    x=pow(omega, 2.0/3)
    yb=x
    y=pow(Hreal*omega, 2.0/3.0)

    d22 = (7.0/3.0)*pow(y, 3.0/2.0) +  (428.0*np.pi/105.0)*pow(y, 3) - 24*nu*pow(yb, 5.0/2.0)
    return d22

def Seff(r, p_r, phi, p_phi, nu):
    return Heff(r, p_r, phi, p_phi, nu)

def T22(r, p_r, phi, p_phi, nu):
    omega = dphi_by_dt(r, p_r, phi, p_phi, nu)
    Hreal = Heob(r, p_r, phi, p_phi, nu)
    T1 = (gamma(3.0 -4.0*1j*omega*Hreal)/gamma(3))
    T2 = np.exp(2*np.pi*omega*Hreal)
    T3 = np.exp(4*1j*omega*Hreal*np.log(8.0*omega))
    return T1*T2*T3

def Y2m2(theta, phi):
    return (1.0/4.0)*np.sqrt(15.0/(2*np.pi))*np.exp(-2*1j*phi)*(np.sin(theta)**2)

def Rh22(r, p_r, phi, p_phi, nu):
    omega = dphi_by_dt(r, p_r, phi, p_phi, nu)
    x=pow(omega, 2.0/3)
    PN_order=5

    n22 = -4*8*np.pi*np.sqrt(6)/(15)
    c22 = 1
    y2m2 = Y2m2(np.pi/2.0, phi)
    seff = Seff(r, p_r, phi, p_phi, nu)
    t22 = T22(r, p_r, phi, p_phi, nu)
    rh = rho(x, nu, PN_order)

    return nu*n22*c22*x*y2m2*seff*t22*pow(rh,2)

# Radiation reaction force
def f_phi(r, p_r, phi, p_phi, nu):  
    omega = dphi_by_dt(r, p_r, phi, p_phi, nu)
    x=pow(omega, 2.0/3)

    absh22 = abs(Rh22(r, p_r, phi, p_phi, nu))
    Fl = (2.0/(16*np.pi))*((2*omega)**2)*absh22*absh22
    F=-Fl/omega
    return F    
 
# time vec for BOB radiation reaction force
def time_vec(r0, rf, ng_orbit, ng_radial):
    t_vec = np.array([])
    r_vec = np.linspace(r0, rf, ng_radial) 
    dt_vec = (1.0/ng_radial)*np.sqrt(4*pow(np.pi,2)*pow(r_vec,3)) 
    dt_vec_full = np.repeat(dt_vec, ng_orbit) 
    t_vec_full = np.cumsum(dt_vec_full) 
    return t_vec_full, dt_vec_full 

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

def Omega_BOB(omega0, tau, t, t0, tp):
    omqnm = Omega_QNM(0.0, 0.0, 0.25)
    k = (pow(omqnm,4)-pow(omega0,4))/(1- np.tanh((t0-tp)/tau))
    om = (pow(omega0,4) + k*(np.tanh((t-tp)/tau) - np.tanh((t0-tp)/tau) ))**(1.0/4)
    return om

#Radiation reaction force from BOB
def f_phi_BOB(r, p_r, phi, p_phi, nu, ng_radial ,t_vec, t0, tp, tau, Ap, r_switch):  
    omega0 = 0.068
    omega = Omega_BOB(omega0, tau, t_vec, t0, tp)

    #Ap = pow(Omega_BOB(omega0, tau, tp, t0, tp),2)
    abs_psi4=abs(Ap/np.cosh((t_vec-tp)/tau))
    h22dot = abs_psi4/(2*omega)

    Fl = (2.0/(16*np.pi))*h22dot*h22dot
    F=-Fl/omega
    return F


