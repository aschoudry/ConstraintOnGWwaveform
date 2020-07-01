import numpy as np

### Including radiation reaction from PN expansion


def f_phi(omega, nu):              # Radiation reaction force 
    x=pow(omega, 2.0/3)
    H22=1 + x*(-107.0/42 + 55*nu/42) + 2*np.pi*(x**(3.0/2))
    h22 = (2*nu*x)*np.sqrt(16*np.pi/5)*(H22)
    Fl = (2.0/(16*np.pi))*((2*omega)**2)*h22*h22
    return Fl



