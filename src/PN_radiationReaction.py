import numpy as np

### Including radiation reaction from PN expansion


def f_phi(omega, nu, PN_order):              # Radiation reaction force 
    x=pow(omega, 2.0/3)
    H=H22(x, nu, PN_order)
    h22 = (2*nu*x)*np.sqrt(16*np.pi/5)*H
    Fl = (2.0/(16*np.pi))*((2*omega)**2)*h22*h22
    F=-Fl/omega
    return F/nu

def H22(x, nu, PN_order):

    gamma =0.57721
    B0 = 1
    B1 = -107.0/42 + 55.0*nu/42
    B1p5 = 2*np.pi
    B2 =-2173.0/1512 - 1069.0*nu/216 + 2047.0*pow(nu,2)/1512
    B2p5 = -107.0*np.pi/21 - 1j*24*nu + 34*np.pi*nu/21.0
    B3 = 27027409.0/646800 - 856.0*gamma/105.0 + 1j*428*np.pi/105 + 2.0*pow(np.pi,2)/3 + nu*(-278185.0/33264 + 41.0*pow(np.pi,2)/96.0)\
            - 20261.0*pow(nu,2)/2772.0 + 114635.0*pow(nu,3)/99792 - 428.0*np.log(16*x)/105.0
    
    if PN_order==0:
    	H = (B0)
    if PN_order==1:
    	H = (B0 + B1*x) 
    if PN_order==1.5:
    	H = (B0 + B1*x + B1p5*pow(x,1.5))	
    if PN_order==2:
    	H = (B0 + B1*x + B1p5*pow(x,1.5)+ B2*pow(x,2))
    if PN_order==2.5:
    	H = (B0 + B1*x + B1p5*pow(x,1.5) + B2*pow(x,2) + B2p5*pow(x,2.5))
    if PN_order==3:
    	H = (B0 + B1*x + B1p5*pow(x,1.5) + B2*pow(x,2) + B2p5*pow(x,2.5) + B3*pow(x,3))
	
    return abs(H)


