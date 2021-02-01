# This script computes final mass and kick velocity using global supermomentum law.
import sys
sys.path.insert(0, '../src')
import BOB_functions as BOB
import matplotlib.pyplot as plt
import numpy as np
import sxs
from scipy.interpolate import interp1d
import os
from EOBUtils import *
import scipy.special as sc
from sympy.physics.quantum.cg import Wigner3j

## Some good q=1 nospin BBH file nums (1132, 3, 4), (1155, 2, 3),(0002, 5,6), (1122, 4,5), spinning (0160, 3,4)
catalog = sxs.load("catalog")
file_num = '1122'
L1=4; L2=5

L3 = sxs.load("SXS:BBH:"+str(file_num)+"/Lev("+str(L1)+")/rhOverM")
w_L3 = L3['OutermostExtraction.dir'] 

metadata = sxs.load("SXS:BBH:"+str(file_num)+"/Lev("+str(L2)+")/metadata.json")

# Shift to some reference time

h22_SXS = np.array(w_L3[:, w_L3.index(2, 2)].data.tolist())
time_SXS = w_L3[:, w_L3.index(2, 2)].t

# Define the triple product of spherical harmonics
def fac(n):
    return np.math.factorial(abs(n))

def gj(kj, lj, mj, sj):
    N= pow(-1.0, kj)*(fac(lj+mj)*fac(lj-mj)*fac(lj+sj)*fac(lj-sj))**(0.5)
    D= fac(kj)*fac(lj+mj-kj)*fac(lj-sj-kj)*fac(sj-mj+kj)
    return N/D

def G(s1,s2,s3,l1,l2,l3,m1,m2,m3):
    K1=pow(-1, s1+s2+s3)*np.sqrt((2*l1+1)*(2*l2+1)*(2*l3+1))/np.sqrt(4.0*np.pi)**3
    K2=2.0*np.pi*np.kron(-m1,m2+m3)
    K3=0.0
    k1i=max(0, m1-s1)
    k2i=max(0, m2-s2)
    k3i=max(0, m3-s3)
    k1f=min(l1+m1, l1-s1)
    k2f=min(l2+m2, l2-s2)
    k3f=min(l2+m2, l2-s2)

    k1_vec = np.arange(k1i, k1f+1)
    k2_vec = np.arange(k2i, k2f+1)
    k3_vec = np.arange(k3i, k3f+1)

    for k1 in k1_vec:
        for k2 in k2_vec:
            for k3 in k3_vec:
                p1=2.0*k1+s1-m1
                p2=2.0*k2+s2-m2
                p3=2.0*k3+s3-m3
                a=1.0+(p1+p2+p3)/2.0
                b=1.0+l1+l2+l3 - (p1+p2+p3)/2.0
                Beta_ab = sc.beta(a, b)
                if K2==0.0:
                    K3+=0.0
                else:
                    K3+=2.0*gj(k1, l1, m1, s1)*gj(k2, l2, m2, s2)*gj(k3, l3, m3, s3)*Beta_ab
    
    return K1*K2*K3

def alpha_SXS_lm(l,m, h):
    SXS_modes = h.LM
    alpha_lm = np.zeros(h.time.size,dtype=complex) 
    for i in SXS_modes:
        l1, m1 = i[0], i[1]
        hl1m1 = np.array(h[:, h.index(l1, m1)].data.tolist())
        for j in SXS_modes:
            l2, m2 = j[0], j[1]
            hl2m2_conj = np.array(h[:, h.index(l2, m2)].data.tolist()).conjugate()
            alpha = hl1m1*hl2m2_conj*G(2,-2,0,l1,l2,l,m1,-m2,-m)
            alpha_lm += alpha
    return alpha_lm

print(alpha_SXS_lm(4,0, w_L3))
print(G(2,2,0,2,2,2,2,-1,0))
    
