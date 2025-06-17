# -*- coding: utf-8 -*-
"""
shearframe_damp
----------

Driver for tuning of TID

Free to use by anyone

"""

# Import packages
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys as sys

# Import functions
from newmark import newmark
from scipy import linalg as linalg

### SYSTEM MATRICES

# floor mass and interstory stiffness
# Total Floors = 5
mf = 1e3
kf = 1e6

M = mf*np.array([[1 , 0 , 0 , 0 , 0],
                 [0 , 1 , 0 , 0 , 0],
                 [0 , 0 , 1 , 0 , 0],
                 [0 , 0 , 0 , 1 , 0],
                 [0 , 0 , 0 , 0 , 1],])

K = kf*np.array([[2 , -1 , 0 , 0 , 0],
                 [-1 , 2 , -1 , 0 , 0],
                 [0 , -1 , 2 , -1 , 0],
                 [0 , 0 , -1 , 2 , -1],
                 [0 , 0 , 0 , -1 , 1],])

# Dimensions of model
ndof = np.size(M,0)

### Absorber properties

# absorber mass / inertance (kg)
ma = 500

# placement in connectivity vector
W = np.array([[1], [0], [0], [0], [0]])

# mode to damp
s = 1

### Free EVP

# Generalized eigenvalue problem
(D,U) = linalg.eig(K,M)

# Natural angular frequencies
omega = np.sqrt(D).real

# Sort frequencies and mode shapes
iw = np.argsort(omega)
omega = omega[iw]
freq  = omega/(2*np.pi)
U = U[:,iw]
print(freq[:5])

# target freq and mode
omegas = omega[s-1]
us = U[:,s-1]

# normalize mode shape to unit absorber displacement
ud = np.dot(np.transpose(W),us)
us = us/ud

# modal mass for mode s
ms = np.dot(np.transpose(us),np.dot(M,us))
ks = np.dot(np.transpose(us),np.dot(K,us))

# mass ratio
mu = ma/ms
print(mu)

### Clamped EVP

# including absorber inertance
Minf = M + ma*np.dot(W,np.transpose(W))

# Generalized eigenvalue problem
(D,U) = linalg.eig(K,Minf)

# Natural frequencies
omegainf = np.sqrt(D).real

# Sort frequencies and mode shapes
iw = np.argsort(omegainf)
omegainf = omegainf[iw]
freqinf  = omegainf/(2*np.pi)
U = U[:,iw]
print(freqinf[:5])

# target freq and mode
omegainfs = omegainf[s-1]

# EMCF
mua = (omegas**2 - omegainfs**2)/omegainfs**2
print(mua)

### Absorber Tuning

# spring stiffness
ka = mua/(1+mua**2)*ks
omegaa = np.sqrt(ka/ma)
freqa = omegaa/(2*np.pi)
print(freqa)

# dashpot viscosity
ca = np.sqrt(2*mua**3/(1+mua)**3)*np.sqrt(ks*ms)
zetaa = ca/(2*np.sqrt(ka*ma))
print(zetaa)

# uncorrected tuning (mua = mu)
kdet = mu/(1+mu**2)*ks
cdet = np.sqrt(2*mu**3/(1+mu)**3)*np.sqrt(ks*ms)

### Simulations of shearframe (without TID)

# Initial conditions
x0 = np.zeros((ndof,1),dtype=float)
v0 = x0

# Time simulation parameters
t0 = 0                  # initial time
N  = 12000              # number of time steps
dt = 0.005              # time step size

# Top floor load
tf = dt*np.arange(0,(N+1),1,dtype=float)    # time vector
omegaf = omega[0]                           # forcing freq
F = np.zeros((ndof,N+1),dtype=float)
F[5-1,:] = 1e3*np.sin(omegaf*tf)            # top floor harmonic force

# Rayleigh structural damping
zetamin = 0.01
omegamin = omega[0]
aR = zetamin*omegamin
bR = zetamin/omegamin
C = aR*M + bR*K

# Newmark time integration - structure without TID
gamma = 0.5
beta  = 0.25
(q0,v0,a0,t) = newmark(K,C,M,x0,v0,dt,N,F,beta=beta,gamma=gamma)

# plot response at dofplot
dofplot = 1

# Plot response history using matplotlib
plt.close('all')

fig1 = plt.figure(1,figsize=[12,4])
fig1.add_subplot(121)
plt.plot(t,q0[dofplot-1,:],color='blue')
plt.title(r'no TID')
plt.xlabel(r'$t$ - [s]')
plt.ylabel(r'$q_{top}$ - [m]')
plt.grid()
#plt.xlim([0, 600])
#plt.ylim([-0.2, 3.0])

fig1.add_subplot(122)           # subplot 2 = acc
plt.plot(t,a0[dofplot-1,:],color='red')
plt.xlabel(r'$t$ - [s]')
plt.ylabel(r'$\ddot{q}_{top}$ - [m/s$^2$]')
plt.grid()

# Save response history to .png file
plt.savefig('shearframe_0.png',bbox_inches='tight')
plt.show()

### Simulations of shearframe + absorber (TID)

# extended matrices with TID
MM1 = np.vstack((M,np.zeros((1,ndof))))
MM2 = np.vstack((np.zeros((ndof,1)),ma))
MM = np.hstack((MM1,MM2))

CC11 = C + ca*np.dot(W,np.transpose(W))
CC12 = -ca*W
CC21 = -ca*np.transpose(W)
CC22 = ca
CC1 = np.vstack((CC11,CC21))
CC2 = np.vstack((CC12,CC22))
CC = np.hstack((CC1,CC2))

KK11 = K + ka*np.dot(W,np.transpose(W))
KK12 = -ka*W
KK21 = -ka*np.transpose(W)
KK22 = ka
KK1 = np.vstack((KK11,KK21))
KK2 = np.vstack((KK12,KK22))
KK = np.hstack((KK1,KK2))

FF = np.zeros((ndof+1,N+1),dtype=float)
FF[5-1,:] = 1e3*np.sin(omegaf*tf)

# Initial conditions
xx0 = np.zeros((ndof+1,1),dtype=float)
vv0 = xx0

# Newmark time integration - structure without TID
(qq,vv,aa,t) = newmark(KK,CC,MM,xx0,vv0,dt,N,FF,beta=beta,gamma=gamma)

fig2 = plt.figure(1,figsize=[12,4])

fig2.add_subplot(121)
plt.plot(t,qq[dofplot-1,:],color='blue')
plt.title(r'with TID')
plt.xlabel(r'$t$ - [s]')
plt.ylabel(r'$q_{top}$ - [m]')
plt.grid()
#plt.xlim([0, 600])
#plt.ylim([-0.2, 3.0])

fig2.add_subplot(122)           # subplot 2 = acc
plt.plot(t,qq[ndof-1+1,:],color='red')
plt.xlabel(r'$t$ - [s]')
plt.ylabel(r'$q_{tid}-q_{1}$ - [m]')
plt.grid()

# Save response history to .png file
plt.savefig('shearframe_a.png',bbox_inches='tight')
plt.show()

# stop code here - move up to break code where you like
# sys.exit()