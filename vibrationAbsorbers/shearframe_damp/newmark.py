# -*- coding: utf-8 -*-
"""
newmark(K,C,M,x0,v0,dt,N,F,beta,gamma)

Created: June 2020

@author: Oscar Bondo Ellekvist, s163774

"""

import numpy as np

def newmark(K,C,M,x0,v0,dt,N,F=None,beta=0.25,gamma=0.5):
    """
    newmark
    -------
    Evaluates the forced dynamic response of a linear system with n degrees
    of freedom at equally spaced times by the Newmark procedure.

    Parameters
    ----------
    K     : Array
            Stiffness matrix.
    C     : Array
            Damping matrix.
    M     : Array
            Mass matrix.
    x0    : Array
            Initial displacements.
    v0    : Array
            Initial velocities.
    dt    : Float
            Size of time step.
    N     : Integer
            Number of time steps.
    F     : Array
            Load amplitudes. (optional)        
    beta  : Float
            Newmark parameter [0 ; 0.5], beta=0.25 (default).
    gamma : Float
            Newmark parameter [0 ; 1.0], beta=0.5 (default).

    Returns
    -------
    x : Array
        Response history.
    v : Array
        Velocity history.
    a : Array
        Acceleration history.
    t : Array
        Discrete times.

    """
    
    # Number of dofs
    ndof = np.size(x0)
    
    # Set load vector
    if F is None:
        F = np.zeros((ndof,N+1),dtype=float)
    elif np.size(F,1) < (N+1):
        F = np.concatenate((F,np.zeros((ndof,N+1-np.size(F,1)),
                                       dtype=float)),axis=1)
    
    # Initialize output arrays
    x = np.zeros((ndof,N+1),dtype=float)
    v = np.zeros((ndof,N+1),dtype=float)
    a = np.zeros((ndof,N+1),dtype=float)
    t = np.zeros(N+1,dtype=float)
    
    # Modified mass matrix
    MM = M + gamma*dt*C + beta*dt*dt*K
    M1 = np.linalg.inv(MM)
    
    # Initial values
    t[0] = 0.0
    x[:,0] = np.ravel(x0)
    v[:,0] = np.ravel(v0)
    a[:,0] = np.linalg.solve(M,F[:,0]) - np.dot(C,v[:,0]) - np.dot(K,x[:,0])
    
    # Time incrementation loop
    for i in range(0,N):
        
        t[i+1] = t[i] + dt
        
        # Prediction step
        v[:,i+1] = v[:,i] + (1-gamma)*dt*a[:,i]
        x[:,i+1] = x[:,i] + dt*v[:,i] + (0.5-beta)*dt*dt*a[:,i]
        
        # Correction step
        a[:,i+1] = np.dot( M1 , F[:,i+1]- np.dot(C,v[:,i+1]) 
                          - np.dot(K,x[:,i+1]) )
        v[:,i+1] = v[:,i+1] + gamma*dt*a[:,i+1]
        x[:,i+1] = x[:,i+1] + beta*dt*dt*a[:,i+1]
        
    return(x,v,a,t)
