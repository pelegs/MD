import numpy as np
cimport numpy as np
from libc.math cimport exp, sqrt
#cython: boundscheck=False, wraparound=False, nonecheck=False

cdef double distance_c(np.ndarray[double, ndim=1] x1,
                       np.ndarray[double, ndim=1] x2):
    ''' Returns the distance between two vectors in R^3.
        Tests showed that this is faster than
        using NumPy (i.e. numpy.linalg.norm(x1-x2))'''

    cdef double dx = x1[0] - x2[0]
    cdef double dy = x1[1] - x2[1]
    cdef double dz = x1[2] - x2[2]
    
    return sqrt(dx**2 + dy**2 + dz**2)

def distance(x1, x2):
    return distance_c(x1, x2)

cdef np.ndarray[double, ndim=1] normalize_c(np.ndarray[double, ndim=1] v):
    L = sqrt(v[0]**2  +  v[1]**2  +  v[2]**2)
    if L != 0:
        return v/L
    else:
        return np.array(3*[float('inf')])

cdef np.ndarray[double, ndim=1] LJ_force_c(np.ndarray[double, ndim=1] x1,
                                           np.ndarray[double, ndim=1] x2):
    ''' Returns a force corresponding to the
        Lennard-Jones potential at the distance
        between two particles at x1 and x2.
        The force is directed from x1 to x2.'''
        
    cdef double r = distance(x1, x2)
    cdef double F = 48 * (r**-14  +  0.5*r**-8)
    cdef np.ndarray[double, ndim=1] n = normalize_c(x1-x2)
    
    return F * n

def LJ_force(x1, x2):
    return LJ_force_c(x1, x2)

cdef double dot_c(np.ndarray[double, ndim=1] v1,
                  np.ndarray[double, ndim=1] v2):
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]

def dot(v1, v2):
    return dot_c(v1, v2)
