########################################
# MD simulation of Argon atoms         #
# Cython library for fast computation  #
# Written by Peleg Bar Sapir           #
########################################

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

cdef double abs_min(double a, double b, double c):
    cdef double abs_min = min(abs(a), abs(b), abs(c))
    index = -1
    for i, x in enumerate([a, b, c]):
        if abs(x) == abs_min:
            index = i
            break
    return [a, b, c][i]

def abs_min_py(a, b, c):
    return abs_min(a, b, c)

cdef np.ndarray[double, ndim=1] LJ_force_c(np.ndarray[double, ndim=1] p1,
                                           np.ndarray[double, ndim=1] p2,
                                           double Lx,
                                           double Ly,
                                           double Lz):
    ''' Returns a force corresponding to the
        Lennard-Jones potential at the distance
        between two particles at x1 and x2.
        The force is directed from x1 to x2.'''
        
    # Minimum image criterion

    cdef double x0 = p2[0]
    cdef double y0 = p2[1]
    cdef double z0 = p2[2]
    p2[0] = abs_min(p1[0]-x0, p1[0]-(x0+Lx), p1[0]-(x0-Lx))
    p2[1] = abs_min(p1[1]-y0, p1[1]-(y0+Ly), p1[1]-(y0-Ly))
    p2[2] = abs_min(p1[2]-z0, p1[2]-(z0+Lz), p1[2]-(z0-Lz))

    cdef double r = distance(p1, p2)
    cdef double F = 48 * (r**-14  +  0.5*r**-8)
    cdef np.ndarray[double, ndim=1] n = normalize_c(p1-p2)
    
    return F * n

def LJ_force(p1, p2, Lx, Ly, Lz):
    return LJ_force_c(p1, p2, Lx, Ly, Lz)

cdef double dot_c(np.ndarray[double, ndim=1] v1,
                  np.ndarray[double, ndim=1] v2):
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]

def dot(v1, v2):
    return dot_c(v1, v2)
