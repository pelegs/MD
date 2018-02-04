#################################
# MD simulation of Argon atoms  #
# Classes and simple functions  #
# Written by Peleg Bar Sapir    #
#################################

import numpy as np
import mdlibc
from itertools import chain
import sys

double = np.float64

axes = np.array([[1, 0, 0], [-1, 0, 0],
                 [0, 1, 0], [0, -1, 0],
                 [0, 0, 1], [0, 0, -1]])

class atom:
    def __init__(self,
                 pos     = np.zeros(3),
                 vel     = np.zeros(3),
                 box     = None,
                 element = 'Ar',
                 ID      = -1):
        
        self.pos  = pos.astype(double)
        self.vel  = vel.astype(double)
        self.acc  = np.zeros(3).astype(double)
        self.F    = np.zeros(3).astype(double)
        
        self.m      = 1.0
        self.v_half = np.zeros(3).astype(double)

        self.cell = np.floor(self.pos / box.L * box.N-2).astype(int)
        self.neighbors = []

        self.element = element
        self.ID = ID

    def set_neighbors(self, box):
        self.neighbors = []
        self.cell = np.floor(self.pos / box.L * box.N).astype(int)
        cells = list(chain(*[box.cells[i][j][k]
                             for (i, j, k) in get_neighboring_indices(self.cell[0],
                                                                      self.cell[1],
                                                                      self.cell[2],
                                                                      box.N)]))
        self.neighbors = [b for b in cells if b is not self]

    def move1(self, dt):
        ''' verlet velocity integration 
            part 1 '''
        self.v_half = self.vel + self.acc*dt/2
        self.pos = self.pos + self.v_half*dt
    
    def add_force(self, F):
        self.F = self.F + F

    def move2(self, dt):
        ''' verlet velocity integration
            part 2 '''
        self.acc = self.F / self.m
        self.F   = np.zeros(3).astype(double)
        self.vel = self.v_half + self.acc*dt/2

    def return_to_box(self, box):
        for i, x in enumerate(self.pos):
            if self.pos[i] < 0:
                self.pos[i] += box.L[i]
            if self.pos[i] > box.L[i]:
                self.pos[i] -= box.L[i]

    def Ekin(self, N):
        return 1.0/(2.0*N) * mdlibc.dot(self.vel, self.vel)

    def data(self):
        outStr = self.element + ' ' \
               + ' '.join(map(str, self.pos)) + ' '
        '''
               + ' '.join(map(str, self.cell)) + ' ' \
               + str(self.ID) + ' '                  \
               + str(len(self.neighbors)) + ': '
        for neighbor in self.neighbors:
            outStr += str(neighbor.ID) + ' '
        '''
        return outStr

class sim_box:
    def __init__(self, N=np.zeros(3), L=np.zeros(3)):
        self.cells = [[[[] for _ in range(N[0])]
                           for _ in range(N[1])]
                           for _ in range(N[2])]
        self.N = N
        self.L = L

    def insert(self, a):
        index = a.cell
        #sys.stderr.write(' '.join(map(str, a.pos )) + '\n')
        #sys.stderr.write(' '.join(map(str, a.cell)) + '\n')
        self.cells[index[0]][index[1]][index[2]].append(a)
       
    def reset(self):
        self.cells = [[[[] for _ in range(self.N[0])]
                           for _ in range(self.N[1])]
                           for _ in range(self.N[2])]

def calc_lattice(s, e, N):
    # this should be made not a cube
    a = np.indices((N, N, N))
    b = a/(N-1)*(e-s)+s
    return b.T.reshape(-1, 3)

def get_neighboring_indices(x, y, z, N):
    a = np.indices((3, 3, 3))
    b = a.T.reshape(-1, 3)
    c = b + np.array([x-1, y-1, z-1])

    for row in c:
        for i, e in enumerate(row):
            row[i] = row[i] % N[i]
    
    return c

def zero_vcm(group):
    v_cm = np.zeros(3)
    for a in group:
        v_cm += a.vel
    v_cm = 1/len(group) * v_cm
    for i, a in enumerate(group):
        group[i].vel += v_cm
    return
