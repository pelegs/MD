import numpy as np
import mdlibc
from itertools import chain
import sys

double = np.float64

class atom:
    def __init__(self,
                 pos    = np.zeros(3),
                 vel    = np.zeros(3),
                 box    = None):
        
        self.pos  = pos.astype(double)
        self.vel  = vel.astype(double)
        self.acc  = np.zeros(3).astype(double)
        self.F    = np.zeros(3).astype(double)
        
        self.m      = 1.0
        self.v_half = np.zeros(3).astype(double)

        self.cell = np.floor(self.pos / box.L * box.N-2).astype(int) + 3*[2]
        self.neighbors = []

    def set_neighbors(self, box):
        self.neighbors = []
        self.cell = np.floor(self.pos / box.L * box.N-2).astype(int) + 3*[2]
        cells = list(chain(*[box.cells[i][j][k]
                             for (i, j, k) in get_neighboring_indices(self.cell[0],
                                                                      self.cell[1],
                                                                      self.cell[2],
                                                                      box.N[0],
                                                                      box.N[1],
                                                                      box.N[2])]))
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
            if x < 0:
                dL = np.ceil(-x/box.L[i]) * box.L[i]
                self.pos[i] += dL
            if x > box.L[i]:
                dL = np.floor(x/box.L[i]) * box.L[i]
                self.pos[i] -= dL 

    def Ekin(self, N):
        return 1.0/(2.0*N) * mdlibc.dot(self.vel, self.vel)

    def data(self):
        outStr = ' '.join(map(str, self.pos)) + ' '
        outStr += ' '.join(map(str, self.cell)) + ' '
        outStr += str(len(self.neighbors))
        return outStr

class sim_box:
    def __init__(self, N=np.zeros(3), L=np.zeros(3)):
        self.cells = [[[[] for _ in range(N[0]+2)]
                           for _ in range(N[1]+2)]
                           for _ in range(N[2]+2)]
        self.N = N + np.array(3*[2]).astype(int)
        self.L = L

    def insert(self, a):
        index = a.cell
        self.cells[index[0]][index[1]][index[2]].append(a)

    def reset(self):
        self.cells = [[[[] for _ in range(self.N[0]+2)]
                           for _ in range(self.N[1]+2)]
                           for _ in range(self.N[2]+2)]

def calc_lattice(s, e, N):
    # this should be made not a cube
    a = np.indices((N, N, N))
    b = a/(N-1)*(e-s)+s
    return b.T.reshape(-1, 3)

def get_neighboring_indices(x, y, z, Nx, Ny, Nz):
    a = np.indices((3, 3, 3))
    b = a.T.reshape(-1, 3)
    c = b + np.array([x-1, y-1, z-1])

    # filtering by edges
    for i, row in reversed(list(enumerate(c))):
        if -1 in row or row[0] >= Nx or row[1] >= Ny or row[2] >= Nz:
            c = np.delete(c, (i), axis=0)
    
    return c
