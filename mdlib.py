import numpy as np
import mdlibc

double = np.float64

class atom:
    def __init__(self,
                 pos  = np.zeros(3),
                 vel  = np.zeros(3),
                 cell = np.zeros(3)):

        self.pos  = pos.astype(double)
        self.vel  = vel.astype(double)
        self.cell = cell.astype(double)
        self.acc  = np.zeros(3).astype(double)
        self.F    = np.zeros(3).astype(double)
        
        self.m      = 1.0
        self.v_half = np.zeros(3).astype(double)

        self.neighbors = []

    def move1(self, dt, L):
        ''' verlet velocity integration 
            part 1 '''
        self.v_half = self.vel + self.acc*dt/2
        self.pos = self.pos + self.v_half*dt
        for i,x in enumerate(self.pos):
            if self.pos[i] < 0:
                self.pos[i] += L
            if self.pos[i] > L:
                self.pos[i] -= L
    
    def add_force(self, F):
        self.F = self.F + F

    def move2(self, dt):
        ''' verlet velocity integration
            part 2 '''
        self.acc = self.F / self.m
        self.F   = np.zeros(3).astype(double)
        self.vel = self.v_half + self.acc*dt/2

    def Ekin(self, N):
        return 1.0/(2.0*N) * mdlibc.dot(self.vel, self.vel)

    def data(self):
        outStr = ' '.join(map(str, self.pos))
        return outStr

def calc_lattice(s, e, N):
    step1 = np.indices((N, N, N))
    step2 = step1/(N-1)*(e-s)+s
    return step2.T.reshape(-1, 3)

#def get_neighboring_indices(x, y, z, N):

