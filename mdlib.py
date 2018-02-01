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

    def Ekin(self, N):
        return 1.0/(2.0*N) * mdlibc.dot(self.vel, self.vel)

    def data(self):
        outStr = ' '.join(map(str, self.pos))
        return outStr
