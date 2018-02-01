import numpy as np
from mdlib import *
import mdlibc

N = 5
max_time = 500
dt = 0.005
L = 3.0
sigma = 1.0
grid = calc_lattice(0, L, N)
atoms = [atom(pos = grid[i],
              vel = np.random.normal(0.0, sigma, size=(1, 3)).flatten())
         for i in range(N**3)]

for t in range(0, max_time):
    print(N**3)
    print('Test')

    for a in atoms:
        a.move1(dt, L*2)
    for a in atoms:
        for b in atoms:
            if a is not b:
                F = mdlibc.LJ_force(a.pos, b.pos)
                a.add_force(F)
    for a in atoms:
        a.move2(dt)
        print('Ar', a.data())
