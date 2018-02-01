import numpy as np
from mdlib import atom
import mdlibc

N = 50 
maxT = 1000
dt = 0.005
L = 5.0
sigma = 3.0
atoms = [atom(pos = np.random.uniform(-L, L, size=(1, 3)).flatten(),
              vel = np.random.normal(0.0, sigma, size=(1, 3)).flatten())
         for _ in range(N)]

for t in range(0, maxT):
    print(N)
    print('Test')

    for a in atoms:
        a.move1(dt)
    for a in atoms:
        for b in atoms:
            if a is not b:
                F = mdlibc.LJ_force(a.pos, b.pos)
                a.add_force(F)
    for a in atoms:
        a.move2(dt)
        print('Ar', a.data())
