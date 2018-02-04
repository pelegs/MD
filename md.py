#################################
# MD simulation of Argon atoms  #
# Written by Peleg Bar Sapir    #
#################################

import numpy as np
from mdlib import *
import mdlibc
import sys

N_atoms = 3
max_time = 300
dt = 0.005
sigma = 1.0
L = 50.0
N_cells = int(np.floor(L/(sigma*2.5)))
box = sim_box(N = 3*[N_cells],
              L = 3*[L])
grid = calc_lattice(0.1, 0.95*L, N_atoms)
atoms = [atom(pos = grid[i],
              vel = np.random.normal(0.0, 50*sigma, size=(1, 3)).flatten(),
              box = box,
              element = 'Ar',
              ID  = i)
         for i in range(N_atoms**3)]

zero_vcm(atoms)

for a in atoms:
    box.insert(a)
    a.set_neighbors(box)

for t in range(0, max_time):
    Ek = sum([a.Ekin(len(atoms)) for a in atoms])
    sys.stderr.write('\rt = ' + str(t) + ': Ek = ' + str(Ek) + ', ' + str(len(atoms)) + '   ')
    print(len(atoms))
    print('Test')

    for a in atoms:
        a.move1(dt)
        a.return_to_box(box)
        print(a.data())
        for b in a.neighbors:
            #F = mdlibc.LJ_force(a.pos, b.pos,
            #                    box.L[0], box.L[1], box.L[2])
            print('', end='')
    
    for a in atoms:
        a.move2(dt)

    for a in atoms:
        box.insert(a)
        a.set_neighbors(box)

sys.stderr.write('\n')
sys.exit()
