#################################
# MD simulation of Argon atoms  #
# Written by Peleg Bar Sapir    #
#################################

import numpy as np
from mdlib import *
import mdlibc
import sys

N_atoms = 7 
max_time = 1000
dt = 0.01 
sigma = 1.0
L = 50.0
N_cells = int(np.floor(L/(sigma*2.5)))
box = sim_box(N = 3*[N_cells],
              L = 3*[L])
grid = calc_lattice(0.1, 0.9*L, N_atoms)
atoms = [atom(pos = grid[i],
              box = box,
              element = 'Ar',
              ID  = i)
         for i in range(N_atoms**3)]

#Ek = np.random.normal(3.0, 1.5)
Ek = 13.37
sys.stderr.write('T = ' + str(Ek) + '\n')
for a in atoms:
    a.vel = np.array(3*[2.5]) - a.pos
set_Ek(atoms, Ek)
zero_vcm(atoms)
for a in atoms:
    box.insert(a)
    a.set_neighbors(box)

for t in range(0, max_time):
    Ek = sum([a.Ekin(len(atoms)) for a in atoms])
    sys.stderr.write('\rt = ' + str(t) + ': T = ' + str(Ek) + ', ' + str(len(atoms)) + '   ')
    print(len(atoms))
    print('Test')

    for a in atoms:
        a.move1(dt)
        a.return_to_box(box)
        print(a.data(element   = True,
                     pos       = True,
                     vel       = False,
                     vel_mag   = False,
                     cells     = False,
                     ID        = False,
                     neighbors = False))
        for b in a.neighbors:
            F = mdlibc.LJ_force(a.pos, b.pos,
                                box.L[0], box.L[1], box.L[2])
            a.add_force(F)
            print('', end='')
    
    for a in atoms:
        a.move2(dt)

    for a in atoms:
        box.insert(a)
        a.set_neighbors(box)

sys.stderr.write('\n')
sys.exit()
