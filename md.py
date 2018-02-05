#################################
# MD simulation of Argon atoms  #
# Written by Peleg Bar Sapir    #
#################################

import numpy as np
from mdlib import *
import mdlibc
import sys
import argparse

parser = argparse.ArgumentParser(description="A simple MD simulation of Argon gas")
parser.add_argument('-o','--output',
                    help='Output file name',
                    required=True)
parser.add_argument('-n','--numatoms',
                    help='Number of atoms in simulation (WILL BE CUBED!)',
                    type=int,
                    required=True)
parser.add_argument('-l','--length',
                    help='Side length of the simulation box',
                    type=float,
                    default = 100.0,
                    required=False)
parser.add_argument('-t','--maxtime',
                    help='Maximum time step',
                    type=int,
                    default = 100,
                    required=False)
parser.add_argument('-k','--temperature',
                    help='Temperature',
                    type=float,
                    default = 2.5,
                    required=False)
parser.add_argument('-dt','--deltatime',
                    help='Time step',
                    type=float,
                    default = 0.05,
                    required=False)
parser.add_argument('-c','--comment',
                    help='Comment for xyz file',
                    required=False)
parser.parse_args()
args = vars(parser.parse_args())

N_atoms = args['numatoms'] 
max_time = args['maxtime']
dt = args['deltatime']
T = args['temperature']
L = args['length']
comment = args['comment']
N_cells = int(np.floor(L/2.5))
box = sim_box(N = 3*[N_cells],
              L = 3*[L])
grid = calc_lattice(0.1, 0.9*L, N_atoms)
atoms = [atom(pos = grid[i],
              box = box,
              element = 'Ar',
              ID  = i)
         for i in range(N_atoms**3)]
sys.stderr.write('T = ' + str(T) + '\n')
set_temperature(atoms, T)
zero_vcm(atoms)

for a in atoms:
    box.insert(a)
    a.set_neighbors(box)

for t in range(0, max_time):
    Ek = sum([a.Ekin(len(atoms)) for a in atoms])
    sys.stderr.write('\rt = ' + str(t) + ': T = ' + str(2/3*Ek/N_atoms) + '    ')
    print(len(atoms))
    print(comment + ', time = ' + str(t))

    for a in atoms:
        a.move1(dt)
        a.return_to_box(box)
        print(a.data(element   = True,
                     pos       = True,
                     vel       = False,
                     vel_mag   = False,
                     cells     = True,
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
