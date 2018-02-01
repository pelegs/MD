My attempt at writing a simple Molecular Dynamics simulation of Argon gas, using verlet velocity integration, unitless normalized Lennard-Jones potential, neighbor cells and neighbor lists, and periodic boundry conditions.

It is written in Python utilyzing NumPy, while the "heavy" calculations are made in Cython as to accelerate them.
