import timeit

def test(what='direct'):
    if what == 'direct':
        return 'mdlibc.distance_direct(vec[0], vec[1])'
    elif what == 'np':
        return 'mdlibc.distance_np(vec[0], vec[1])'
    else:
        return 'np.linalg.norm(vec[1]-vec[0])'

setup = '''
import numpy as np
import mdlibc

vec = np.random.uniform(-1, 1, size=(2,3))
'''

test_direct = test('direct')
test_np     = test('np')
test_pure   = test('pure')

t_direct = timeit.Timer(test_direct, setup).repeat(1000, 10000)
print('Direct:', min(t_direct))

t_np = timeit.Timer(test_np, setup).repeat(1000, 10000)
print('Cython NumPy:', min(t_np))

t_pure = timeit.Timer(test_pure, setup).repeat(1000, 10000)
print('Pure NumPy:', min(t_pure))
