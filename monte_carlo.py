import random
import matplotlib.pyplot as plt
import numpy as np
from math import floor
from itertools import product

def f(I,X_1,X_2,G,M):
    N = len(I)
    new_I = [max(I[k]-G[k],0) for k in range(0,N)]
    if I[X_1] == I[X_2]:
        return new_I
    elif I[X_1] == 0:
        new_I[X_1] += M
    else:
        new_I[X_2] += M
    return new_I
    
def simulate_I(N,p,q,I_0,T):
    process = [I_0]
    for k in range(1,T):
        X_1 = random.randint(0, N-1)
        X_2 = random.randint(0, N-1)
        M = np.random.binomial(1,q)
        G = np.zeros(N)
        for j in range(0,N):
            G[j] = np.random.binomial(1,p)
        process.append(f(process[k-1],X_1,X_2,G,M))
    return process


def monte_carlo(x,y,N,p,q, n=10000):
    freq = 0
    for k in range(n):
        if np.array_equal(simulate_I(N, p, q, x, 2)[1], y):
             freq += 1
    return freq/n

def first_col(x,N,p,q):
    s = sum(x)
    a = p*(1-p)
    return (q + ((1-q)/N**2) * (s**2 + (N-s)**2) ) * ((1+a)**(N-s)) * (a**s)

def gen_arr(N):
    return list(product([0, 1], repeat=N))


def P(N, p, q):
    P = np.zeros((2**N, 2**N))
    arr = gen_arr(N)
    for i, x in enumerate(arr):
        for j, y in enumerate(arr):
            P[i, j] = monte_carlo(list(x), list(y), N, p, q, n=200)
    return P


N = 8
p = 0.5
q = 0.5
x = np.random.binomial(1,0.5,N)
y = x[:]
P_m = P(N,p,q)
print(P_m)
print(gen_arr(N))

plt.figure(figsize=(6, 4))
plt.imshow(P_m, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Value')
plt.title('Transition matrix heatmap')
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.grid(False)
plt.show()

