import random
import matplotlib.pyplot as plt
import numpy as np
from math import floor

def f(I, X_1, X_2, G, M):
    new_I = np.maximum(I - G, 0)
    mask = I[X_1] == I[X_2]
    new_I[X_1] += np.where(mask, 0, M)
    new_I[X_2] += np.where(mask, 0, M)
    return new_I

def f_bis(I, X_1, X_2, G, M, N):
    delta = np.where((X_1 <= I < X_2) | (X_2 <= I < X_1), M, 0)
    return np.minimum(N, np.maximum(0, I - G + delta))

def simulate_I_bis(N, p, q, I_0, T):
    process = [I_0]
    for _ in range(1, T):
        X_1 = np.random.randint(0, N, size=1)
        X_2 = (X_1 + np.random.randint(1, N, size=1)) % N
        M = np.random.binomial(1, q)
        G = np.random.binomial(1, p, N)
        process.append(f_bis(process[-1], X_1, X_2, G, M, N))
    return process

def simulate_I(N, p, q, I_0, T):
    process = [I_0]
    for _ in range(1, T):
        X_1 = np.random.randint(0, N, size=1)
        X_2 = np.random.randint(0, N, size=1)
        while X_2 == X_1:
            X_2 = np.random.randint(0, N, size=1)
        M = np.random.binomial(1, q)
        G = np.random.binomial(1, p, N)
        process.append(f(process[-1], X_1, X_2, G, M))
    return process

T = 2001
N = 2000
p = 0.02
q = 0.9
I_0 = np.random.binomial(1, 0.5, N)

pro = simulate_I(N,p,q,I_0,T)
S = [sum(I) for I in pro]
S = [s/N for s in S]

def SIS_lim(t):
    global S
    global N
    return S[floor(t*N)]

M = np.zeros((2, 2))

n_sub = 100
t_values = np.linspace(0, 1, n_sub)
SIS_values = [SIS_lim(t) for t in t_values]

def find_Kgamma(I, eps = 0.01):
    M = np.zeros((2, 2))
    dIdt_0 = (I[1] - I[0])/eps
    dIdt_0_1 = (I[11]-I[10])/eps
    Y = np.array([dIdt_0, dIdt_0_1])
    M[0][0] = (1-I[0])*I[0]
    M[0][1] = -I[0]
    M[1][0] = (1-I[10])*I[10]
    M[1][1] = -I[10]
    
    M_inv = np.linalg.inv(M)

    return np.dot(M_inv, Y)

X = find_Kgamma(SIS_values, 1/n_sub)
print(X)

def real_I(t):
    global X
    global I_0
    global N
    I_0_s = sum(I_0)/N
    R_0 = X[0]/X[1]
    return (R_0 - 1)/(((1/I_0_s * (R_0-1)) - R_0)*np.exp(-X[1]*(R_0-1)*t) + R_0 )

I_ana = [ real_I(t) for t in t_values]

plt.plot(t_values, SIS_values, label='SIS_lim(t)')
plt.plot(t_values, I_ana, color='red', label='I(t)')

#plt of the normalized stochastic process
#plt.plot(S, color='red')
#plt.scatter(range(0,T), S, color='red', marker='o', s=15)



plt.xlabel('t')
plt.ylabel('I_pop')
plt.legend(loc='upper right')

plt.title('Fake Covid19, etp Donald')

# Show the plot
plt.show()