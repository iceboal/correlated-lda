#!/usr/bin/env python
import numpy as np

N = 10000
K = 1000

U = np.zeros((N, K+1), dtype=np.float32)
V = np.zeros((N, K+1), dtype=np.float32)
X = np.zeros((N, K), dtype=np.float32)

for n in range(1, N):
    U[n, 1] = n

for k in range(2, K+1):
    if k % 100 == 0:
        print k
    V[k, k] = 1.0 / U[k-1, k-1]
    U[k, k] = U[k-1, k-1] + k
    for n in range(k+1, N):
        V[n, k] = (1 + (n-1) * V[n-1, k]) / U[n-1, k-1]
        U[n, k] = 1.0 / V[n, k] + n

for k in range(K):
    for n in range(N):
        X[n, k] = U[n, k+1] * V[n, k+1]
for n in range(min((N, K))):
    X[n, n] = 1

U = U[:,:K]

np.save('U.npy', U)
np.save('X.npy', X)
