from Tail import *
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(1000,5000).astype(np.float64)
tail = Tail()
tail.sigma.init_value = 10.
tail.tau.init_value = 100.
tail.mu.init_value = 2000.

p0 = get_p0()

iters = 100000

y = np.zeros(shape=(iters,x.shape[0]),dtype=np.float64)

import time

t0 = time.time()

for k in range(iters):
    tail.evaluate(x,p0,y[k,:])

t1 = time.time()


print((t1-t0)/iters/x.shape[0])

for k in range(1,iters):
    assert((y[k-1,:] == y[k,:]).all())