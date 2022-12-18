#!/usr/bin/python

import time

start = time.time()

import numpy as np
import cupy as cp
from cupyx.scipy.sparse import diags
from cupyx.scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from operators import *
from plot_figures import *


v = cp.loadtxt('model.marmousi')


model = domain(v)

boundary = PML(model)

src_par = source(model)

wavefields = forward_operator(model, src_par, boundary)

temp1, temp2, temp3 = wavefields["P"].shape
P = wavefields["P"].get()
PP = P.reshape(temp1*temp3, temp2)
np.savetxt('/scratch/mssujith/wavefields.freq', PP)

seismogram = freq2time(wavefields, src_par)

convert2su(seismogram)


plot_wavefields(wavefields, src_par, model)

plot_seismogram(seismogram, model)


end = time.time()

print('*'*100)
print()
print('END OF MODELING')
print()
print(f'Elapsed time   ::   {(end-start)//60} minutes and {np.round((end-start)%60, 2)} seconds')
print()
print('*'*100)
