#!/usr/bin/python



import numpy as np
import cupy as cp
import matplotlib.pyplot as plt


def plot_wavefields(wavefields, src_par, model, cmap = 'seismic'):

  P = wavefields["P"].get().real
  F = src_par["F"].get()
  nf = len(F)

  nx, nz = model["nx"], model["nz"]
  dx, dz = model["dx"], model["dz"]
  n_pml = model["n_pml"]

  nrows, ncols = 4, 5
  fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize=(20, 12))
  plt.suptitle('Pressure wavefields', fontweight="bold")

  num = nf//(nrows*ncols - 1)

  F = np.round(F, 1)
  _i = 0

  extent = [0, (nx-2*n_pml) * dx, (nz-n_pml) * dz, 0]

  for ax in axes.flat:
    im_data = P[:, :, int(num * _i)]
    pmax = np.sqrt(np.mean(im_data**2))
    im = ax.imshow(im_data.real, cmap = cmap, vmin = -pmax, vmax = pmax, extent = extent) 
    ax.set_title(f'{F[num * _i]} Hz')
    if _i%5 != 0:
      ax.set_yticks([])
    else:
      ax.set_ylabel('Z  (m)')
    if _i <= 14: 
      ax.set_xticks([])
    else:
      ax.set_xlabel('X  (m)')
    _i += 1


  fig.colorbar(im, ax=axes.ravel().tolist())
  plt.savefig('./wavefields_freq.pdf', format = 'pdf',  dpi = 300)





def plot_seismogram(seismogram, model):

  seis, t = seismogram["seis"].get(), seismogram["t"].get()
  nx, dx = model["nx"], model["dx"]
  n_pml = model["n_pml"]

  pmax = np.percentile(seis, 99)
  extent = [0, (nx-2*n_pml) * dx, t[-1], t[0]]

  fig, ax = plt.subplots(figsize = (12, 6))
  fig1 = ax.imshow(seis, cmap = 'gray', aspect = 'auto', extent = extent, vmin = -pmax, vmax = pmax)
  fig.colorbar(fig1)
  ax.set_title('Seismogram')
  ax.set_xlabel('X (m)')
  ax.set_ylabel('Time (s)')

  plt.savefig('./seismogram.pdf', format = 'pdf', dpi = 300)
