#!/usr/bin/python



import numpy as np
import cupy as cp
from cupyx.scipy.sparse import diags
from cupyx.scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from obspy import read, Trace, Stream, UTCDateTime
from obspy.core import AttribDict
from obspy.io.segy.segy import SEGYTraceHeader, SEGYBinaryFileHeader
from obspy.io.segy.core import _read_segy



def domain(v):

  v1 = v.copy()

  n_pml = 60

  nz, nx = v1.shape[0] + n_pml, v1.shape[1] + 2*n_pml
  dx, dz = 20, 40

  x, z = cp.arange(nx) * dx, cp.arange(nz) * dz

  vel = cp.zeros((nz, nx))

  vel[:-n_pml, n_pml:-n_pml] = v1.copy()

  vel[:-n_pml, :n_pml] = cp.repeat(cp.vstack(v1[:, 0]), n_pml, axis = 1)
  vel[:-n_pml, -n_pml:] = cp.repeat(cp.vstack(v1[:, -1]), n_pml, axis = 1)

  vel1 = vel[-n_pml-1, :]
  vel1.shape = (1, vel1.size)

  vel[-n_pml:, :] = cp.repeat(vel1, n_pml, axis = 0)

  v = vel.copy()
  d = cp.power(vel, .25) * 310 

  K = cp.power(v, 2) * d

  d1 = cp.ones((nz+4, nx+4))
  d1[:-4, 2:-2] = d.copy()

  d2 = d1[-5, 2:-2]
  d2.shape = (1, d2.size)

  d1[-4:, 2:-2] = cp.repeat(d2, 4, axis = 0)
  d1[:, :2] = cp.repeat(cp.vstack(d1[:, 3]), 2, axis = 1)
  d1[:, -2:] = cp.repeat(cp.vstack(d1[:, -3]), 2, axis = 1)

  b = 1/d1 

  bi = cp.zeros((nz+4, nx+3))
  bj = cp.zeros((nz+3, nx+4))

  for i in range(nz+3):
    bj[i, :] = 1/2 * (b[i+1, :] + b[i, :]) 

  for i in range(nx+3):
    bi[:, i] = 1/2 * (b[:, i+1] + b[:, i]) 

  bj = bj[:, 2:-2]
  bi = bi[2:-2, :]

  model = {"nx":nx, "nz":nz, "dx":dx, "dz":dz, "n_pml":n_pml, "vel":v, "den":d, "bulk_modulus":K, "bi":bi, "bj":bj}

  return model




def PML(model):

  nx, nz = model["nx"], model["nz"]
  dx, dz = model["dx"], model["dz"]
  n_pml = model["n_pml"]

  z1 = cp.zeros(nz+2-n_pml)
  z1 = cp.append(z1, cp.arange(n_pml-1, -3, -1)) * dz
  x1 = cp.arange(-2, n_pml)
  x1 = cp.append(x1, cp.zeros(nx-2*n_pml))
  x1 = cp.append(x1, cp.arange(n_pml-1, -3, -1)) * dx

  c_pml_x = cp.zeros(nx+4)
  c_pml_z = cp.zeros(nz+4)

  c_pml1 = 20

  c_pml_x[:n_pml+2] = c_pml1
  c_pml_x[-n_pml-2:] = c_pml1
  c_pml_z[-n_pml-2:] = c_pml1


  Lx = n_pml * dx
  Lz = n_pml * dz

  gx = 1j * c_pml_x * np.cos(np.pi/2 * x1/Lx)
  gz = 1j * c_pml_z * np.cos(np.pi/2 * z1/Lz)
  gz.shape = (nz+4, 1)

  boundary = {"n_pml":n_pml, "gx":gx, "gz":gz}

  return boundary




def src_rec(coord, model):

  nx, nz = model["nx"], model["nz"]
  dx, dz = model["dx"], model["dz"]
  n_pml = model["n_pml"]

  x_src, z_src = coord["x_src"], coord["z_src"]
  x_rec, z_rec = coord["x_rec"], coord["z_rec"]

  Ts = cp.zeros((nz, nx))

  for i in range(len(x_src)):
    si = x_src[i]//dx + n_pml
    sj = z_src//dz

    if x_src[i]%dx == 0 and z_src%dz == 0:
      Ts[sj, si] = 1
      Ts[sj+1, si] = 0
      Ts[sj, si+1] = 0
      Ts[sj+1, si+1] = 0

    if x_src[i]%dx == 0 and z_src%dz != 0:
      Ts[sj, si] = (dz-z_src%dz)/dz
      Ts[sj+1, si] = (z_src%dz)/dz
      Ts[sj, si+1] = 0
      Ts[sj+1, si+1] = 0

    if x_src[i]%dx != 0 and z_src%dz == 0:
      Ts[sj, si] = (dx-x_src[i]%dx)/dx
      Ts[sj+1, si] = 0
      Ts[sj, si+1] = (x_src[i]%dx)/dx
      Ts[sj+1, si+1] = 0

    if x_src[i]%dx != 0 and z_src%dz != 0:
      Ts[sj, si] =  ((dx - x_src[i]%dx)/dx + (dz - z_src%dz)/dz)/2
      Ts[sj, si+1] = ((x_src[i]%dx)/dx + (dz - z_src%dz)/dz)/2
      Ts[sj+1, si] =  ((dx - x_src[i]%dx)/dx + (z_src%dz)/dz)/2
      Ts[sj+1, si+1] =  ((x_src[i]%dx)/dx + (z_src%dz)/dz)/2

  n_rec = len(x_rec)
  Tr = cp.zeros((n_rec, nx*nz))

  for i in range(len(x_rec)):
    ri = int(x_rec[i]//dx) + n_pml
    rj = z_src//dz
        
    Tr1 = cp.zeros((nz, nx))

    if x_rec[i]%dx == 0 and z_rec%dz == 0:
      Tr1[rj, ri] = 1
      Tr1[rj+1, ri] = 0
      Tr1[rj, ri+1] = 0
      Tr1[rj+1, ri+1] = 0

    if x_rec[i]%dx == 0 and z_rec%dz != 0:
      Tr1[rj, ri] = (dz-z_rec%dz)/dz
      Tr1[rj+1, ri] = (z_rec%dz)/dz
      Tr1[rj, ri+1] = 0
      Tr1[rj+1, ri+1] = 0

    if x_rec[i]%dx != 0 and z_rec%dz == 0:
      Tr1[rj, ri] = (dx-x_rec[i]%dx)/dx
      Tr1[rj+1, ri] = 0
      Tr1[rj, ri+1] = (x_rec[i]%dx)/dx
      Tr1[rj+1, ri+1] = 0

    if x_rec[i]%dx != 0 and z_rec%dz != 0:
      Tr1[rj, ri] =  ((dx - x_rec[i]%dx)/dx + (dz - z_rec%dz)/dz)/2
      Tr1[rj, ri+1] = ((x_rec[i]%dx)/dx + (dz - z_rec%dz)/dz)/2
      Tr1[rj+1, ri] =  ((dx - x_rec[i]%dx)/dx + (z_rec%dz)/dz)/2
      Tr1[rj+1, ri+1] =  ((x_rec[i]%dx)/dx + (z_rec%dz)/dz)/2
            
    Tr2 = Tr1.flatten()
    Tr2 = cp.hstack(Tr2)
        
    Tr[i, :] = Tr2.copy()

  return Ts, Tr




def source(model):

  Fp = 3
  df = .05
  fmax = 12
  F = cp.arange(df, fmax+df, df)
  src = (2/cp.sqrt(np.pi))  *  (F**2/Fp**3)  *  cp.exp(-((F**2/Fp**2)))

  nx, nz = model["nx"], model["nz"]
  dx, dz = model["dx"], model["dz"]
  n_pml = model["n_pml"]

  x_src = [(nx-2*n_pml)*dx//2]
  z_src = 5

  x_rec = cp.arange(dx, int((nx-2*n_pml)*dx), dx)
  z_rec = 10

  coord = {"x_src":x_src, "z_src":z_src, "x_rec":x_rec, "z_rec":z_rec}

  Ts, Tr = src_rec(coord, model)
  n_rec = len(x_rec)

  src_par = {"F":F, "df":df, "src":src, "Ts":Ts, "Tr":Tr, "n_rec":n_rec}

  return src_par




def forward_operator(model, src_par, boundary):

  nx, nz = model["nx"], model["nz"]
  dx, dz = model["dx"], model["dz"]
  n_pml = model["n_pml"]
  K = model["bulk_modulus"]
  bi, bj = model["bi"], model["bj"]

  F, src = src_par["F"], src_par["src"]
  nf = len(F)

  Ts, Tr = src_par["Ts"], src_par["Tr"]
  n_rec = src_par["n_rec"]

  gx1, gz1 = boundary["gx"], boundary["gz"]

  nx_tru, nz_tru = nx - 2*n_pml, nz - n_pml

  P = cp.zeros((nz_tru, nx_tru, nf), dtype = "complex_")
  data = cp.zeros((nf, n_rec), dtype = "complex_")

  for _k in range(nf):

    w = 2 * cp.pi * F[_k]

    gx = 1 + gx1/w
    gz = 1+ gz1/w

    gxi = 1/2 * (gx[1:] + gx[:-1])
    gzj = 1/2 * (gz[1:] + gz[:-1])

    C1 = (w**2/K) - (1/(gx[2:-2] * dx**2)) * ((1/24 * 1/24 * (bi[:, :-3]/gxi[:-3] + bi[:, 3:]/gxi[3:])) + (9/8 * 9/8 * (bi[:, 1:-2]/gxi[1:-2] + bi[:, 2:-1]/gxi[2:-1]))) -\
              (1/(gz[2:-2] * dz**2)) * ((1/24 * 1/24 * (bj[:-3, :]/gzj[:-3] + bj[3:, :]/gzj[3:])) + (9/8 * 9/8 * (bj[1:-2, :]/gzj[1:-2] + bj[2:-1, :]/gzj[2:-1])))

    C2 = (1/(gx[2:-2] * dx**2)) * (1/24 * 1/24 * bi[:, :-3]/gxi[:-3])
    C3 = (-1/(gx[2:-2] * dx**2)) * (9/8 * 1/24 * (bi[:, 1:-2]/gxi[1:-2] + bi[:, :-3]/gxi[:-3]))
    C4 = (1/(gx[2:-2] * dx**2)) * ((9/8 * 1/24 * (bi[:, 2:-1]/gxi[2:-1] + bi[:, :-3]/gxi[:-3])) + (9/8 * 9/8 * bi[:, 1:-2]/gxi[1:-2]))
    C5 = (1/(gx[2:-2] * dx**2)) * ((9/8 * 1/24 * (bi[:, 1:-2]/gxi[1:-2] + bi[:, 3:]/gxi[3:])) + (9/8 * 9/8 * bi[:, 2:-1]/gxi[2:-1]))
    C6 = (-1/(gx[2:-2] * dx**2)) * (9/8 * 1/24 * (bi[:, 2:-1]/gxi[2:-1] + bi[:, 3:]/gxi[3:]))
    C7 = (1/(gx[2:-2] * dx**2)) * (1/24 * 1/24 * bi[:, 3:]/gxi[3:])
    C8 = (1/(gz[2:-2] * dz**2)) * (1/24 * 1/24 * bj[:-3, :]/gzj[:-3])
    C9 = (-1/(gz[2:-2] * dz**2)) * (9/8 * 1/24 * (bj[1:-2, :]/gzj[1:-2] + bj[:-3, :]/gzj[:-3]))
    C10 = (1/(gz[2:-2] * dz**2)) * ((9/8 * 1/24 * (bj[2:-1, :]/gzj[2:-1] + bj[:-3, :]/gzj[:-3])) + (9/8 * 9/8 * bj[1:-2, :]/gzj[1:-2]))
    C11 = (1/(gz[2:-2] * dz**2)) * ((9/8 * 1/24 * (bj[1:-2, :]/gzj[1:-2] + bj[3:, :]/gzj[3:])) + (9/8 * 9/8 * bj[2:-1, :]/gzj[2:-1]))
    C12 = (-1/(gz[2:-2] * dz**2)) * (9/8 * 1/24 * (bj[2:-1, :]/gzj[2:-1] + bj[3:, :]/gzj[3:]))
    C13 = (1/(gz[2:-2] * dz**2)) * (1/24 * 1/24 * bj[3:, :]/gzj[3:])

    C1 = C1.flatten()   # (i, j)
    C2 = C2.flatten()   # (i-3, j)
    C3 = C3.flatten()   # (i-2, j)
    C4 = C4.flatten()   # (i-1, j)
    C5 = C5.flatten()   # (i+1, j)
    C6 = C6.flatten()   # (i+2, j)
    C7 = C7.flatten()   # (i+3, j)
    C8 = C8.flatten()   # (i, j-3)
    C9 = C9.flatten()   # (i, j-2)
    C10 = C10.flatten() # (i, j-1)
    C11 = C11.flatten() # (i, j+1)
    C12 = C12.flatten() # (i, j+2)
    C13 = C13.flatten() # (i, 2j+3)

    M = diags([C8[3*nx:], C9[2*nx:], C10[nx:], C2[3:], C3[2:], C4[1:], C1, C5, C6, C7, C11, C12, C13], [-3*nx, -2*nx, -nx, -3, -2, -1, 0, 1, 2, 3, nx, 2*nx, 3*nx], shape = (nx*nz, nx*nz), format = 'csr')

    s = Ts * src[_k]
    s = cp.vstack(s.flatten())

    p1 = spsolve(M, s)
    data[_k, :] = cp.matmul(Tr, p1)
    p = p1.reshape(nz, nx)[:-n_pml, n_pml:-n_pml]
    # data[_k, :] = cp.matmul(Tr, cp.vsatck(p.flatten()))
    P[:, :, _k] = p.copy()

  wavefields = {"P":P, "data":data}

  return wavefields




def freq2time(wavefields, src_par):

  data = wavefields["data"]
  nf, n_rec = data.shape
  F, df = src_par["F"], src_par["df"]
  W = 2 * cp.pi * cp.vstack(F)

  tmax = 5
  dt = .001
  t = cp.arange(0, tmax, dt)
  nt = len(t)

  seis = cp.zeros((nt, n_rec))

  for _i in range(nt):
    seis1 = 1/(cp.sqrt(2*cp.pi)) * cp.sum(data * cp.exp(-1j * W * t[_i]) * df, axis = 0)
    seis[_i, :] = seis1.real

  seismogram = {"t":t, "seis":seis}
  return seismogram




def convert2su(seismogram):

  stream = Stream()
  seis_data = seismogram["seis"].get()

  nt, n_rec = seis_data.shape

  for _i in range(n_rec):
    data = seis_data[:, _i]
    data = np.require(data, dtype = np.float32)
    trace = Trace(data = data)

    trace.stats.delta = 0.001

    trace.stats.starttime = UTCDateTime(2011,11,11,11,11,11)

    if not hasattr(trace.stats, 'segy.trace_header'):
        trace.stats.segy = {}
    trace.stats.segy.trace_header = SEGYTraceHeader()
    trace.stats.segy.trace_header.trace_sequence_number_within_line = _i + 1
    trace.stats.segy.trace_header.original_field_record_number =  1
    trace.stats.segy.trace_header.receiver_group_elevation = 444

    stream.append(trace)

  stream.stats = AttribDict()
  stream.stats.textual_file_header = 'Textual Header!'
  stream.stats.binary_file_header = SEGYBinaryFileHeader()
  stream.stats.binary_file_header.trace_sorting_code = 5

  stream.write("./seismogram.su", format="SU")

