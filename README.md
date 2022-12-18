[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

# 2D Acoustic Wave Propagation on GPU (Using cupy)
## acoustic wave propagation in frequency domain using 13 point stencils

The wave propagation modeling is created using the fourth-order staggered-grid finite difference approximation of the scalar wave equation resulting in 13-stencils. The modeling is performed in frequency domain for discreet frequencies. For stability and for avoiding numerical error, there should be at least 4 gridpoints per minimum wavelength. The left, right and bottom boundaries are incorporated with Perfectly Matching Layers (PMLs) to suppress undesired reflections from edges. The top boundary is left as free surface.


## Reference

Bernhard Hustedt, Stéphane Operto, Jean Virieux, Mixed-grid and staggered-grid finite-difference methods for frequency-domain acoustic wave modelling, Geophysical Journal International, Volume 157, Issue 3, June 2004, Pages 1269–1296, https://doi.org/10.1111/j.1365-246X.2004.02289.x
