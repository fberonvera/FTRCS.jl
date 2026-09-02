# FTRCS.jl

`FTRCS.jl` is a Julia implementation of the finite-time rotational coherent set (FTRCS) framework for two-dimensional, time-dependent velocity fields.

The framework combines finite-time transport coherence, identified using the inflated dynamic Laplacian (IDL), with intrinsic material rotation, diagnosed using the Lagrangian-averaged vorticity deviation (LAVD). Sparse eigenbasis approximation (SEBA) is used to localize coherent-set candidates within the leading IDL eigenspace.

The method and its application to photospheric supergranulation are described in:

**F. J. Beron-Vera, "Quasi-material finite-time rotationally coherent sets in photospheric supergranulation," submitted to *Physics of Plasmas* (2026).**

arXiv:2606.17108  
https://arxiv.org/abs/2606.17108

## Contributors

- Francisco J. Beron-Vera

- Maria Josefina Olascoaga


## Overview

Finite-time flow organization need not be rotational. A region may remain coherent because it rotates as a material body, but coherent transport can also arise through persistent contraction, convergence, or other nonrotational mechanisms. This distinction is particularly relevant in strongly compressible flows such as photospheric supergranulation.

FTRCS separates two aspects of finite-time organization:

1. **Finite-time transport coherence**, identified using the inflated dynamic Laplacian (IDL).
2. **Intrinsic rotational organization**, diagnosed using LAVD.

The IDL identifies quasi-material coherent regions that may form, persist for only part of the observation window, or lose coherence before the end of the interval. SEBA is applied to the leading IDL eigenspace to obtain localized coherent-set candidates.

The rotational character of these candidates is then assessed using their accumulated LAVD. This allows rotationally coherent structures to be distinguished from coherent sets associated primarily with contraction, convergence, or other nonrotational transport mechanisms.

The basic computational sequence is therefore

```text
velocity field
     |
     v
flow map and Cauchy--Green tensor
     |
     v
inflated dynamic Laplacian
     |
     v
leading IDL eigenspace
     |
     v
SEBA localization
     |
     v
coherent-set candidates
     |
     v
LAVD classification
     |
     v
finite-time rotational coherent sets
```


## Repository structure

```text
src/
    FTRCS.jl

scripts/
    run_FTRCS_gom.jl
    run_FTRCS_supergranule.jl

matlab/
    generate_supergranule_synthetic.m
    supergranule_synthetic_parameters.mat
    plot_gom.m
    plot_supergranule.m

data/
    aviso_20130518_20140421.nc

runs/
    generated output

Project.toml
Manifest.toml
README.md
```

The main numerical implementation is entirely in Julia.

`src/FTRCS.jl` contains the reusable FTRCS routines. Application-specific choices are kept in the run scripts under `scripts/`.

The MATLAB files are auxiliary:

- `generate_supergranule_synthetic.m` reconstructs the synthetic supergranule-like velocity dataset used by the second example;
- `supergranule_synthetic_parameters.mat` contains the fixed parameters defining that synthetic realization;
- `plot_gom.m` visualizes the Gulf of Mexico output;
- `plot_supergranule.m` visualizes the synthetic-supergranule output.

MATLAB is not required for the FTRCS calculations themselves.


## Velocity-data convention

The run scripts read two-dimensional time-dependent velocity fields from NetCDF.

The common external convention is

```text
x       spatial coordinate                 [km]
y       spatial coordinate                 [km]
t       time                               [h]

u       x velocity component               [km/h]
v       y velocity component               [km/h]
```

Velocity arrays in the NetCDF files are stored as

```text
u(y,x,t), v(y,x,t)  ->  [Ny Nx Nt]
```

The run scripts convert these arrays once using

```julia
u = permutedims(u,(2,1,3))
v = permutedims(v,(2,1,3))
```

before constructing the internal velocity field.

Inside `FTRCS.jl`, the invariant convention is

```text
u(x,y,t), v(x,y,t)  ->  [Nx Ny Nt]

dimension 1 -> x
dimension 2 -> y
dimension 3 -> t
```

All velocity interpolation, trajectory integration, spatial differentiation, Cauchy--Green calculations, vorticity, and LAVD calculations use this internal convention.


## Installation

The Julia environment is specified by `Project.toml` and `Manifest.toml`.

Clone the repository, enter its root directory, and instantiate the environment once:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Application scripts are then executed from the repository root. For example,

```bash
julia --project=. --threads=18 scripts/run_FTRCS_gom.jl
```

The number of Julia threads may be changed according to the available hardware.


## Computational pipeline

For a prescribed velocity field, the code performs the following principal steps.

### 1. Flow-map integration

Trajectories are integrated over the prescribed finite-time analysis window on a user-defined spatial grid.

Two trajectory integrators are available.

Adaptive Dormand--Prince:

```julia
flow_integrator = :DP5
```

with user-specified relative and absolute tolerances, or fixed-step classical Runge--Kutta:

```julia
flow_integrator = :RK4
```

with a prescribed timestep.

The trajectory ensemble is divided into vectorized batches and distributed across Julia threads.

### 2. Cauchy--Green tensor

Spatial derivatives of the flow map are used to construct

```text
DF = [ X_x  X_y
       Y_x  Y_y ]
```

and the finite-time Cauchy--Green strain tensor

```text
C = DF' * DF.
```

The flow-map/Cauchy--Green spatial resolution is specified independently of the velocity-data grid.

### 3. Inflated dynamic Laplacian

The inflated dynamic Laplacian is assembled on a space-time finite-element grid.

The IDL/FEM spatial and temporal resolutions may be coarser than the flow-map grid and are specified independently in the run script.

### 4. IDL eigenspectrum

Leading IDL eigenpairs are computed over a user-defined sequence of inflation parameters,

```julia
a_values = [...]
```

using a prescribed number of modes,

```julia
nmodes = ...
```

The implementation reports the spatial and material contributions to the IDL Rayleigh quotient and uses their balance to diagnose a spatial/material crossover value of the inflation parameter.

### 5. SEBA localization

Sparse eigenbasis approximation is applied to the leading IDL eigenspace to obtain localized coherent-set candidates.

The number of retained IDL modes controls the dimension of the spectral space available to SEBA and should therefore be regarded as a spectral-resolution parameter.

### 6. LAVD

Relative vorticity is computed as

```text
omega = dv/dx - du/dy
```

and accumulated along trajectories to obtain the Lagrangian-averaged vorticity deviation.

### 7. Rotational classification

IDL--SEBA candidates are not required to be rotational.

Their rotational enrichment is measured by

```text
           mean LAVD within candidate
E_LAVD =  ----------------------------
             mean LAVD over domain
```

and the default FTRCS classification uses

```text
E_LAVD > 1.
```

Thus the IDL--SEBA stage identifies finite-time coherent sets, while the LAVD stage distinguishes those exhibiting enhanced accumulated intrinsic rotation.


## Application parameters

All quantities intended to be changed by the user are specified in the application run scripts rather than in `src/FTRCS.jl`.

These include:

- input and output files;
- analysis domain;
- initial time and analysis duration;
- flow-map/Cauchy--Green spatial resolution;
- IDL/FEM spatial resolution;
- IDL/FEM temporal resolution;
- trajectory integrator;
- fixed timestep or adaptive-solver tolerances;
- trajectory batch size and parallel settings;
- inflation-parameter range;
- number of IDL eigenmodes;
- eigensolver parameters;
- SEBA thresholding parameters; and
- LAVD classification threshold.


## Example 1: Gulf of Mexico altimetry

The Gulf of Mexico example uses the included AVISO altimetry dataset

```text
data/aviso_20130518_20140421.nc
```

and is configured by

```text
scripts/run_FTRCS_gom.jl
```

The run script selects a regional Gulf of Mexico domain and finite-time analysis interval, converts the geographic velocity data to the Cartesian coordinates and units used internally by FTRCS, and constructs a wet-domain mask to accommodate land boundaries.

Run the example from the repository root with

```bash
julia --project=. --threads=18 scripts/run_FTRCS_gom.jl
```

The resulting NetCDF output is written to `runs/`.

The corresponding MATLAB visualization script is

```text
matlab/plot_gom.m
```


## Example 2: synthetic supergranule-like flow

The second example uses a synthetic, strongly compressible cellular velocity field with spatial and temporal characteristics qualitatively resembling photospheric supergranulation.

The synthetic field is not a reconstruction of the observational velocity field used in the accompanying paper. It is provided as a freely reproducible test velocity field on which the complete FTRCS pipeline can be exercised.

The velocity is constructed from time-dependent Gaussian potential and rotational components. The potential component produces strong horizontal convergence and divergence, while the weaker rotational component introduces localized intrinsic rotation.

The large NetCDF velocity file is not stored in the repository. Instead, it is reconstructed locally from a small fixed parameter set.

In MATLAB, run

```matlab
matlab/generate_supergranule_synthetic.m
```

using the accompanying parameter file

```text
matlab/supergranule_synthetic_parameters.mat
```

The generator creates

```text
data/supergranule_synthetic.nc
```

with

```text
x,y       km
t         h
u,v       km/h

size(u) = size(v) = [Ny Nx Nt].
```

The generator also displays the synthetic velocity field through vorticity and streamline snapshots as a visual check.

The FTRCS calculation is then run from the repository root with

```bash
julia --project=. --threads=18 scripts/run_FTRCS_supergranule.jl
```

The corresponding MATLAB visualization script is

```text
matlab/plot_supergranule.m
```


## Output

Application scripts write their NetCDF results to

```text
runs/
```

unless another location is specified in the corresponding run script.

The output contains the quantities required for subsequent visualization and analysis, including the IDL--SEBA candidate masks, candidate diagnostics, LAVD classification information, and the spatial and temporal grids used by the calculation.

Generated files under `runs/` are not tracked by Git.


## Notes on interpretation

IDL--SEBA and LAVD address related but distinct aspects of finite-time flow organization.

IDL--SEBA identifies regions that are coherent from the standpoint of finite-time transport. LAVD measures accumulated intrinsic material rotation. Consequently, not every IDL--SEBA coherent set is expected to satisfy the rotational criterion.

Similarly, IDL--SEBA boundaries should not in general be interpreted as equivalent to material-vortex boundaries obtained from other definitions, such as geodesic vortex boundaries. The methods optimize different notions of finite-time coherence.

The Gulf of Mexico and synthetic supergranule-like examples illustrate the framework in dynamically different settings, including an ocean flow with land boundaries and a strongly compressible cellular flow.


## Reference

If you use this implementation, please refer to:

F. J. Beron-Vera,  
**"Quasi-material finite-time rotationally coherent sets in photospheric supergranulation,"**  
submitted to *Physics of Plasmas*, 2026.  
arXiv:2606.17108.

https://arxiv.org/abs/2606.17108


## Status

This repository accompanies ongoing research and is currently under development. Numerical parameters and interfaces may continue to evolve as the implementation is tested across additional velocity fields.