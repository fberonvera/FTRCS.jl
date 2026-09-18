# FTRCS.jl

`FTRCS.jl` is a Julia implementation of the finite-time rotational
coherent set (FTRCS) framework for two-dimensional, time-dependent
velocity fields.

The framework combines finite-time transport coherence, identified using
the inflated dynamic Laplacian (IDL), with intrinsic material rotation,
diagnosed using the Lagrangian-averaged vorticity deviation (LAVD).
Sparse eigenbasis approximation (SEBA) is used to localize features
within the leading IDL eigenspace, and a subpartition-of-unity threshold
is used to extract their high-membership supports.

The method and its application to photospheric supergranulation are
described in:

**F. J. Beron-Vera, "Quasi-material finite-time rotationally coherent
sets in photospheric supergranulation," submitted to *Physics of
Plasmas* (2026).**

arXiv:2606.17108\
https://arxiv.org/abs/2606.17108

------------------------------------------------------------------------

## Contributors

-   Francisco J. Beron-Vera

-   Maria Josefina Olascoaga

------------------------------------------------------------------------

## Overview

Finite-time flow organization need not be rotational. A region may
remain coherent because it rotates as a material body, but coherent
transport can also arise through persistent contraction, convergence, or
other nonrotational mechanisms. This distinction is particularly
relevant in strongly compressible flows such as photospheric
supergranulation.

FTRCS separates two aspects of finite-time organization:

1.  **Finite-time transport coherence**, identified using the inflated
    dynamic Laplacian (IDL).
2.  **Intrinsic rotational organization**, diagnosed using LAVD.

The IDL identifies quasi-material coherent regions that may form,
persist for only part of the observation window, or lose coherence
before the end of the interval. For an analysis window `[t0,t0+T]`, the
IDL calculation is restricted to the material survival domain

``` text
W_T = {x0 in D : x(t;x0) remains in D throughout [t0,t0+T]}.
```

SEBA is applied to the leading IDL eigenspace to obtain sparse localized
membership functions. Their positive parts are hard-thresholded using a
global subpartition-of-unity threshold, producing high-membership
coherent-set supports. Consecutive active time slices define
finite-lifetime FTCS episodes with their own birth and death times
`[ta,tb]`.

The rotational character of each episode is then assessed using LAVD
accumulated only over its own lifetime. This allows rotationally
coherent structures to be distinguished from coherent sets associated
primarily with contraction, convergence, or other nonrotational
transport mechanisms.

The basic computational sequence is therefore

``` text
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
subpartition-of-unity supports
     |
     v
finite-lifetime FTCS episodes
     |
     v
lifetime-specific LAVD classification
     |
     v
finite-time rotational coherent sets
```

------------------------------------------------------------------------

## Repository structure

``` text
src/
    FTRCS.jl

scripts/
    run_FTRCS_hinode_synthetic.jl

matlab/
    hinode_synthetic_nc.m
    hinode_synthetic_parameters.mat
    plot_hinode_synthetic.m

data/
    hinode_synthetic.nc          generated locally; not tracked

runs/
    generated output

Project.toml
Manifest.toml
README.md
```

The main numerical implementation is entirely in Julia.

`src/FTRCS.jl` contains the reusable FTRCS routines. The Hinode-like
application choices are kept in `scripts/run_FTRCS_hinode_synthetic.jl`.

The public example uses a **synthetic Hinode-like photospheric flow**: a
two-dimensional, time-dependent horizontal velocity field designed to
reproduce the cellular, strongly compressible character of solar
photospheric supergranulation without distributing the observational
Hinode velocity data used in the research analysis.

The MATLAB files are auxiliary:

-   `hinode_synthetic_nc.m` generates the synthetic Hinode-like
    photospheric velocity field using the parameters stored in
    `hinode_synthetic_parameters.mat`;
-   `plot_hinode_synthetic.m` visualizes the FTRCS output produced by
    the Julia calculation.

MATLAB is required only to generate and inspect the synthetic test
dataset and for the supplied visualization. The FTRCS calculations
themselves are performed in Julia.

------------------------------------------------------------------------

## Velocity-data convention

The run scripts read two-dimensional time-dependent velocity fields from
NetCDF.

The common external convention is

``` text
x       spatial coordinate                 [km]
y       spatial coordinate                 [km]
t       time                               [h]

u       x velocity component               [km/h]
v       y velocity component               [km/h]
```

Velocity arrays in the NetCDF files are stored as

``` text
u(y,x,t), v(y,x,t)  ->  [Ny Nx Nt]
```

The run scripts convert these arrays once using

``` julia
u = permutedims(u,(2,1,3))
v = permutedims(v,(2,1,3))
```

before constructing the internal velocity field.

Inside `FTRCS.jl`, the invariant convention is

``` text
u(x,y,t), v(x,y,t)  ->  [Nx Ny Nt]

dimension 1 -> x
dimension 2 -> y
dimension 3 -> t
```

All velocity interpolation, trajectory integration, spatial
differentiation, Cauchy--Green calculations, vorticity, and LAVD
calculations use this internal convention.

------------------------------------------------------------------------

## Installation

The Julia environment is specified by `Project.toml` and
`Manifest.toml`.

Clone the repository, enter its root directory, and instantiate the
environment once:

``` bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Application scripts are then executed from the repository root. For
example,

``` bash
julia --project=. --threads=18 scripts/run_FTRCS_hinode_synthetic.jl
```

The number of Julia threads may be changed according to the available
hardware.

## Computational pipeline

For a prescribed velocity field, the code performs the following
principal steps.

### 1. Flow-map integration

Trajectories are integrated over the prescribed finite-time analysis
window on a user-defined rectangular initial-condition grid.

The analysis rectangle may be smaller than the velocity-data domain.
This allows trajectories to be initialized only in the region where
FTRCS are to be computed while still using velocity data from a larger
surrounding domain. Trajectories may leave the analysis rectangle during
the integration; they remain admissible as long as they remain inside
the velocity-data domain.

The flow-map routine accepts the optional keyword

``` julia
seed_bounds = (
    (xmin,xmax),
    (ymin,ymax)
)
```

with bounds expressed in the same Cartesian coordinates as the velocity
field. Setting

``` julia
seed_bounds = nothing
```

uses the complete velocity-data domain, preserving the original
full-domain behavior.

Two trajectory integrators are available.

Adaptive Dormand--Prince:

``` julia
flow_integrator = :DP5
```

with user-specified relative and absolute tolerances, or fixed-step
classical Runge--Kutta:

``` julia
flow_integrator = :RK4
```

with a prescribed timestep.

The trajectory ensemble is divided into vectorized batches and
distributed across Julia threads.

### 2. Cauchy--Green tensor

Spatial derivatives of the flow map are used to construct

``` text
DF = [ X_x  X_y
       Y_x  Y_y ]
```

and the finite-time Cauchy--Green strain tensor

``` text
C = DF' * DF.
```

The flow-map/Cauchy--Green spatial resolution is specified independently
of the velocity-data grid.

### 3. Inflated dynamic Laplacian

The inflated dynamic Laplacian is assembled on a space-time
finite-element grid.

The IDL/FEM spatial and temporal resolutions may be coarser than the
flow-map grid and are specified independently in the run script.

### 4. IDL eigenspectrum

Leading IDL eigenpairs are computed over a user-defined sequence of
inflation parameters,

``` julia
a_values = [...]
```

using a prescribed number of modes,

``` julia
nmodes = ...
```

The implementation reports the spatial and material contributions to the
IDL Rayleigh quotient and uses their balance to diagnose a
spatial/material crossover value of the inflation parameter.

### 5. Material survival domain

The velocity-data domain and the analysis domain are distinct. Initial
conditions are seeded in the chosen analysis rectangle, while their
trajectories are evaluated using the complete velocity-data domain.

For an analysis window `[t0,t0+T]`, the IDL operator is constructed on
the seeded initial conditions whose trajectories remain inside the
velocity-data domain throughout the complete window,

``` text
W_T = {x0 in D_analysis :
       x(t;x0) remains in D_data
       for all t in [t0,t0+T]}.
```

The same original material labels `x0 in W_T` are retained throughout
the IDL, SEBA, support-extraction, and LAVD-classification stages. A
larger velocity-data domain can therefore provide a buffer around a
smaller analysis rectangle and reduce trajectory loss through the data
boundaries. In an open or insufficiently buffered data domain, the
survival domain generally becomes smaller as the analysis window is
increased.

### 6. SEBA localization and subpartition-of-unity supports

Sparse eigenbasis approximation is applied to the leading IDL eigenspace
to obtain sparse localized vectors

``` text
S = [s1 ... sm].
```

SEBA chooses the sign of each vector to make it predominantly
nonnegative and scales each column to have maximum value one. Positive
entries can therefore be interpreted as membership strengths in the
corresponding localized feature.

To obtain hard supports, the code applies the subpartition-of-unity
threshold of Froyland, Rock, and Sakellariou (2019). At each space-time
degree of freedom, the positive SEBA memberships are sorted in
descending order and their cumulative sums are formed. The global
threshold `tau_PU` is the largest membership value associated with a
cumulative sum exceeding one. Values at or below `tau_PU` are removed.
By construction, the retained membership functions satisfy

``` text
sum_j p_j(x,t) <= 1
```

at every space-time point. Some points may therefore remain unassigned.

The number of retained IDL modes controls the dimension of the spectral
space available to SEBA and should therefore be regarded as a
spectral-resolution parameter.

### 7. Finite-lifetime FTCS episodes

A thresholded SEBA support need not remain active throughout the full
analysis window. Consecutive active time slices define a finite-lifetime
FTCS episode with birth and death times `[ta,tb]`. Different episodes
may therefore coexist over different subintervals of the same IDL
analysis window.

### 8. Lifetime-specific LAVD

Relative vorticity is computed as

``` text
omega = dv/dx - du/dy.
```

For an FTCS episode alive on `[ta,tb]`, LAVD is accumulated only over
that episode lifetime while retaining the original material labels
`x0 in W_T`:

``` text
LAVD_[ta,tb](x0)
    = integral_[ta,tb]
      |omega(x(t;x0),t) - <omega>(t)| dt.
```

The full trajectories initialized at `t0` are reused; trajectories are
not reinitialized at the episode birth time `ta`. Thus the SEBA support,
material survival mask, and LAVD field remain indexed by the same
initial conditions `x0`.

### 9. Rotational classification

Finite-lifetime IDL--SEBA episodes are not required to be rotational.
For an episode with birth support `A`, its rotational enrichment is

``` text
                    mean LAVD_[ta,tb] over A
E_LAVD = ------------------------------------------------
         mean LAVD_[ta,tb] over (W_T minus A).
```

The default FTRCS classification uses

``` text
E_LAVD >= 1.
```

Thus the IDL--SEBA/subpartition stage identifies finite-time coherent
set episodes, while the lifetime-specific LAVD stage distinguishes those
exhibiting enhanced accumulated intrinsic rotation.

------------------------------------------------------------------------

## Application parameters

All quantities intended to be changed by the user are specified in the
application run scripts rather than in `src/FTRCS.jl`.

These include:

-   input and output files;
-   velocity-data domain;
-   rectangular analysis/initial-condition domain (`seed_bounds`);
-   initial time and analysis duration;
-   flow-map/Cauchy--Green spatial resolution;
-   IDL/FEM spatial resolution;
-   IDL/FEM temporal resolution;
-   trajectory integrator;
-   fixed timestep or adaptive-solver tolerances;
-   trajectory batch size and parallel settings;
-   inflation-parameter range;
-   number of IDL eigenmodes;
-   eigensolver parameters;
-   SEBA parameters and subpartition-of-unity support extraction; and
-   LAVD rotational-enrichment threshold.

------------------------------------------------------------------------

## Synthetic Hinode-like photospheric example

The public example uses a synthetic horizontal photospheric velocity
field with spatial and temporal characteristics qualitatively resembling
solar supergranulation as observed by Hinode.

The synthetic field is **not** the observational Hinode velocity field
used in the accompanying research. It is a freely reproducible test
field intended to exercise the complete FTRCS pipeline while avoiding
distribution of the observational dataset.

The flow is two-dimensional and time dependent, with cellular horizontal
convergence and divergence characteristic of photospheric
supergranulation and localized rotational motion. It provides a compact
test of trajectory integration, material survival, the IDL,
IDL-eigenspace localization by SEBA, subpartition-of-unity support
extraction, finite-lifetime episode detection, and lifetime-specific
LAVD classification.

The supplied Hinode run script uses the complete synthetic velocity
domain by default,

``` julia
analysis_bounds = nothing
```

but the FTRCS calculation can be restricted to any rectangular
subdomain, for example,

``` julia
analysis_bounds = (
    (10_000.0,40_000.0),
    (15_000.0,35_000.0)
)
```

with distances in km. Only the initial-condition grid is restricted; the
trajectories continue to use the complete synthetic velocity field.

The large synthetic velocity file is not stored in the repository.
Instead, generate it locally in MATLAB using

``` matlab
matlab/hinode_synthetic_nc.m
```

together with the fixed parameter file

``` text
matlab/hinode_synthetic_parameters.mat
```

The generator creates

``` text
data/hinode_synthetic.nc
```

The generated synthetic flow can be inspected before running the FTRCS
calculation using

``` matlab
matlab/plot_hinode_synthetic.m
```

This provides a direct visualization of the synthetic horizontal
photospheric velocity field used by the reproducible example.

with the external convention

``` text
x,y       km
t         h
u,v       km/h

size(u) = size(v) = [Ny Nx Nt].
```

Then run the FTRCS calculation from the repository root:

``` bash
julia --project=. --threads=18 scripts/run_FTRCS_hinode.jl
```

The corresponding MATLAB visualization script is

``` text
matlab/plot_hinode.m
```

The example is intended as a reproducible end-to-end test of the
software, not as a replacement for the observational Hinode analysis.

------------------------------------------------------------------------

## Output

Application scripts write their NetCDF results to

``` text
runs/
```

unless another location is specified in the corresponding run script.

The output contains the quantities required for subsequent visualization
and analysis, including the material survival mask, SEBA fields,
subpartition-of-unity threshold and support masks, finite-lifetime
episode birth/death information, lifetime-specific LAVD classification
diagnostics, and the spatial and temporal grids used by the calculation.

Generated files under `runs/` are not tracked by Git.

------------------------------------------------------------------------

## Notes on interpretation

IDL--SEBA support extraction and LAVD address related but distinct
aspects of finite-time flow organization.

IDL and SEBA identify localized finite-time transport features, while
the subpartition-of-unity threshold defines their high-membership
supports. LAVD measures accumulated intrinsic material rotation over
each support episode's own lifetime. Consequently, not every finite-time
coherent-set episode is expected to satisfy the rotational criterion.

Similarly, IDL--SEBA boundaries should not in general be interpreted as
equivalent to material-vortex boundaries obtained from other
definitions, such as geodesic vortex boundaries. The methods optimize
different notions of finite-time coherence.

The synthetic Hinode-like example is provided to exercise the complete
software pipeline on a reproducible photospheric flow. Observational
Hinode velocity data are not distributed with this repository.

------------------------------------------------------------------------

## Code development

The Julia implementation was adapted from MATLAB codes developed by F.
J. Beron-Vera, with contributions from M. J. Olascoaga to the
translation to Julia. ChatGPT was used to assist with code translation,
debugging, and documentation. The numerical methodology, detection
strategy, and scientific design are those of F. J. Beron-Vera.

------------------------------------------------------------------------

## Status

This repository accompanies ongoing research and is currently under
development. The included synthetic Hinode-like example provides a
reproducible end-to-end test of the current implementation. Numerical
parameters and interfaces may continue to evolve.
