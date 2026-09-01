# FTRCS.jl

`FTRCS.jl` is a Julia implementation of the finite-time rotational coherent set (FTRCS) framework for two-dimensional, time-dependent velocity fields.

The framework is introduced and applied to photospheric supergranulation in:

**F. J. Beron-Vera, "Quasi-material finite-time rotationally coherent sets in photospheric supergranulation," submitted to *Physics of Plasmas* (2026).**

arXiv:2606.17108  
https://arxiv.org/abs/2606.17108

## Overview

Finite-time flow organization involves more than rotation alone. A region may remain coherent because it rotates as a material body, but coherent transport can also arise through persistent contraction, convergence, or other mechanisms. This distinction is particularly relevant in compressible flows such as photospheric supergranulation.

FTRCS separates two aspects of finite-time coherence:

1. **Finite-time transport coherence**, identified using the inflated dynamic Laplacian (IDL).
2. **Intrinsic rotational organization**, diagnosed using the Lagrangian-averaged vorticity deviation (LAVD).

The IDL identifies quasi-material coherent regions that may have finite lifetimes within the analysis interval. Sparse eigenbasis approximation (SEBA) is then applied to the leading IDL eigenspace to obtain spatially localized coherent-set candidates.

The rotational character of these candidates is assessed using LAVD. This distinguishes coherent regions exhibiting enhanced accumulated intrinsic rotation from coherent sets whose organization is primarily associated with nonrotational transport mechanisms.

The resulting FTRCS framework is therefore intended to separate

```text
finite-time transport coherence
              +
intrinsic rotational organization
```

in general time-dependent flows.

For the mathematical formulation, interpretation, and photospheric application, see the accompanying paper.

## Computational pipeline

For a prescribed two-dimensional velocity field, the implementation performs the following main steps:

1. **Flow-map integration**

   Trajectories are integrated over the prescribed finite-time analysis window on a user-defined spatial grid.

2. **Cauchy--Green tensor**

   Spatial derivatives of the flow map are used to construct the finite-time Cauchy--Green strain tensor.

3. **Inflated dynamic Laplacian**

   The IDL is assembled on a space-time finite-element grid. Spatial and material contributions are combined through the inflation parameter `a`.

4. **IDL eigenspectrum**

   Leading eigenpairs are computed for a user-specified range of inflation parameters. The implementation includes an automatic spatial/material crossover diagnostic.

5. **SEBA localization**

   Sparse eigenbasis approximation is applied to the leading IDL eigenspace to obtain localized coherent-set candidates.

6. **LAVD**

   Relative vorticity

   ```text
   omega = dv/dx - du/dy
   ```

   is computed from the velocity field and accumulated along trajectories to obtain LAVD.

7. **FTRCS classification**

   IDL--SEBA candidates are classified according to their LAVD enrichment relative to the analysis domain. The default criterion is

   ```text
   E_LAVD > 1
   ```

The numerical and physical parameters controlling these steps are specified in the application run scripts rather than in `src/FTRCS.jl`.

## Repository structure

```text
src/
    FTRCS.jl

scripts/
    run_FTRCS_gom.jl
    run_FTRCS_supergranule.jl

data/
    aviso_20130518_20140421.nc
    supergranule_synthetic.nc

runs/
    generated output
```

`src/FTRCS.jl` contains the reusable numerical implementation.

The scripts under `scripts/` define application-specific choices including:

- input and output files;
- analysis domain and time window;
- flow-map and Cauchy--Green resolution;
- trajectory integrator;
- integration timestep or adaptive-solver tolerances;
- IDL/FEM spatial and temporal resolution;
- inflation-parameter range;
- number of IDL eigenmodes;
- SEBA thresholding parameters; and
- LAVD classification threshold.

Users wishing to apply FTRCS to another velocity dataset should generally create a new run script rather than modify `src/FTRCS.jl`.

## Velocity-data convention

The example run scripts read velocity fields from NetCDF files.

The common external data convention is

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

The run scripts perform one spatial permutation before constructing the internal velocity field:

```julia
u = permutedims(u,(2,1,3))
v = permutedims(v,(2,1,3))
```

Inside `FTRCS.jl`, the invariant convention is therefore

```text
u(x,y,t), v(x,y,t)  ->  [Nx Ny Nt]

dimension 1 -> x
dimension 2 -> y
dimension 3 -> t
```

All velocity interpolation, trajectory integration, spatial differentiation, Cauchy--Green calculations, vorticity, and LAVD calculations use this internal convention.

## Installation

The code uses a Julia project environment defined by `Project.toml` and `Manifest.toml`.

Clone the repository and enter its root directory. Then instantiate the environment once with

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The examples can then be executed directly from the repository root.

The number of Julia threads can be selected according to the available hardware. A generic application is run as

```bash
julia --project=. --threads=18 scripts/run_FTRCS_velocity.jl
```

where `run_FTRCS_velocity.jl` denotes an application-specific run script.

## Example 1: Gulf of Mexico altimetry

The Gulf of Mexico example uses AVISO altimetric velocity data and demonstrates the FTRCS calculation on an ocean domain containing land boundaries.

The example dataset is

```text
data/aviso_20130518_20140421.nc
```

and the calculation is configured in

```text
scripts/run_FTRCS_gom.jl
```

Run it from the repository root with

```bash
julia --project=. --threads=18 scripts/run_FTRCS_gom.jl
```

The run script selects the Gulf of Mexico analysis domain and time interval, converts the geographic velocity data to the Cartesian coordinates and units used internally by FTRCS, and constructs the corresponding wet-domain mask.

All numerical resolutions and IDL/SEBA parameters used in the example are visible near the beginning of the run script.

## Example 2: synthetic supergranule-like flow

The second example uses a synthetic, strongly compressible cellular velocity field designed to have spatial and temporal characteristics qualitatively similar to photospheric supergranulation.

It is not a reconstruction of the observational velocity field used in the accompanying paper. Its purpose is to provide a freely shareable velocity dataset on which the complete FTRCS pipeline can be reproduced.

The run script expects

```text
data/supergranule_synthetic.nc
```

and is configured in

```text
scripts/run_FTRCS_supergranule.jl
```

Run it with

```bash
julia --project=. --threads=18 scripts/run_FTRCS_supergranule.jl
```

The synthetic field is strongly compressible and contains evolving cellular structures with both divergent/convergent and rotational components.

Because of its file size, `supergranule_synthetic.nc` may be distributed separately from the main Git repository. Place the file in `data/` before running the example.

## Trajectory integration

The flow-map calculation supports two trajectory-integration options.

### Adaptive Dormand--Prince

```julia
flow_integrator = :DP5
```

uses adaptive Dormand--Prince integration with user-specified relative and absolute tolerances.

### Fixed-step RK4

```julia
flow_integrator = :RK4
```

uses classical fourth-order Runge--Kutta integration with a prescribed timestep.

For large altimetric trajectory ensembles, fixed-step RK4 can provide a substantial performance advantage. The appropriate timestep should be chosen according to the temporal resolution and character of the velocity data.

The trajectory ensemble is divided into vectorized batches and distributed across Julia threads. Batch size and parallel settings are specified in the run script.

## IDL resolution

The flow-map/Cauchy--Green grid and the IDL/FEM grid are controlled independently.

For example,

```julia
cg_space_km  = ...
idl_space_km = ...
```

specify the spatial resolutions of the two calculations.

This separation allows the flow map and deformation field to be resolved more finely than the space-time spectral problem when appropriate.

Similarly, the IDL temporal spacing is specified independently of the native velocity-data cadence.

The actual spatial and temporal resolutions resulting from the requested values are printed when a run begins.

## Inflation parameter and eigenmodes

The IDL calculation is performed for a user-defined sequence of inflation parameters,

```julia
a_values = [...]
```

and a prescribed number of leading eigenmodes,

```julia
nmodes = ...
```

The code reports the spatial and material contributions to the IDL Rayleigh quotient and identifies a spatial/material crossover value of `a`.

The retained eigenspace is subsequently localized using SEBA.

The number of retained modes controls the dimension of the spectral subspace available to SEBA and should therefore be regarded as a spectral-resolution parameter rather than simply as a numerical tolerance.

## LAVD classification

IDL--SEBA identifies finite-time coherent-set candidates without requiring them to be rotational.

The rotational classification is based on the mean LAVD of a candidate relative to the mean LAVD over the analysis domain,

```text
           mean LAVD within candidate
E_LAVD =  ----------------------------
             mean LAVD over domain
```

The default classification uses

```text
E_LAVD > 1
```

to identify candidates with enhanced accumulated intrinsic rotation.

This second stage is important because an IDL coherent set may arise through contraction or other transport mechanisms without corresponding to a rotationally coherent material region.

## Output

Application scripts write NetCDF output to

```text
runs/
```

unless another output location is specified in the run script.

The output contains the quantities required for subsequent visualization and analysis, including IDL--SEBA candidate masks, candidate diagnostics, LAVD classification information, and the spatial and temporal grids used by the calculation.

The `runs/` directory is excluded from version control except for its placeholder file.

## Notes on interpretation

IDL--SEBA and LAVD answer related but distinct questions.

IDL--SEBA identifies regions that are coherent from the standpoint of finite-time transport. LAVD measures accumulated intrinsic material rotation. Consequently, not every IDL--SEBA coherent set is expected to satisfy the rotational criterion.

Likewise, IDL--SEBA boundaries should not in general be interpreted as equivalent to boundaries obtained from other material-vortex definitions, such as geodesic vortex boundaries. The methods optimize different notions of coherence.

The Gulf of Mexico and synthetic supergranule examples are included in part to illustrate the behavior of the framework in dynamically different settings.

## Reference

If you use this implementation, please refer to:

F. J. Beron-Vera,  
**"Quasi-material finite-time rotationally coherent sets in photospheric supergranulation,"**  
submitted to *Physics of Plasmas*, 2026.  
arXiv:2606.17108.

https://arxiv.org/abs/2606.17108

## Status

This repository accompanies ongoing research and is currently under development. Numerical parameters and interfaces may continue to evolve as the implementation is tested across additional velocity fields.