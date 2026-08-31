# FTRCS

Julia implementation for the computation of finite-time rotational coherent structures (FTRCS) from two-dimensional, time-dependent velocity fields.

The method combines an inflated dynamic Laplacian (IDL) calculation with sparse eigenbasis approximation (SEBA) to identify finite-time coherent sets, followed by a Lagrangian-averaged vorticity deviation (LAVD) criterion to identify rotationally coherent structures.

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

