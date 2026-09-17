module FTRCS

using Interpolations
using OrdinaryDiffEqLowOrderRK
using OrdinaryDiffEq
using DifferentialEquations
using ProgressMeter
using SparseArrays
using Statistics
using Arpack
using LinearAlgebra
using NCDatasets
using Printf

# ============================================================
# FINITE-TIME ROTATIONAL COHERENT STRUCTURES (FTRCS)
# ============================================================
#
# Core numerical routines for detecting finite-time rotational
# coherent structures from two-dimensional, time-dependent
# velocity fields.
#
# This file contains the reusable numerical implementation only.
# Application-specific choices are made in the run scripts in
#
#     scripts/
#
# In particular, the run scripts specify
#
#   - input and output files,
#   - analysis domain and time window,
#   - flow-map / Cauchy-Green resolution,
#   - trajectory integrator and tolerances or timestep,
#   - IDL/FEM spatial and temporal resolution,
#   - inflation-parameter sweep,
#   - number of eigenmodes,
#   - SEBA thresholding parameters, and
#   - LAVD classification parameters.
#
#
# ------------------------------------------------------------
# INPUT-DATA CONVENTION
# ------------------------------------------------------------
#
# The example run scripts read velocity fields from NetCDF.
#
# NetCDF files are expected to contain
#
#     x       spatial coordinate                 [km]
#     y       spatial coordinate                 [km]
#     t       time                               [h]
#     u       x velocity component               [km/h]
#     v       y velocity component               [km/h]
#
# with velocity arrays stored externally as
#
#     u(y,x,t), v(y,x,t)  ->  [Ny Nx Nt].
#
# The run script converts these arrays once using
#
#     permutedims(u,(2,1,3))
#     permutedims(v,(2,1,3))
#
# before constructing VelocityField.
#
# Inside FTRCS.jl the invariant convention is therefore
#
#     u(x,y,t), v(x,y,t)  ->  [Nx Ny Nt],
#
# i.e.
#
#     dimension 1 -> x
#     dimension 2 -> y
#     dimension 3 -> t.
#
# All spatial derivatives, trajectory calculations,
# Cauchy-Green tensors, vorticity, and LAVD calculations use
# this internal convention.
#
#
# ------------------------------------------------------------
# RUNNING AN APPLICATION
# ------------------------------------------------------------
#
# From the repository root, instantiate the Julia environment
# once with
#
#     julia --project=. -e 'using Pkg; Pkg.instantiate()'
#
# Then run an application script with, for example,
#
#     julia --project=. --threads=18 \
#         scripts/run_FTRCS_velocity.jl
#
# where the run script specifies the velocity dataset,
# analysis window, numerical resolution, solver settings,
# IDL/SEBA parameters, and output files.
#
# The number of Julia threads may be changed according to the
# available hardware.
#
# Example run scripts distributed with the repository use
#
#   - Gulf of Mexico AVISO velocity data, and
#   - a synthetic supergranule-like velocity field.
#
# Output files are written to
#
#     runs/
#
# unless changed in the corresponding run script.
#
#
# ------------------------------------------------------------
# IMPORTANT
# ------------------------------------------------------------
#
# Do not place application-specific numerical parameters in this
# module. Parameters intended to be changed by the user belong in
# the corresponding script under scripts/.
#
# ============================================================

# ============================================================
# DATA TYPES
# ============================================================

# Internal velocity representation.  Arrays MUST be [Nx Ny Nt].
struct VelocityField
    x::Vector{Float64}
    y::Vector{Float64}
    t::Vector{Float64}

    u::Array{Float64,3}
    v::Array{Float64,3}

    dx::Float64
    dy::Float64
    dt::Float64
end


struct FlowMap
    PhiX::Array{Float64,3}
    PhiY::Array{Float64,3}

    xseed::Vector{Float64}
    yseed::Vector{Float64}

    times::Vector{Float64}

    seed_dx::Float64
    seed_dy::Float64
end

# ============================================================
# VELOCITY INTERPOLANTS
#
# MATLAB reference:
#
# Fu = griddedInterpolant({x,y,t},u,'linear','none');
# Fv = griddedInterpolant({x,y,t},v,'linear','none');
#
# The interpolants below are linear in x, y, and t.
# Out-of-domain handling is performed explicitly in velocity().
# ============================================================

struct VelocityInterpolants{TU,TV}
    uitp::TU
    vitp::TV

    xmin::Float64
    xmax::Float64

    ymin::Float64
    ymax::Float64

    tmin::Float64
    tmax::Float64
end


function build_interpolants(V::VelocityField)

    uitp = interpolate(
        (V.x, V.y, V.t),
        V.u,
        Gridded(Linear())
    )

    vitp = interpolate(
        (V.x, V.y, V.t),
        V.v,
        Gridded(Linear())
    )

    VelocityInterpolants(
        uitp,
        vitp,
        first(V.x),
        last(V.x),
        first(V.y),
        last(V.y),
        first(V.t),
        last(V.t)
    )
end


# ============================================================
# VELOCITY EVALUATION
#
# This reproduces the MATLAB run_02 behavior:
# if interpolation would be outside the observed domain,
# set the velocity to zero.
# ============================================================

@inline function velocity(
    x::Real,
    y::Real,
    t::Real,
    I::VelocityInterpolants
)

    if x < I.xmin || x > I.xmax ||
       y < I.ymin || y > I.ymax ||
       t < I.tmin || t > I.tmax

        return 0.0, 0.0
    end

    return I.uitp(x,y,t), I.vitp(x,y,t)
end


# ============================================================
# VECTORIZED FLOW-MAP RHS
#
# State ordering:
#
# X = [x1,y1,x2,y2,...]
#
# Trajectories are integrated in vectorized batches.  Within each
# batch the particle states are packed into one ODE system.
# ============================================================

function flowmap_rhs!(
    dX,
    X,
    I::VelocityInterpolants,
    t
)

    Np = length(X) ÷ 2

    @inbounds for p in 1:Np
        k = 2p - 1

        u, v = velocity(
            X[k],
            X[k+1],
            t,
            I
        )

        dX[k]   = u
        dX[k+1] = v
    end

    nothing
end


# ============================================================
# FLOW MAP
#
# Independent trajectories are integrated in moderate-size
# vectorized batches. This avoids constructing a few enormous
# ODE systems whose state vectors contain hundreds of thousands
# of particles.
#
# Output is stored only at the requested native velocity times.
# ============================================================

function compute_flowmap(
    V::VelocityField;
    seed_dx::Real,
    seed_dy::Real,
    window::Tuple{<:Real,<:Real},
    integrator::Symbol,
    rk4_dt::Real,
    reltol::Real,
    abstol::Real,
    batch_size::Int,
    nchunks::Int
)

    t0 = Float64(window[1])
    t1 = Float64(window[2])

    t0 < t1 ||
        error("Flow-map window must satisfy t0 < t1")

    t0 >= first(V.t) ||
        error("Flow-map initial time lies before velocity record")

    t1 <= last(V.t) ||
        error("Flow-map final time lies after velocity record")

    seed_dx > 0 ||
        error("seed_dx must be positive")

    seed_dy > 0 ||
        error("seed_dy must be positive")

    nchunks >= 1 ||
        error("nchunks must be >= 1")

    integrator in (:DP5,:RK4) ||
        error("integrator must be :DP5 or :RK4")

    rk4_dt > 0 ||
        error("rk4_dt must be positive")

    ids =
        findall(
            t -> t0 <= t <= t1,
            V.t
        )

    isempty(ids) &&
        error(
            "No native velocity times lie in requested window"
        )

    times =
        V.t[ids]

    # --------------------------------------------------------
    # Trajectory grid
    # --------------------------------------------------------

    xseed =
        collect(
            range(
                first(V.x),
                stop = last(V.x),
                step = Float64(seed_dx)
            )
        )

    yseed =
        collect(
            range(
                first(V.y),
                stop = last(V.y),
                step = Float64(seed_dy)
            )
        )

    Nx0 = length(xseed)
    Ny0 = length(yseed)

    Np = Nx0 * Ny0
    Nt = length(times)

    # --------------------------------------------------------
    # Batch size
    #
    # Supplied by the run script.  Moderate batches avoid very
    # large coupled ODE states while retaining vectorized velocity
    # evaluation and threaded parallelism.
    # --------------------------------------------------------

    batch_size >= 1 ||
        error("batch_size must be >= 1")

    nbatches =
        cld(
            Np,
            batch_size
        )

    println()
    println("Computing flow map")
    println("window       = $(times[1]) -- $(times[end]) h")
    println("seed spacing = $(seed_dx) x $(seed_dy) km")
    println("seed grid    = $Nx0 x $Ny0")
    println("trajectories = $Np")
    println("output times = $Nt")
    println("batch size   = $batch_size")
    println("batches      = $nbatches")
    println("Julia threads= $(Threads.nthreads())")
    println("integrator   = $(integrator)")

    if integrator == :DP5
        println("reltol       = $reltol")
        println("abstol       = $abstol")
    else
        println("RK4 dt       = $(Float64(rk4_dt)) h")
    end

    # --------------------------------------------------------
    # Initial positions
    #
    # Linear particle ordering:
    #
    #     p = i + (j-1)Nx
    #
    # x varies fastest.
    # --------------------------------------------------------

    x0 =
        Vector{Float64}(undef,Np)

    y0 =
        Vector{Float64}(undef,Np)

    @inbounds for j in eachindex(yseed)

        for i in eachindex(xseed)

            p =
                i +
                (j-1)*Nx0

            x0[p] =
                xseed[i]

            y0[p] =
                yseed[j]

        end

    end

    # --------------------------------------------------------
    # Velocity interpolants
    # --------------------------------------------------------

    I =
        build_interpolants(V)

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    PhiX =
        Array{Float64}(
            undef,
            Nx0,
            Ny0,
            Nt
        )

    PhiY =
        Array{Float64}(
            undef,
            Nx0,
            Ny0,
            Nt
        )

    progress =
        Progress(
            nbatches;
            desc = "Flow-map batches: ",
            dt = 0.5
        )

    elapsed =
        @elapsed begin

            Threads.@threads for ib in 1:nbatches

                p1 =
                    (ib-1)*batch_size + 1

                p2 =
                    min(
                        ib*batch_size,
                        Np
                    )

                Nb =
                    p2-p1+1

                # --------------------------------------------
                # Vectorized batch state
                #
                # X = [x1,y1,x2,y2,...]
                # --------------------------------------------

                X0 =
                    Vector{Float64}(
                        undef,
                        2Nb
                    )

                @inbounds for q in 1:Nb

                    p =
                        p1+q-1

                    X0[2q-1] =
                        x0[p]

                    X0[2q] =
                        y0[p]

                end

                prob =
                    ODEProblem(
                        flowmap_rhs!,
                        X0,
                        (
                            times[1],
                            times[end]
                        ),
                        I
                    )

                sol =
                    if integrator == :DP5

                        solve(
                            prob,
                            DP5();
                            saveat = times,
                            save_start = true,
                            save_everystep = false,
                            dense = false,
                            reltol = Float64(reltol),
                            abstol = Float64(abstol)
                        )

                    else

                        solve(
                            prob,
                            RK4();
                            adaptive = false,
                            dt = Float64(rk4_dt),
                            saveat = times,
                            save_start = true,
                            save_everystep = false,
                            dense = false
                        )

                    end

                # --------------------------------------------
                # Copy batch to flow-map arrays
                # --------------------------------------------

                @inbounds for k in 1:Nt

                    Xk =
                        sol.u[k]

                    for q in 1:Nb

                        p =
                            p1+q-1

                        i =
                            mod1(
                                p,
                                Nx0
                            )

                        j =
                            fld(
                                p-1,
                                Nx0
                            ) + 1

                        PhiX[i,j,k] =
                            Xk[2q-1]

                        PhiY[i,j,k] =
                            Xk[2q]

                    end

                end

                next!(progress)

            end

        end

    println(
        "elapsed time = ",
        round(
            elapsed,
            digits = 2
        ),
        " s"
    )

    return FlowMap(
        PhiX,
        PhiY,
        xseed,
        yseed,
        collect(times),
        Float64(seed_dx),
        Float64(seed_dy)
    )

end

# ============================================================
# CAUCHY-GREEN TENSOR
# ============================================================

struct CauchyGreenField

    C11::Array{Float64,3}
    C12::Array{Float64,3}
    C22::Array{Float64,3}

    detC::Array{Float64,3}
    detDF::Array{Float64,3}

    lam_min::Array{Float64,3}
    lam_max::Array{Float64,3}

    x::Vector{Float64}
    y::Vector{Float64}
    times::Vector{Float64}

end


# ============================================================
# FIRST DERIVATIVE ON A UNIFORM GRID
#
# Arrays use the (x,y) convention:
#
#     dimension 1 -> x
#     dimension 2 -> y
#
# Centered differences are used in the interior and one-sided
# differences at the boundaries.
# ============================================================

function derivative_x(
    A::AbstractMatrix,
    dx::Real
)

    Nx, Ny = size(A)

    D =
        Matrix{Float64}(
            undef,
            Nx,
            Ny
        )

    h =
        Float64(dx)

    inv2h =
        1.0 / (2.0*h)

    invh =
        1.0 / h

    @inbounds for j in 1:Ny

        D[1,j] =
            (A[2,j] - A[1,j]) *
            invh

        for i in 2:Nx-1

            D[i,j] =
                (A[i+1,j] - A[i-1,j]) *
                inv2h

        end

        D[Nx,j] =
            (A[Nx,j] - A[Nx-1,j]) *
            invh

    end

    return D

end


function derivative_y(
    A::AbstractMatrix,
    dy::Real
)

    Nx, Ny = size(A)

    D =
        Matrix{Float64}(
            undef,
            Nx,
            Ny
        )

    h =
        Float64(dy)

    inv2h =
        1.0 / (2.0*h)

    invh =
        1.0 / h

    @inbounds for i in 1:Nx

        D[i,1] =
            (A[i,2] - A[i,1]) *
            invh

        for j in 2:Ny-1

            D[i,j] =
                (A[i,j+1] - A[i,j-1]) *
                inv2h

        end

        D[i,Ny] =
            (A[i,Ny] - A[i,Ny-1]) *
            invh

    end

    return D

end


# ============================================================
# COMPUTE CAUCHY-GREEN FIELD
#
# Flow-map arrays use the (x,y,t) convention:
#
#     dimension 1 -> x
#     dimension 2 -> y
#
# For each time:
#
#         DF = [ X_x  X_y
#                Y_x  Y_y ]
#
# and
#
#         C = DF' * DF
#
# so that
#
#         C11 = X_x^2 + Y_x^2
#
#         C12 = X_x X_y + Y_x Y_y
#
#         C22 = X_y^2 + Y_y^2.
# ============================================================

function compute_cauchy_green(
    F::FlowMap
)

    Nx =
        length(F.xseed)

    Ny =
        length(F.yseed)

    Nt =
        length(F.times)

    dx =
        Float64(F.seed_dx)

    dy =
        Float64(F.seed_dy)

    C11 =
        fill(
            NaN,
            Nx,
            Ny,
            Nt
        )

    C12 =
        fill(
            NaN,
            Nx,
            Ny,
            Nt
        )

    C22 =
        fill(
            NaN,
            Nx,
            Ny,
            Nt
        )

    detC =
        fill(
            NaN,
            Nx,
            Ny,
            Nt
        )

    detDF =
        fill(
            NaN,
            Nx,
            Ny,
            Nt
        )

    lam_min =
        fill(
            NaN,
            Nx,
            Ny,
            Nt
        )

    lam_max =
        fill(
            NaN,
            Nx,
            Ny,
            Nt
        )

    println()
    println("Computing Cauchy-Green tensor")
    println("grid = $Nx x $Ny x $Nt")
    println("dx   = $dx km")
    println("dy   = $dy km")

    progress =
        Progress(
            Nt;
            desc = "Cauchy-Green slices: ",
            dt = 0.5
        )

    elapsed =
        @elapsed begin

            Threads.@threads for k in 1:Nt

                X =
                    @view F.PhiX[:,:,k]

                Y =
                    @view F.PhiY[:,:,k]

                #
                # Correct spatial derivatives for arrays stored
                # as (x,y,t).
                #

                X_x =
                    derivative_x(
                        X,
                        dx
                    )

                X_y =
                    derivative_y(
                        X,
                        dy
                    )

                Y_x =
                    derivative_x(
                        Y,
                        dx
                    )

                Y_y =
                    derivative_y(
                        Y,
                        dy
                    )

                @inbounds for j in 1:Ny

                    for i in 1:Nx

                        if !isfinite(X[i,j]) ||
                           !isfinite(Y[i,j]) ||
                           !isfinite(X_x[i,j]) ||
                           !isfinite(X_y[i,j]) ||
                           !isfinite(Y_x[i,j]) ||
                           !isfinite(Y_y[i,j])

                            continue

                        end

                        xx =
                            X_x[i,j]

                        xy =
                            X_y[i,j]

                        yx =
                            Y_x[i,j]

                        yy =
                            Y_y[i,j]

                        c11 =
                            xx*xx +
                            yx*yx

                        c12 =
                            xx*xy +
                            yx*yy

                        c22 =
                            xy*xy +
                            yy*yy

                        dc =
                            c11*c22 -
                            c12*c12

                        trC =
                            c11 +
                            c22

                        disc =
                            sqrt(
                                max(
                                    (c11-c22)^2 +
                                    4.0*c12*c12,
                                    0.0
                                )
                            )

                        lmin =
                            0.5 * (
                                trC -
                                disc
                            )

                        lmax =
                            0.5 * (
                                trC +
                                disc
                            )

                        C11[i,j,k] =
                            c11

                        C12[i,j,k] =
                            c12

                        C22[i,j,k] =
                            c22

                        lam_min[i,j,k] =
                            lmin

                        lam_max[i,j,k] =
                            lmax

                        if dc > 0.0

                            detC[i,j,k] =
                                dc

                            detDF[i,j,k] =
                                sqrt(dc)

                        end

                    end

                end

                next!(progress)

            end

        end

    println(
        "elapsed time = ",
        round(
            elapsed,
            digits = 2
        ),
        " s"
    )

    return CauchyGreenField(
        C11,
        C12,
        C22,
        detC,
        detDF,
        lam_min,
        lam_max,
        F.xseed,
        F.yseed,
        F.times
    )

end

# ============================================================
# INFLATED DYNAMIC LAPLACIAN
# ============================================================

abstract type AbstractIDLOperator end

struct IDLOperator <: AbstractIDLOperator

    Kspace::SparseMatrixCSC{Float64,Int}
    Kmaterial::SparseMatrixCSC{Float64,Int}
    G::SparseMatrixCSC{Float64,Int}

    Nx::Int
    Ny::Int
    Nt::Int

    Nxy::Int
    Nst::Int

    x::Vector{Float64}
    y::Vector{Float64}
    t::Vector{Float64}

    ix::Vector{Int}
    iy::Vector{Int}
    it::Vector{Int}

    dx::Float64
    dy::Float64
    dt::Float64

end

struct MaskedIDLOperator <: AbstractIDLOperator

    Kspace::SparseMatrixCSC{Float64,Int}
    Kmaterial::SparseMatrixCSC{Float64,Int}
    G::SparseMatrixCSC{Float64,Int}

    Nx::Int
    Ny::Int
    Nt::Int

    Nxy::Int
    Nst::Int

    x::Vector{Float64}
    y::Vector{Float64}
    t::Vector{Float64}

    ix::Vector{Int}
    iy::Vector{Int}
    it::Vector{Int}

    dx::Float64
    dy::Float64
    dt::Float64

    wet_mask::BitMatrix
    active_full::Vector{Int}
    dof_map::Matrix{Int}

end

is_masked(::IDLOperator) = false
is_masked(::MaskedIDLOperator) = true


# ============================================================
# 1-D CENTERED DERIVATIVE MATRIX
#
# Centered differences in the interior and one-sided
# differences at the endpoints, matching MATLAB run_05.
# ============================================================

function derivative_matrix_1d(N::Int, h::Real)

    h = Float64(h)

    I = Int[]
    J = Int[]
    V = Float64[]

    #
    # Left endpoint
    #

    push!(I,1); push!(J,1); push!(V,-1/h)
    push!(I,1); push!(J,2); push!(V, 1/h)

    #
    # Interior
    #

    for i in 2:N-1

        push!(I,i); push!(J,i-1); push!(V,-1/(2h))
        push!(I,i); push!(J,i+1); push!(V, 1/(2h))

    end

    #
    # Right endpoint
    #

    push!(I,N); push!(J,N-1); push!(V,-1/h)
    push!(I,N); push!(J,N);   push!(V, 1/h)

    return sparse(I,J,V,N,N)

end


# ============================================================
# Q1 TENSOR STIFFNESS FOR ONE TIME SLICE
#
# Input coefficient:
#
#       A = [ A11 A12
#             A12 A22 ]
#
# ============================================================

function assemble_q1_tensor_stiffness(
    A11::AbstractMatrix,
    A12::AbstractMatrix,
    A22::AbstractMatrix,
    dx::Real,
    dy::Real
)

    Nx, Ny = size(A11)

    N = Nx * Ny

    dx = Float64(dx)
    dy = Float64(dy)

    nel = (Nx-1)*(Ny-1)

    II = Vector{Int}(undef,16*nel)
    JJ = Vector{Int}(undef,16*nel)
    VV = Vector{Float64}(undef,16*nel)

    gp = (-1/sqrt(3),1/sqrt(3))

    Bgauss = Matrix{Float64}[]

    for ξ in gp
        for η in gp

            dN_dξ = 0.25 .* [
                -(1-η),
                 (1-η),
                 (1+η),
                -(1+η)
            ]

            dN_dη = 0.25 .* [
                -(1-ξ),
                -(1+ξ),
                 (1+ξ),
                 (1-ξ)
            ]

            B = [
                (2/dx .* dN_dξ)';
                (2/dy .* dN_dη)'
            ]

            push!(Bgauss,B)

        end
    end

    jac = dx*dy/4

    ptr = 0

    for j in 1:Ny-1
        for i in 1:Nx-1

            n1 = i     + (j-1)*Nx
            n2 = i + 1 + (j-1)*Nx
            n3 = i + 1 +  j   *Nx
            n4 = i     +  j   *Nx

            nodes = (n1,n2,n3,n4)

            a11 = 0.25*(
                A11[i,j] +
                A11[i+1,j] +
                A11[i+1,j+1] +
                A11[i,j+1]
            )

            a12 = 0.25*(
                A12[i,j] +
                A12[i+1,j] +
                A12[i+1,j+1] +
                A12[i,j+1]
            )

            a22 = 0.25*(
                A22[i,j] +
                A22[i+1,j] +
                A22[i+1,j+1] +
                A22[i,j+1]
            )

            A = [
                a11 a12
                a12 a22
            ]

            Ke = zeros(Float64,4,4)

            for B in Bgauss
                Ke .+= B' * A * B .* jac
            end

            @inbounds for a in 1:4
                for b in 1:4

                    ptr += 1

                    II[ptr] = nodes[a]
                    JJ[ptr] = nodes[b]
                    VV[ptr] = Ke[a,b]

                end
            end

        end
    end

    K = sparse(II,JJ,VV,N,N)

    return 0.5*(K + K')

end


# ============================================================
# LUMPED Q1 MASS MATRIX
# ============================================================

function build_lumped_mass(
    Nx::Int,
    Ny::Int,
    dx::Real,
    dy::Real
)

    mass = fill(Float64(dx*dy),Nx,Ny)

    mass[1,:]   ./= 2
    mass[end,:] ./= 2

    mass[:,1]   ./= 2
    mass[:,end] ./= 2

    return spdiagm(0 => vec(mass))

end


# ============================================================
# ASSEMBLE IDL COMPONENTS
#
# MATLAB-equivalent construction of
#
#       Kspace
#       Kmaterial
#       G
#
# No inflation parameter a appears here.
# ============================================================

function assemble_idl_operator_rectangular(
    CG::CauchyGreenField,
    V::VelocityField;
    space_stride::Int,
    time_stride::Int
)

    space_stride >= 1 ||
        error("space_stride must be >= 1")

    time_stride >= 1 ||
        error("time_stride must be >= 1")

    ix = collect(1:space_stride:length(CG.x))
    iy = collect(1:space_stride:length(CG.y))
    it = collect(1:time_stride:length(CG.times))

    x = CG.x[ix]
    y = CG.y[iy]
    t = CG.times[it]

    Nx = length(x)
    Ny = length(y)
    Nt = length(t)

    Nxy = Nx*Ny
    Nst = Nxy*Nt

    dx = x[2]-x[1]
    dy = y[2]-y[1]
    dt = mean(diff(t))

    println()
    println("Assembling inflated dynamic Laplacian components")
    println("space grid = $Nx x $Ny")
    println("time levels = $Nt")
    println("space-time DOF = $Nst")
    println("dx = $dx km")
    println("dy = $dy km")
    println("dt = $dt h")

    # --------------------------------------------------------
    # SPATIAL CONTRIBUTION
    # --------------------------------------------------------

    Kblocks = Vector{SparseMatrixCSC{Float64,Int}}(undef,Nt)

    progress = Progress(
        Nt;
        desc = "IDL spatial slices: ",
        dt = 0.5
    )

    Threads.@threads for kk in 1:Nt

        k = it[kk]

        C11 = @view CG.C11[ix,iy,k]
        C12 = @view CG.C12[ix,iy,k]
        C22 = @view CG.C22[ix,iy,k]
        detC = @view CG.detC[ix,iy,k]

        A11 = zeros(Float64,Nx,Ny)
        A12 = zeros(Float64,Nx,Ny)
        A22 = zeros(Float64,Nx,Ny)

        @inbounds for j in 1:Ny
            for i in 1:Nx

                d = detC[i,j]

                if isfinite(d) && d > 0

                    J = sqrt(d)

                    A11[i,j] =  C22[i,j]/d * J
                    A12[i,j] = -C12[i,j]/d * J
                    A22[i,j] =  C11[i,j]/d * J

                end

            end
        end

        Kblocks[kk] =
            assemble_q1_tensor_stiffness(
                A11,A12,A22,dx,dy
            )

        next!(progress)

    end

    #
    # Block diagonal Kspace.
    #

    Kspace = blockdiag(Kblocks...)

    # --------------------------------------------------------
    # MATERIAL DERIVATIVE
    #
    # D/Dt = d/dt + u d/dx + v d/dy
    # --------------------------------------------------------

    println("Building material derivative")

    Dx1 = derivative_matrix_1d(Nx,dx)
    Dy1 = derivative_matrix_1d(Ny,dy)
    Dt1 = derivative_matrix_1d(Nt,dt)

    Ix = spdiagm(0 => ones(Float64,Nx))
    Iy = spdiagm(0 => ones(Float64,Ny))
	 Ixy = spdiagm(0 => ones(Float64,Nxy))

    Dx = kron(Iy,Dx1)
    Dy = kron(Dy1,Ix)

    Dt = kron(Dt1,Ixy)

    #
    # Velocity is evaluated at the physical IDL grid.
    #
    # This is more general than the MATLAB indexing scheme,
    # because our trajectory/CG grid need not coincide with
    # the original velocity grid.
    #

    VI = build_interpolants(V)

    Adv_blocks =
        Vector{SparseMatrixCSC{Float64,Int}}(undef,Nt)

    progress = Progress(
        Nt;
        desc = "IDL advection blocks: ",
        dt = 0.5
    )

    Threads.@threads for kk in 1:Nt

        uvec = Vector{Float64}(undef,Nxy)
        vvec = Vector{Float64}(undef,Nxy)

        p = 1

        @inbounds for j in 1:Ny
            for i in 1:Nx

                uu,vv = velocity(
                    x[i],y[j],t[kk],VI
                )

                uvec[p] = uu
                vvec[p] = vv

                p += 1

            end
        end

        Adv_blocks[kk] =
            spdiagm(0 => uvec)*Dx +
            spdiagm(0 => vvec)*Dy

        next!(progress)

    end

    Adv = blockdiag(Adv_blocks...)

    Dmat = Dt + Adv

    # --------------------------------------------------------
    # SPACE-TIME MASS / GRAM MATRIX
    # --------------------------------------------------------

    Mxy = build_lumped_mass(
        Nx,Ny,dx,dy
    )

    wt = fill(dt,Nt)

    wt[1]   = dt/2
    wt[end] = dt/2

    Mt = spdiagm(0 => wt)

    G = kron(Mt,Mxy)

    # --------------------------------------------------------
    # MATERIAL CONTRIBUTION
    # --------------------------------------------------------

    println("Forming Kmaterial")

    Kmaterial = Dmat' * G * Dmat
    Kmaterial = 0.5*(Kmaterial + Kmaterial')

    println()
    println("IDL component assembly complete")
    println("matrix size      = $Nst x $Nst")
    println("nnz(Kspace)      = ",nnz(Kspace))
    println("nnz(Kmaterial)   = ",nnz(Kmaterial))
    println("nnz(G)           = ",nnz(G))

	 return IDLOperator(
        Kspace,
        Kmaterial,
        G,
        Nx,
        Ny,
        Nt,
        Nxy,
        Nst,
        collect(x),
        collect(y),
        collect(t),
        ix,
        iy,
        it,
        dx,
        dy,
        dt
    )

end

# ============================================================
# STEP 5A MATERIAL SURVIVAL DOMAIN
#
# Determine the initial conditions whose trajectories remain
# inside the observed velocity domain for the complete analysis
# interval.
#
# The returned mask is defined on the FLOW-MAP seed grid.
# A point is retained only if its trajectory is finite and lies
# inside the velocity domain at every output time.
# ============================================================

function material_survival_mask(
    F::FlowMap,
    V::VelocityField;
    verbose::Bool = true
)

    Nx = length(F.xseed)
    Ny = length(F.yseed)

    size(F.PhiX,1) == Nx ||
        error("PhiX x dimension inconsistent with xseed")

    size(F.PhiX,2) == Ny ||
        error("PhiX y dimension inconsistent with yseed")

    size(F.PhiX) == size(F.PhiY) ||
        error("PhiX and PhiY sizes differ")

    xmin = minimum(V.x)
    xmax = maximum(V.x)
    ymin = minimum(V.y)
    ymax = maximum(V.y)

    survive = trues(Nx,Ny)

    exit_time = fill(NaN,Nx,Ny)

    for k in eachindex(F.times)

        X = @view F.PhiX[:,:,k]
        Y = @view F.PhiY[:,:,k]

        inside =
            isfinite.(X) .&
            isfinite.(Y) .&
            (X .>= xmin) .&
            (X .<= xmax) .&
            (Y .>= ymin) .&
            (Y .<= ymax)

        newly_exited =
            survive .& .!inside

        exit_time[newly_exited] .= F.times[k]

        survive .&= inside
    end

    if verbose

        Ns = count(survive)
        Ntot = length(survive)

        println()
        println("STEP 5A material survival domain")
        println("--------------------------------")
        println(
            "analysis interval = ",
            F.times[1]," -- ",F.times[end]," h"
        )
        println(
            "surviving CG nodes = ",
            Ns," / ",Ntot,
            " (",
            round(100*Ns/Ntot,digits=2),
            "%)"
        )
    end

    return BitMatrix(survive),exit_time
end

# ============================================================
# MASKED IDL / FEM HELPERS
# ============================================================

function nearest_grid_index(grid::AbstractVector, value::Real)

    i = searchsortedfirst(grid,value)

    if i <= 1
        return 1
    elseif i > length(grid)
        return length(grid)
    else
        return abs(grid[i]-value) < abs(grid[i-1]-value) ? i : i-1
    end

end


function masked_idl_wet_grid(
    CG::CauchyGreenField,
    V::VelocityField,
    wet_mask::AbstractMatrix{Bool},
    ix::AbstractVector{Int},
    iy::AbstractVector{Int}
)

    # Preferred case: mask is already defined on the
    # Cauchy--Green / flow-map seed grid.

    if size(wet_mask) == (length(CG.x),length(CG.y))
        return BitMatrix(wet_mask[ix,iy])
    end

    # Backward-compatible case: mask is defined on the
    # original velocity grid.

    if size(wet_mask) == (length(V.x),length(V.y))

        Nx = length(ix)
        Ny = length(iy)

        wet = falses(Nx,Ny)

        for j in 1:Ny

            yj = CG.y[iy[j]]
            jv = nearest_grid_index(V.y,yj)

            for i in 1:Nx

                xi = CG.x[ix[i]]
                iv = nearest_grid_index(V.x,xi)

                wet[i,j] = wet_mask[iv,jv]
            end
        end

        return wet
    end

    error(
        "wet_mask must be defined either on the " *
        "Cauchy--Green grid or on the velocity grid"
    )
end


function build_wet_cells_and_dofs(wet::AbstractMatrix{Bool})

    Nx,Ny = size(wet)

    wet_cells = falses(Nx-1,Ny-1)
    active = falses(Nx,Ny)

    for j in 1:Ny-1
        for i in 1:Nx-1

            if wet[i,j] && wet[i+1,j] && wet[i+1,j+1] && wet[i,j+1]
                wet_cells[i,j] = true
                active[i,j] = true
                active[i+1,j] = true
                active[i+1,j+1] = true
                active[i,j+1] = true
            end

        end
    end

    dof = zeros(Int,Nx,Ny)
    active_full = Int[]

    n = 0
    for j in 1:Ny
        for i in 1:Nx
            if active[i,j]
                n += 1
                dof[i,j] = n
                push!(active_full,i + (j-1)*Nx)
            end
        end
    end

    n > 0 || error("wet_mask contains no complete wet FEM cells")

    return wet_cells,active,dof,active_full
end


function assemble_q1_tensor_stiffness_masked(
    A11::AbstractMatrix,
    A12::AbstractMatrix,
    A22::AbstractMatrix,
    wet_cells::AbstractMatrix{Bool},
    dof::AbstractMatrix{Int},
    dx::Real,
    dy::Real
)

    Nx,Ny = size(A11)
    N = maximum(dof)

    gp = (-1/sqrt(3),1/sqrt(3))
    Bgauss = Matrix{Float64}[]

    for ξ in gp
        for η in gp

            dN_dξ = 0.25 .* [
                -(1-η),
                 (1-η),
                 (1+η),
                -(1+η)
            ]

            dN_dη = 0.25 .* [
                -(1-ξ),
                -(1+ξ),
                 (1+ξ),
                 (1-ξ)
            ]

            B = [
                (2/Float64(dx) .* dN_dξ)';
                (2/Float64(dy) .* dN_dη)'
            ]

            push!(Bgauss,B)
        end
    end

    jac = Float64(dx*dy)/4

    II = Int[]
    JJ = Int[]
    VV = Float64[]

    for j in 1:Ny-1
        for i in 1:Nx-1

            wet_cells[i,j] || continue

            vals11 = (A11[i,j],A11[i+1,j],A11[i+1,j+1],A11[i,j+1])
            vals12 = (A12[i,j],A12[i+1,j],A12[i+1,j+1],A12[i,j+1])
            vals22 = (A22[i,j],A22[i+1,j],A22[i+1,j+1],A22[i,j+1])

            all(isfinite,vals11) && all(isfinite,vals12) && all(isfinite,vals22) || continue

            nodes = (
                dof[i,j],
                dof[i+1,j],
                dof[i+1,j+1],
                dof[i,j+1]
            )

            minimum(nodes) > 0 || continue

            A = [
                sum(vals11)/4  sum(vals12)/4
                sum(vals12)/4  sum(vals22)/4
            ]

            Ke = zeros(Float64,4,4)

            for B in Bgauss
                Ke .+= B' * A * B .* jac
            end

            for a in 1:4
                for b in 1:4
                    push!(II,nodes[a])
                    push!(JJ,nodes[b])
                    push!(VV,Ke[a,b])
                end
            end

        end
    end

    K = sparse(II,JJ,VV,N,N)
    return 0.5*(K + K')
end


function build_lumped_mass_masked(
    wet_cells::AbstractMatrix{Bool},
    dof::AbstractMatrix{Int},
    dx::Real,
    dy::Real
)

    N = maximum(dof)
    mass = zeros(Float64,N)
    mloc = Float64(dx*dy)/4

    Nx1,Ny1 = size(wet_cells)

    for j in 1:Ny1
        for i in 1:Nx1
            wet_cells[i,j] || continue

            nodes = (
                dof[i,j],
                dof[i+1,j],
                dof[i+1,j+1],
                dof[i,j+1]
            )

            for n in nodes
                n > 0 && (mass[n] += mloc)
            end
        end
    end

    all(mass .> 0) || error("masked FEM contains zero-mass active nodes")

    return spdiagm(0 => mass)
end


function derivative_matrix_masked_x(
    active::AbstractMatrix{Bool},
    dof::AbstractMatrix{Int},
    h::Real
)

    Nx,Ny = size(active)
    N = maximum(dof)
    h = Float64(h)

    I = Int[]; J = Int[]; V = Float64[]

    for j in 1:Ny
        for i in 1:Nx
            active[i,j] || continue

            p = dof[i,j]
            left  = i > 1  && active[i-1,j]
            right = i < Nx && active[i+1,j]

            if left && right
                push!(I,p); push!(J,dof[i-1,j]); push!(V,-1/(2h))
                push!(I,p); push!(J,dof[i+1,j]); push!(V, 1/(2h))
            elseif right
                push!(I,p); push!(J,p);          push!(V,-1/h)
                push!(I,p); push!(J,dof[i+1,j]); push!(V, 1/h)
            elseif left
                push!(I,p); push!(J,dof[i-1,j]); push!(V,-1/h)
                push!(I,p); push!(J,p);          push!(V, 1/h)
            end
        end
    end

    return sparse(I,J,V,N,N)
end


function derivative_matrix_masked_y(
    active::AbstractMatrix{Bool},
    dof::AbstractMatrix{Int},
    h::Real
)

    Nx,Ny = size(active)
    N = maximum(dof)
    h = Float64(h)

    I = Int[]; J = Int[]; V = Float64[]

    for j in 1:Ny
        for i in 1:Nx
            active[i,j] || continue

            p = dof[i,j]
            down = j > 1  && active[i,j-1]
            up   = j < Ny && active[i,j+1]

            if down && up
                push!(I,p); push!(J,dof[i,j-1]); push!(V,-1/(2h))
                push!(I,p); push!(J,dof[i,j+1]); push!(V, 1/(2h))
            elseif up
                push!(I,p); push!(J,p);          push!(V,-1/h)
                push!(I,p); push!(J,dof[i,j+1]); push!(V, 1/h)
            elseif down
                push!(I,p); push!(J,dof[i,j-1]); push!(V,-1/h)
                push!(I,p); push!(J,p);          push!(V, 1/h)
            end
        end
    end

    return sparse(I,J,V,N,N)
end


function assemble_idl_operator_masked(
    CG::CauchyGreenField,
    V::VelocityField,
    wet_mask::AbstractMatrix{Bool};
    space_stride::Int,
    time_stride::Int
)

    space_stride >= 1 || error("space_stride must be >= 1")
    time_stride >= 1 || error("time_stride must be >= 1")

    ix = collect(1:space_stride:length(CG.x))
    iy = collect(1:space_stride:length(CG.y))
    it = collect(1:time_stride:length(CG.times))

    x = CG.x[ix]
    y = CG.y[iy]
    t = CG.times[it]

    Nx = length(x)
    Ny = length(y)
    Nt = length(t)

    dx = x[2]-x[1]
    dy = y[2]-y[1]
    dt = mean(diff(t))

    wet = masked_idl_wet_grid(CG,V,wet_mask,ix,iy)
    wet_cells,active,dof,active_full = build_wet_cells_and_dofs(wet)

    Nxy = length(active_full)
    Nst = Nxy*Nt

    println()
    println("Assembling masked inflated dynamic Laplacian components")
    println("full space grid = $Nx x $Ny")
    println("active ocean nodes = $Nxy")
    println("wet cells = ",count(wet_cells))
    println("time levels = $Nt")
    println("space-time DOF = $Nst")
    println("dx = $dx km")
    println("dy = $dy km")
    println("dt = $dt h")

    Kblocks = Vector{SparseMatrixCSC{Float64,Int}}(undef,Nt)

    progress = Progress(Nt; desc="IDL masked spatial slices: ", dt=0.5)

    Threads.@threads for kk in 1:Nt

        k = it[kk]

        C11 = @view CG.C11[ix,iy,k]
        C12 = @view CG.C12[ix,iy,k]
        C22 = @view CG.C22[ix,iy,k]
        detC = @view CG.detC[ix,iy,k]

        A11 = fill(NaN,Nx,Ny)
        A12 = fill(NaN,Nx,Ny)
        A22 = fill(NaN,Nx,Ny)

        @inbounds for j in 1:Ny
            for i in 1:Nx

                active[i,j] || continue

                d = detC[i,j]

                if isfinite(d) && d > 0 &&
                   isfinite(C11[i,j]) && isfinite(C12[i,j]) && isfinite(C22[i,j])

                    J = sqrt(d)
                    A11[i,j] =  C22[i,j]/d * J
                    A12[i,j] = -C12[i,j]/d * J
                    A22[i,j] =  C11[i,j]/d * J
                end
            end
        end

        Kblocks[kk] = assemble_q1_tensor_stiffness_masked(
            A11,A12,A22,wet_cells,dof,dx,dy
        )

        next!(progress)
    end

    Kspace = blockdiag(Kblocks...)

    println("Building masked material derivative")

    Dx = derivative_matrix_masked_x(active,dof,dx)
    Dy = derivative_matrix_masked_y(active,dof,dy)
    Dt1 = derivative_matrix_1d(Nt,dt)

    Ixy = spdiagm(0 => ones(Float64,Nxy))
    Dt = kron(Dt1,Ixy)

    VI = build_interpolants(V)
    Adv_blocks = Vector{SparseMatrixCSC{Float64,Int}}(undef,Nt)

    active_ij = Tuple{Int,Int}[]
    for j in 1:Ny
        for i in 1:Nx
            active[i,j] && push!(active_ij,(i,j))
        end
    end

    sort!(active_ij, by = ij -> dof[ij[1],ij[2]])

    progress = Progress(Nt; desc="IDL masked advection blocks: ", dt=0.5)

    Threads.@threads for kk in 1:Nt

        uvec = Vector{Float64}(undef,Nxy)
        vvec = Vector{Float64}(undef,Nxy)

        for p in 1:Nxy
            i,j = active_ij[p]
            uu,vv = velocity(x[i],y[j],t[kk],VI)
            uvec[p] = uu
            vvec[p] = vv
        end

        Adv_blocks[kk] = spdiagm(0 => uvec)*Dx + spdiagm(0 => vvec)*Dy
        next!(progress)
    end

    Adv = blockdiag(Adv_blocks...)
    Dmat = Dt + Adv

    Mxy = build_lumped_mass_masked(wet_cells,dof,dx,dy)

    wt = fill(dt,Nt)
    wt[1] = dt/2
    wt[end] = dt/2
    Mt = spdiagm(0 => wt)

    G = kron(Mt,Mxy)

    println("Forming masked Kmaterial")

    Kmaterial = Dmat' * G * Dmat
    Kmaterial = 0.5*(Kmaterial + Kmaterial')

    println()
    println("Masked IDL component assembly complete")
    println("matrix size      = $Nst x $Nst")
    println("nnz(Kspace)      = ",nnz(Kspace))
    println("nnz(Kmaterial)   = ",nnz(Kmaterial))
    println("nnz(G)           = ",nnz(G))

    return MaskedIDLOperator(
        Kspace,
        Kmaterial,
        G,
        Nx,
        Ny,
        Nt,
        Nxy,
        Nst,
        collect(x),
        collect(y),
        collect(t),
        ix,
        iy,
        it,
        dx,
        dy,
        dt,
        BitMatrix(active),
        active_full,
        dof
    )
end


function assemble_idl_operator(
    CG::CauchyGreenField,
    V::VelocityField;
    space_stride::Int,
    time_stride::Int,
    wet_mask = nothing
)

    if wet_mask === nothing
        return assemble_idl_operator_rectangular(
            CG,V;
            space_stride=space_stride,
            time_stride=time_stride
        )
    end

    return assemble_idl_operator_masked(
        CG,V,wet_mask;
        space_stride=space_stride,
        time_stride=time_stride
    )
end


# ============================================================
# FORM K(a)
# ============================================================

function inflated_stiffness(
    O::AbstractIDLOperator,
    a::Real
)

    K =
        O.Kspace +
        Float64(a)^2 * O.Kmaterial

    return 0.5*(K + K')

end

# ============================================================
# IDL EIGENVALUE SWEEP
# ============================================================

struct IDLEigenResult

    a::Float64

    mu::Vector{Float64}
    V::Matrix{Float64}

    elapsed::Float64

end


# ============================================================
# LEADING IDL EIGENPAIRS FOR ONE VALUE OF a
#
# Solve the generalized eigenproblem
#
#       K(a) v = mu G v
#
# using the symmetric standard-form transformation
#
#       A q = mu q,
#
#       A = G^(-1/2) K(a) G^(-1/2),
#       v = G^(-1/2) q.
#
# We seek the eigenvalues closest to zero.  Since ARPACK is
# most effective at finding extremal eigenvalues, shift-invert
# is used to transform the desired near-zero eigenvalues into
# large-magnitude eigenvalues of the transformed problem.
#
# The small shift sigma is only a numerical device for this
# purpose; it has no physical or IDL significance.
#
# The constant mode is retained as a numerical diagnostic and
# discarded before SEBA.
# ============================================================

function idl_standard_components(
    O::AbstractIDLOperator
)

    #
    # G is diagonal in the present lumped FEM formulation.
    # Transform the two stiffness components only once:
    #
    #   A(a) = Aspace + a^2 Amaterial
    #
    # rather than rebuilding G^(-1/2) K(a) G^(-1/2)
    # independently for every a in the sweep.
    #

    g =
        LinearAlgebra.diag(O.G)

    any(g .<= 0.0) &&
        error(
            "Gram matrix contains nonpositive diagonal entries"
        )

    dinv =
        1.0 ./ sqrt.(g)

    Dinv =
        spdiagm(
            0 => dinv
        )

    Aspace =
        Dinv *
        O.Kspace *
        Dinv

    Amaterial =
        Dinv *
        O.Kmaterial *
        Dinv

    Aspace =
        0.5 * (
            Aspace +
            Aspace'
        )

    Amaterial =
        0.5 * (
            Amaterial +
            Amaterial'
        )

    dropzeros!(Aspace)
    dropzeros!(Amaterial)

    return Aspace,Amaterial,dinv

end


function solve_idl_standard_eigenproblem(
    Aspace::SparseMatrixCSC,
    Amaterial::SparseMatrixCSC,
    dinv::AbstractVector,
    a::Real;
    nmodes::Int,
    tol::Real = 1e-5,
    maxiter::Int = 300,
    sigma::Real = 1e-10
)

    a =
        Float64(a)

    sigma =
        abs(Float64(sigma))

    sigma > 0.0 ||
        error(
            "sigma must be positive"
        )

    #
    # Standard-form IDL operator.
    #

    A =
        Aspace +
        a^2 * Amaterial

    A =
        0.5 * (
            A +
            A'
        )

    dropzeros!(A)

    N =
        size(A,1)

    ncv =
        min(
            max(
                2 * nmodes + 1,
                30
            ),
            N
        )

    mu =
        Vector{Float64}()

    Q =
        Matrix{Float64}(undef,0,0)

    #
    # A is positive semidefinite and contains the constant
    # eigenmode at zero.  Shift-invert exactly at zero would
    # require factorizing a singular matrix.
    #
    # Use a small NEGATIVE shift instead:
    #
    #       sigma_ARPACK = -sigma
    #
    # so ARPACK factorizes
    #
    #       A + sigma I,
    #
    # which is nonsingular, while still targeting the
    # eigenvalues nearest zero.  eigs returns the eigenvalues
    # of A itself, so no shift correction is required.
    #

    elapsed =
        @elapsed begin

            mu_raw, Qraw =
                eigs(
                    A;
                    nev = nmodes,
                    ncv = ncv,
                    sigma = -sigma,
                    which = :LM,
                    tol = Float64(tol),
                    maxiter = maxiter,
                    ritzvec = true
                )

            mu =
                real.(mu_raw)

            Q =
                real.(Qraw)

        end

    #
    # Sort from smallest eigenvalue upward.
    #

    p =
        sortperm(mu)

    mu =
        mu[p]

    Q =
        Q[:,p]

    #
    # Transform back to generalized eigenvectors:
    #
    #       v = G^(-1/2) q.
    #
    # Scale rows directly rather than forming another sparse
    # diagonal matrix multiplication.
    #

    V =
        Q .* dinv

    return IDLEigenResult(
        a,
        mu,
        Matrix(V),
        elapsed
    )

end


function solve_idl_eigenproblem(
    O::AbstractIDLOperator,
    a::Real;
    nmodes::Int,
    tol::Real = 1e-5,
    maxiter::Int = 300,
    sigma::Real = 1e-10
)

    Aspace,Amaterial,dinv =
        idl_standard_components(O)

    return solve_idl_standard_eigenproblem(
        Aspace,
        Amaterial,
        dinv,
        a;
        nmodes = nmodes,
        tol = tol,
        maxiter = maxiter,
        sigma = sigma
    )

end


# ============================================================
# THREADED SWEEP OVER a
# ============================================================

function sweep_idl_eigenproblems(
    O::AbstractIDLOperator,
    a_values::AbstractVector;
    nmodes::Int,
    nworkers::Int,
    tol::Real = 1e-8,
    maxiter::Int = 3000,
    sigma::Real = 1e-10
)

    Na =
        length(a_values)

    nworkers =
        max(
            1,
            min(
                nworkers,
                Na
            )
        )

    println()
    println("IDL inflation-parameter sweep")
    println("a values       = $Na")
    println("modes per a    = $nmodes")
    println("workers        = $nworkers")
    println("Julia threads  = $(Threads.nthreads())")
    println("eigensolver tol= $tol")
    println("shift          = -$(abs(Float64(sigma)))")

    #
    # The Gram transformation is independent of a.  Build the
    # transformed spatial/material operators once for the whole
    # sweep.
    #

    println("Preparing standard-form IDL components")

    prep_elapsed =
        @elapsed begin

            Aspace,Amaterial,dinv =
                idl_standard_components(O)

        end

    println(
        "standard-form preparation = ",
        round(
            prep_elapsed,
            digits = 2
        ),
        " s"
    )

    results =
        Vector{Union{Nothing,IDLEigenResult}}(
            nothing,
            Na
        )

    #
    # Divide a-values into independent chunks.
    #
    # nworkers = 1 is the memory-conservative choice for very
    # large IDL systems.  Larger values allow independent
    # a-problems to run concurrently when memory permits.
    #

    edges =
        round.(
            Int,
            range(
                0,
                Na,
                length = nworkers + 1
            )
        )

    progress =
        Progress(
            Na;
            desc = "IDL eigensweep: ",
            dt = 0.5
        )

    elapsed =
        @elapsed begin

            Threads.@threads for worker in 1:nworkers

                i1 =
                    edges[worker] + 1

                i2 =
                    edges[worker+1]

                for ia in i1:i2

                    a =
                        Float64(
                            a_values[ia]
                        )

                    result =
                        solve_idl_standard_eigenproblem(
                            Aspace,
                            Amaterial,
                            dinv,
                            a;
                            nmodes = nmodes,
                            tol = tol,
                            maxiter = maxiter,
                            sigma = sigma
                        )

                    results[ia] =
                        result

                    next!(progress)

                end

            end

        end

    println(
        "total eigensweep time = ",
        round(
            elapsed,
            digits = 2
        ),
        " s"
    )

    return IDLEigenResult[
        results[i]::IDLEigenResult
        for i in eachindex(results)
    ]

end


# ============================================================
# LOCALIZED EIGENBASIS
# ============================================================

function varimax_rotation(
    X::AbstractMatrix;
    maxiter::Int = 200,
    tol::Real = 1e-8
)

    n, p = size(X)

    R = Matrix{Float64}(I,p,p)

    dold = 0.0

    for iter in 1:maxiter

        Z = X * R

        z2 = sum(Z.^2,dims=1) ./ n

        tmp = X' * (
            Z.^3 .-
            Z .* z2
        )

        F = svd(tmp)

        R = F.U * F.Vt

        d = sum(F.S)

        if dold > 0 &&
           abs(d-dold) < Float64(tol)*dold

            break
        end

        dold = d
    end

    return R
end


function localize_eigenbasis(
    V::AbstractMatrix;
    maxiter::Int = 200,
    tol::Real = 1e-8
)

    n, p = size(V)

    #
    # Thin QR factorization.
    #

    F = qr(V)

    Q = Matrix(F.Q[:,1:p])

    R = varimax_rotation(
        Q;
        maxiter = maxiter,
        tol = tol
    )

    S = Q * R

    #
    # Choose sign so that the largest-amplitude side is positive.
    #

    for j in 1:p

        if abs(minimum(@view S[:,j])) >
           abs(maximum(@view S[:,j]))

            S[:,j] .*= -1.0
        end
    end

    #
    # Retain positive support.
    #

    S[S .< 0.0] .= 0.0

    #
    # Normalize each localized vector.
    #

    for j in 1:p

        nrm = norm(@view S[:,j])

        if nrm > 0.0
            S[:,j] ./= nrm
        end
    end

    return S,R
end

# ============================================================
# IDL-SEBA RAYLEIGH DIAGNOSTICS
# ============================================================

struct IDLRayleighResult

    a::Float64

    S::Matrix{Float64}

    rho::Vector{Float64}
    rho_space::Vector{Float64}
    rho_material::Vector{Float64}
    rho_a2material::Vector{Float64}

    r_space::Vector{Float64}
    r_material::Vector{Float64}

    rho_space_median::Float64
    rho_a2material_median::Float64
    rho_total_median::Float64

end


function idl_rayleigh_diagnostics(
    O::AbstractIDLOperator,
    E::IDLEigenResult
)

    #
    # Discard constant mode.
    #

    X = copy(E.V[:,2:end])

    #
    # MATLAB normalized each eigenvector in the ordinary
    # Euclidean norm before localization.
    #

    for j in axes(X,2)

        nrm = norm(@view X[:,j])

        if nrm > 0.0
            X[:,j] ./= nrm
        end
    end

    S,R = localize_eigenbasis(X)

    nobj = size(S,2)

    rho            = zeros(Float64,nobj)
    rho_space      = zeros(Float64,nobj)
    rho_material   = zeros(Float64,nobj)
    rho_a2material = zeros(Float64,nobj)

    r_space    = zeros(Float64,nobj)
    r_material = zeros(Float64,nobj)

    a2 = E.a^2

    for j in 1:nobj

        s = @view S[:,j]

        den = dot(s,O.G*s)

        rs = dot(s,O.Kspace*s) / den
        rm = dot(s,O.Kmaterial*s) / den

        ram = a2 * rm
        rt  = rs + ram

        rho_space[j]      = rs
        rho_material[j]   = rm
        rho_a2material[j] = ram
        rho[j]            = rt

        r_space[j]    = rs / rt
        r_material[j] = ram / rt
    end

    return IDLRayleighResult(
        E.a,
        S,
        rho,
        rho_space,
        rho_material,
        rho_a2material,
        r_space,
        r_material,
        median(rho_space),
        median(rho_a2material),
        median(rho)
    )
end

# ============================================================
# SPATIAL-MATERIAL CROSSOVER
#
# For each a, compute the median candidate contributions
#
#       Rspace(a) = median rho_space
#
#       Rmaterial(a) = median a^2 rho_material.
#
# Select the sampled a for which the two contributions are
# closest on a logarithmic scale:
#
#       | log(Rspace / Rmaterial) |.
#
# This selects a representative point in the crossover
# regime.  It is a practical numerical choice, not a
# theoretical optimum of the IDL.
# ============================================================

struct IDLCrossoverResult

    diagnostics::Vector{IDLRayleighResult}

    a_selected::Float64
    selected_index::Int

    crossover_score::Vector{Float64}

end


function select_idl_crossover(
    O::AbstractIDLOperator,
    eig_results::Vector{IDLEigenResult}
)

    Na = length(eig_results)

    diagnostics =
        Vector{IDLRayleighResult}(undef,Na)

    score = fill(Inf,Na)

    progress = Progress(
        Na;
        desc = "IDL localization/crossover: ",
        dt = 0.5
    )

    for ia in 1:Na

        D = idl_rayleigh_diagnostics(
            O,
            eig_results[ia]
        )

        diagnostics[ia] = D

        Rs = D.rho_space_median
        Rm = D.rho_a2material_median

        if Rs > 0.0 &&
           Rm > 0.0 &&
           isfinite(Rs) &&
           isfinite(Rm)

            score[ia] =
                abs(log(Rs/Rm))
        end

        next!(progress)
    end

    iselect = argmin(score)

    return IDLCrossoverResult(
        diagnostics,
        diagnostics[iselect].a,
        iselect,
        score
    )
end

# ============================================================
# IDL-SEBA CANDIDATE EXTRACTION BY SUBPARTITION OF UNITY
#
# For the selected inflation parameter a:
#
#   1. take the positive part of each complete space-time SEBA
#      vector,
#   2. normalize each complete SEBA vector so that its global
#      space-time maximum is one, exactly as in Gary Froyland's
#      SEBA postprocessing,
#   3. determine one global subpartition-of-unity threshold
#      tau_pu from the full collection of normalized SEBA
#      vectors,
#   4. define candidate supports by S_m > tau_pu.
#
# No plateau detection, per-slice normalization, connected-
# component selection, or physical area threshold is used.
# ============================================================

struct IDLCandidateResult

    a::Float64

    # Positive, globally normalized space-time SEBA vectors.
    # Each column has maximum one.
    S::Matrix{Float64}

    # Same global subpartition threshold for every object.
    thresholds::Vector{Float64}

    masks::BitArray{4}

    rho::Vector{Float64}
    rho_space::Vector{Float64}
    rho_a2material::Vector{Float64}

    order::Vector{Int}

    tau_pu::Float64

    # Retained for compatibility with legacy downstream code.
    # No area filtering is performed in candidate extraction.
    min_area::Int
    max_area::Int

end

# ============================================================
# FINITE-LIFETIME IDL-SEBA EPISODES
#
# A localized IDL-SEBA candidate need not be active over the
# full analysis interval.  CandidateEpisode records one
# contiguous interval over which a candidate mask is nonempty.
#
# rank       : position in C.order (Rayleigh-quotient ordering)
# seba_index : column/mask index in the SEBA representation
# episode    : contiguous active episode number for this candidate
# birth_idx  : first active IDL time index
# death_idx  : last active IDL time index
# active_slices : number of consecutive active IDL slices
#
# Time values are intentionally not stored here.  The episode
# indices refer to the IDL time grid and can later be mapped to
# O.t[birth_idx] and O.t[death_idx].
# ============================================================

struct CandidateEpisode
    rank::Int
    seba_index::Int
    episode::Int
    birth_idx::Int
    death_idx::Int
    active_slices::Int
end

# ============================================================
# EXTRACT CONTIGUOUS ACTIVE EPISODES FROM IDL-SEBA MASKS
#
# Each candidate is active at time index k when its thresholded
# spatial mask contains at least one grid cell.  If a candidate
# disappears and later reappears, the disconnected periods are
# returned as separate CandidateEpisode records.
# ============================================================

function extract_candidate_episodes(
    C::IDLCandidateResult;
    verbose::Bool = true
)

    Nx,Ny,Nt,nobj = size(C.masks)

    length(C.order) == nobj ||
        error("C.order is inconsistent with candidate masks")

    episodes = CandidateEpisode[]

    if verbose
        println()
        println("IDL-SEBA finite-lifetime episodes")
        println("---------------------------------")
        println("candidates = ",nobj)
        println("time slices = ",Nt)
        println()
        println(" rank   SEBA object   episode   birth   death   active slices")
    end

    for (rank,m) in enumerate(C.order)

        active = falses(Nt)

        @inbounds for k in 1:Nt
            active[k] = any(@view C.masks[:,:,k,m])
        end

        k = 1
        iepisode = 0

        while k <= Nt

            # Skip inactive slices.
            while k <= Nt && !active[k]
                k += 1
            end

            k > Nt && break

            kb = k

            # Advance through this contiguous active interval.
            while k <= Nt && active[k]
                k += 1
            end

            kd = k - 1
            iepisode += 1

            ep = CandidateEpisode(
                rank,
                m,
                iepisode,
                kb,
                kd,
                kd-kb+1
            )

            push!(episodes,ep)

            if verbose
                println(
                    lpad(ep.rank,5), "   ",
                    lpad(ep.seba_index,11), "   ",
                    lpad(ep.episode,7), "   ",
                    lpad(ep.birth_idx,5), "   ",
                    lpad(ep.death_idx,5), "   ",
                    lpad(ep.active_slices,13)
                )
            end

        end

    end

    if verbose
        println()
        println("episodes = ",length(episodes))
        println(
            "candidates with >= 1 episode = ",
            length(unique([ep.seba_index for ep in episodes])),
            " / ",
            nobj
        )
    end

    return episodes
end

# ============================================================
# FILTER AND GROUP FINITE-LIFETIME EPISODES
#
# Raw episode extraction is deliberately permissive: even a
# single isolated active IDL slice is recorded. Such one-slice
# events are useful diagnostics of threshold crossings but are
# not promoted to finite-lifetime coherent episodes used later
# for FTRCS classification.
#
# Step 2 therefore:
#
#   1. retains only episodes with at least min_episode_slices
#      consecutive active IDL slices;
#
#   2. keeps rejected short episodes for diagnostics;
#
#   3. groups the retained episodes by their exact lifespan
#      (birth_idx,death_idx).
#
# All episodes with the same lifespan can later reuse one
# flow-map/LAVD calculation.
# ============================================================

struct CandidateEpisodeGroup
    birth_idx::Int
    death_idx::Int
    episodes::Vector{CandidateEpisode}
end


struct CandidateEpisodeGrouping
    min_episode_slices::Int
    retained::Vector{CandidateEpisode}
    rejected::Vector{CandidateEpisode}
    groups::Vector{CandidateEpisodeGroup}
end


function group_candidate_episodes(
    episodes::Vector{CandidateEpisode};
    min_episode_slices::Int = 2,
    verbose::Bool = true
)

    min_episode_slices >= 1 ||
        error("min_episode_slices must be >= 1")

    retained =
        CandidateEpisode[
            ep for ep in episodes
            if ep.active_slices >= min_episode_slices
        ]

    rejected =
        CandidateEpisode[
            ep for ep in episodes
            if ep.active_slices < min_episode_slices
        ]

    grouped =
        Dict{
            Tuple{Int,Int},
            Vector{CandidateEpisode}
        }()

    for ep in retained

        key =
            (
                ep.birth_idx,
                ep.death_idx
            )

        push!(
            get!(
                grouped,
                key,
                CandidateEpisode[]
            ),
            ep
        )

    end

    keys_sorted =
        sort(
            collect(keys(grouped));
            by = key -> (key[1],key[2])
        )

    groups =
        CandidateEpisodeGroup[]

    for key in keys_sorted

        eps =
            grouped[key]

        sort!(
            eps;
            by = ep -> (ep.rank,ep.episode)
        )

        push!(
            groups,
            CandidateEpisodeGroup(
                key[1],
                key[2],
                eps
            )
        )

    end

    if verbose

        println()
        println("IDL-SEBA finite-lifetime episode filtering")
        println("------------------------------------------")
        println(
            "minimum consecutive active slices = ",
            min_episode_slices
        )
        println(
            "raw episodes                      = ",
            length(episodes)
        )
        println(
            "retained episodes                 = ",
            length(retained)
        )
        println(
            "rejected short episodes           = ",
            length(rejected)
        )

        if !isempty(rejected)

            println()
            println("Rejected short episodes")
            println("-----------------------")
            println(
                " rank   SEBA object   episode   birth   death   active slices"
            )

            for ep in rejected

                println(
                    lpad(ep.rank,5), "   ",
                    lpad(ep.seba_index,11), "   ",
                    lpad(ep.episode,7), "   ",
                    lpad(ep.birth_idx,5), "   ",
                    lpad(ep.death_idx,5), "   ",
                    lpad(ep.active_slices,13)
                )

            end

        end

        println()
        println("IDL-SEBA lifespan groups")
        println("------------------------")
        println(
            " group   birth   death   slices   episodes   SEBA objects"
        )

        for (ig,G) in enumerate(groups)

            objects =
                join(
                    [
                        string(ep.seba_index)
                        for ep in G.episodes
                    ],
                    ","
                )

            println(
                lpad(ig,6), "   ",
                lpad(G.birth_idx,5), "   ",
                lpad(G.death_idx,5), "   ",
                lpad(G.death_idx-G.birth_idx+1,6), "   ",
                lpad(length(G.episodes),8), "   ",
                objects
            )

        end

        println()
        println(
            "unique retained lifespan groups = ",
            length(groups)
        )

    end

    return CandidateEpisodeGrouping(
        min_episode_slices,
        retained,
        rejected,
        groups
    )
end

# ============================================================
# LARGEST 8-CONNECTED COMPONENT OF A BINARY 2-D MASK
# ============================================================

function largest_component(
    mask::AbstractMatrix{Bool};
    min_area::Int = 20
)

    Nx, Ny = size(mask)

    nnz_mask = count(mask)

    if nnz_mask < min_area
        return falses(Nx,Ny)
    end

    visited = falses(Nx,Ny)

    best_component = CartesianIndex{2}[]
    component = CartesianIndex{2}[]

    queue = Vector{CartesianIndex{2}}(
        undef,
        nnz_mask
    )

    @inbounds for j in 1:Ny
        for i in 1:Nx

            if !mask[i,j] || visited[i,j]
                continue
            end

            empty!(component)

            head = 1
            tail = 1

            queue[1] = CartesianIndex(i,j)
            visited[i,j] = true

            while head <= tail

                I = queue[head]
                head += 1

                push!(component,I)

                ii = I[1]
                jj = I[2]

                for dj in -1:1
                    for di in -1:1

                        if di == 0 && dj == 0
                            continue
                        end

                        iii = ii + di
                        jjj = jj + dj

                        if 1 <= iii <= Nx &&
                           1 <= jjj <= Ny &&
                           mask[iii,jjj] &&
                           !visited[iii,jjj]

                            tail += 1
                            queue[tail] =
                                CartesianIndex(iii,jjj)

                            visited[iii,jjj] = true
                        end
                    end
                end
            end

            if length(component) >
               length(best_component)

                best_component =
                    copy(component)
            end
        end
    end

    out = falses(Nx,Ny)

    if length(best_component) >= min_area

        @inbounds for I in best_component
            out[I] = true
        end

    end

    return out
end


function expand_idl_vectors(
    O::IDLOperator,
    S::AbstractMatrix
)
    return reshape(S,O.Nx,O.Ny,O.Nt,size(S,2))
end

function expand_idl_vectors(
    O::MaskedIDLOperator,
    S::AbstractMatrix
)

    nobj = size(S,2)
    Q = zeros(Float64,O.Nx,O.Ny,O.Nt,nobj)

    for m in 1:nobj
        for k in 1:O.Nt
            src = @view S[(k-1)*O.Nxy+1:k*O.Nxy,m]
            dst = @view Q[:,:,k,m]
            dst[O.active_full] .= src
        end
    end

    return Q
end


# ============================================================
# EXTRACT IDL-SEBA CANDIDATES AT SELECTED a
# ============================================================

function extract_idl_candidates(
    O::AbstractIDLOperator,
    crossover::IDLCrossoverResult
)

    isel = crossover.selected_index
    D = crossover.diagnostics[isel]

    Sraw = D.S
    nobj = size(Sraw,2)

    size(Sraw,1) == O.Nst ||
        error("SEBA vectors are inconsistent with IDL grid")

    # --------------------------------------------------------
    # Gary-style SEBA normalization.
    #
    # Work with the positive part and normalize each COMPLETE
    # space-time SEBA vector by its own global maximum.
    # This is not a separate normalization at each time slice.
    # --------------------------------------------------------

    S = max.(Sraw,0.0)

    println()
    println("Extracting IDL-SEBA candidates by subpartition of unity")
    println("--------------------------------------------------------")
    println("a       = ",D.a)
    println("objects = ",nobj)

    for m in 1:nobj

        sm = maximum(@view S[:,m])

        isfinite(sm) && sm > 0.0 ||
            error("SEBA object $m has no positive finite support")

        @views S[:,m] ./= sm

    end

    colmax = [maximum(@view S[:,m]) for m in 1:nobj]

    println(
        "normalized column maxima = ",
        minimum(colmax)," ... ",maximum(colmax)
    )

    # --------------------------------------------------------
    # Gary Froyland's subpartition_unity construction.
    #
    # At each space-time point, sort object memberships in
    # descending order and form their cumulative sum.  tau_pu
    # is the largest SEBA value participating wherever that
    # cumulative sum exceeds one.
    # --------------------------------------------------------

    S_descend = sort(S; dims=2, rev=true)
    S_sum = cumsum(S_descend; dims=2)

    violating_values =
        S_descend[S_sum .> 1.0]

    tau_pu =
        isempty(violating_values) ?
        0.0 :
        maximum(violating_values)

    # Strict inequality reproduces Gary's
    #     S(S <= taupu) = 0
    # convention.
    support = S .> tau_pu

    # Verify the subpartition condition using the thresholded
    # SEBA amplitudes, not merely the Boolean masks.
    S_pu = S .* support
    row_sums = vec(sum(S_pu; dims=2))

    max_sum = maximum(row_sums)
    nviol =
        count(x -> x > 1.0 + 1e-12,row_sums)

    println()
    println("Subpartition threshold")
    println("----------------------")
    println("tau_pu = ",tau_pu)
    println("max sum_m S_m after threshold = ",max_sum)
    println("rows with sum > 1             = ",nviol)

    nviol == 0 ||
        error("Subpartition-of-unity check failed")

    # --------------------------------------------------------
    # Expand the thresholded supports back to x-y-time-object.
    # No connected-component pruning is performed.
    # --------------------------------------------------------

    Nx,Ny,Nt = O.Nx,O.Ny,O.Nt

    # Expand the thresholded IDL vectors from the active
    # space-time DOFs back onto the complete x-y-time grid.
    #
    # For MaskedIDLOperator, O.Nst is smaller than Nx*Ny*Nt,
    # so a direct reshape is not valid.

    support_full =
        expand_idl_vectors(
            O,
            Float64.(support)
        )

    masks =
        BitArray(
            support_full .> 0.5
        )

    thresholds = fill(tau_pu,nobj)

    order = sortperm(D.rho)

    println()
    println("Subpartition-defined IDL-SEBA supports")
    println("--------------------------------------")
    println(" object   active slices   median area [cells]")

    for m in order

        active_areas = Int[]

        for k in 1:Nt
            A = count(@view masks[:,:,k,m])
            A > 0 && push!(active_areas,A)
        end

        medA =
            isempty(active_areas) ?
            NaN :
            median(active_areas)

        @printf(
            "%6d      %3d / %3d          %8.1f\n",
            m,
            length(active_areas),
            Nt,
            medA
        )

    end

    return IDLCandidateResult(
        D.a,
        S,
        thresholds,
        masks,
        copy(D.rho),
        copy(D.rho_space),
        copy(D.rho_a2material),
        order,
        tau_pu,
        1,
        typemax(Int)
    )

end


# ============================================================
# LAVD
# ============================================================

struct LAVDField

    lavd::Array{Float64,3}
    omega_mean::Vector{Float64}

    x::Vector{Float64}
    y::Vector{Float64}
    times::Vector{Float64}

end

# ============================================================
# VORTICITY
#
#       omega = dv/dx - du/dy
#
# evaluated on the native velocity grid.
#
# Velocity arrays use the (x,y,t) convention:
#
#       dimension 1 -> x
#       dimension 2 -> y
# ============================================================

function compute_vorticity(
    V::VelocityField
)

    Nx, Ny, Nt = size(V.u)

    omega =
        Array{Float64}(undef,Nx,Ny,Nt)

    omega_mean =
        Vector{Float64}(undef,Nt)

    println()
    println("Computing vorticity")

    progress = Progress(
        Nt;
        desc = "Vorticity slices: ",
        dt = 0.5
    )

    Threads.@threads for k in 1:Nt

        u =
            @view V.u[:,:,k]

        v =
            @view V.v[:,:,k]

        #
        # Spatial derivatives:
        #
        #     dv/dx -> derivative along dimension 1
        #     du/dy -> derivative along dimension 2
        #

        dv_dx =
            derivative_x(
                v,
                V.dx
            )

        du_dy =
            derivative_y(
                u,
                V.dy
            )

        w =
            dv_dx .-
            du_dy

        omega[:,:,k] .= w

        vals =
            w[isfinite.(w)]

        omega_mean[k] =
            isempty(vals) ?
            NaN :
            mean(vals)

        next!(progress)

    end

    return omega,omega_mean

end

# ============================================================
# LAVD OVER A SUBINTERVAL OF AN EXISTING FULL FLOW MAP
#
# The flow map F is NOT recomputed.  Its trajectories retain
# the original material labels x0 at the initial time of the
# full IDL analysis.
#
# For each x0, this computes
#
#   integral from ta to tb of
#   |omega(x(t;x0),t)-<omega>(t)| dt
#
# using the same right-endpoint rectangular rule as
# compute_lavd.
# ============================================================

function compute_lavd_interval(
    V::VelocityField,
    F::FlowMap,
    omega::Array{Float64,3},
    omega_mean::Vector{Float64},
    ta::Real,
    tb::Real
)

    size(omega) == size(V.u) ||
        error(
            "Precomputed vorticity is inconsistent with velocity grid"
        )

    length(omega_mean) == length(V.t) ||
        error(
            "Precomputed mean vorticity is inconsistent with velocity times"
        )

    ta < tb ||
        error("LAVD interval must satisfy ta < tb")

    #
    # Find the full-flow-map times corresponding to ta and tb.
    #
    # The IDL episode times should lie on the flow-map time grid.
    # We nevertheless use nearest indices and verify the match.
    #

    ka = argmin(abs.(F.times .- ta))
    kb = argmin(abs.(F.times .- tb))

    tol =
        100*eps(Float64) *
        max(
            1.0,
            abs(Float64(ta)),
            abs(Float64(tb))
        )

    abs(F.times[ka]-ta) <= tol ||
        error(
            "Episode birth time $ta h is not on the flow-map time grid; " *
            "nearest flow-map time is $(F.times[ka]) h"
        )

    abs(F.times[kb]-tb) <= tol ||
        error(
            "Episode death time $tb h is not on the flow-map time grid; " *
            "nearest flow-map time is $(F.times[kb]) h"
        )

    ka < kb ||
        error("LAVD interval contains fewer than two flow-map times")

    #
    # Linear x-y-t interpolation of vorticity.
    #

    W = interpolate(
        (V.x,V.y,V.t),
        omega,
        Gridded(Linear())
    )

    Nx = length(F.xseed)
    Ny = length(F.yseed)

    lavd_score = zeros(Float64,Nx,Ny)

    xmin,xmax = first(V.x),last(V.x)
    ymin,ymax = first(V.y),last(V.y)
    tmin,tmax = first(V.t),last(V.t)

    println()
    println("Accumulating finite-lifetime LAVD")
    println("full flow-map window = ",
            first(F.times)," -- ",last(F.times)," h")
    println("LAVD interval        = ",
            F.times[ka]," -- ",F.times[kb]," h")
    println("material grid        = $Nx x $Ny")

    progress = Progress(
        kb-ka;
        desc = "LAVD slices: ",
        dt = 0.5
    )

    #
    # IMPORTANT:
    #
    # F.PhiX[:,:,k] and F.PhiY[:,:,k] remain indexed by the
    # original material labels x0.  We merely begin accumulating
    # at ka+1 rather than constructing a new flow map at ta.
    #

    for k in (ka+1):kb

        tk = F.times[k]
        dtk = F.times[k]-F.times[k-1]

        #
        # Nearest native velocity time, matching compute_lavd
        # and the MATLAB implementation.
        #

        kt = argmin(abs.(V.t .- tk))

        wmean = omega_mean[kt]

        @inbounds for j in 1:Ny
            for i in 1:Nx

                #
                # Once a material trajectory has become invalid
                # during this LAVD interval, keep its score NaN.
                #

                if !isfinite(lavd_score[i,j])
                    continue
                end

                x = F.PhiX[i,j,k]
                y = F.PhiY[i,j,k]

                if !isfinite(x) ||
                   !isfinite(y) ||
                   x < xmin || x > xmax ||
                   y < ymin || y > ymax ||
                   tk < tmin || tk > tmax

                    lavd_score[i,j] = NaN
                    continue
                end

                w = W(x,y,tk)

                if isfinite(w) &&
                   isfinite(wmean)

                    lavd_score[i,j] +=
                        dtk*abs(w-wmean)

                else

                    lavd_score[i,j] = NaN

                end
            end
        end

        next!(progress)
    end

    return lavd_score

end

# ============================================================
# FINITE-LIFETIME LAVD CLASSIFICATION (STEP 4)
# ============================================================

struct EpisodeFTRCSRecord
    rank::Int
    seba_index::Int
    episode::Int
    birth_idx::Int
    death_idx::Int
    birth_time::Float64
    death_time::Float64
    active_slices::Int
    area_birth::Int
    lavd_candidate::Float64
    lavd_background::Float64
    E_LAVD::Float64
    pass::Bool
end

struct EpisodeFTRCSResult
    records::Vector{EpisodeFTRCSRecord}
    ratio_min::Float64
    n_groups::Int
end

function classify_ftrcs_episodes(
    V::VelocityField,
    O::AbstractIDLOperator,
    C::IDLCandidateResult,
    grouping::CandidateEpisodeGrouping,
    F::FlowMap;
    lavd_ratio_min::Real = 1.0,
    verbose::Bool = true
)
    println()
    println("Preparing finite-lifetime LAVD classification")
    println("---------------------------------------------")
    println("retained episodes = ",length(grouping.retained))
    println("lifespan groups   = ",length(grouping.groups))
    println("E_LAVD minimum    = ",lavd_ratio_min)

    omega,omega_mean = compute_vorticity(V)
    records = EpisodeFTRCSRecord[]

    # LAVD classification uses the same material survival domain
    # on which the IDL operator and SEBA objects were constructed.
    #
    # C.masks is indexed by the original material labels at the
    # initial time of the full IDL analysis.  Therefore this mask
    # must remain in those same material coordinates.
    domain_mask =
        O isa MaskedIDLOperator ?
        copy(O.wet_mask) :
        trues(O.Nx,O.Ny)

    verbose && println(
        "IDL material survival domain = ",
        count(domain_mask),
        " / ",
        length(domain_mask),
        " initial material labels"
    )

    for (ig,G) in enumerate(grouping.groups)

        ta = O.t[G.birth_idx]
        tb = O.t[G.death_idx]

        ta < tb ||
            error(
                "Retained episode group must span at least two IDL times"
            )

        verbose && println(
            "\nLifespan group ",ig," / ",length(grouping.groups),
            ": slices ",G.birth_idx,":",G.death_idx,
            "  time ",ta," -- ",tb," h",
            "  episodes = ",length(G.episodes)
        )

		 lavd_score =
            compute_lavd_interval(
                V,
                F,
                omega,
                omega_mean,
                ta,
                tb
            )

	    any(.!isfinite.(lavd_score[domain_mask])) &&
            error(
                "Non-finite LAVD found inside the IDL material " *
                "survival domain"
            )		

        for ep in G.episodes

            candidate_mask =
                (@view C.masks[
                    :,
                    :,
                    G.birth_idx,
                    ep.seba_index
                ]) .&
                domain_mask

            area_birth = count(candidate_mask)

            area_birth > 0 ||
                error(
                    "Retained episode has empty birth mask"
                )

            background_mask =
                domain_mask .&
                .!candidate_mask

            count(background_mask) > 0 ||
                error(
                    "FTRCS candidate fills the complete admissible " *
                    "material domain"
                )

            lavd_candidate =
                mean(
                    lavd_score[candidate_mask]
                )

            lavd_background =
                mean(
                    lavd_score[background_mask]
                )

            E =
                lavd_background > 0 ?
                lavd_candidate/lavd_background :
                NaN

            pass =
                isfinite(E) &&
                E >= Float64(lavd_ratio_min)

            push!(
                records,
                EpisodeFTRCSRecord(
                    ep.rank,
                    ep.seba_index,
                    ep.episode,
                    ep.birth_idx,
                    ep.death_idx,
                    ta,
                    tb,
                    ep.active_slices,
                    area_birth,
                    lavd_candidate,
                    lavd_background,
                    E,
                    pass
                )
            )

        end
    end

    sort!(
        records;
        by = R -> (
            R.birth_idx,
            R.death_idx,
            R.rank,
            R.episode
        )
    )

    if verbose

        println()
        println("Finite-lifetime FTRCS classification")
        println("------------------------------------")
        println(
            " object  ep   birth[h]  death[h]  slices  area0   ",
            "<LAVD>_A   <LAVD>_B   E_LAVD   FTRCS"
        )

        for R in records

            @printf(
                "%6d %3d   %8.3f  %8.3f   %5d  %5d   %8.4f   %8.4f   %7.3f   %s\n",
                R.seba_index,
                R.episode,
                R.birth_time,
                R.death_time,
                R.active_slices,
                R.area_birth,
                R.lavd_candidate,
                R.lavd_background,
                R.E_LAVD,
                R.pass ? "yes" : "no"
            )

        end

        println(
            "\nfinite-lifetime FTRCS selected = ",
            count(R -> R.pass,records),
            " / ",
            length(records)
        )

    end

    return EpisodeFTRCSResult(
        records,
        Float64(lavd_ratio_min),
        length(grouping.groups)
    )

end

# ============================================================
# FTRCS OVERLAP / REDUNDANCY DIAGNOSTICS
#
# Compare all LAVD-selected finite-lifetime FTRCS over their
# common active time interval.
#
# Jaccard:
#
#       J(A,B) = |A intersection B| / |A union B|
#
# Containment:
#
#       C(A,B) = |A intersection B| / min(|A|,|B|)
#
# C is useful for identifying nested structures: a small set
# may be almost completely contained in a larger set even when
# their Jaccard overlap is modest.
#
# This routine is DIAGNOSTIC ONLY.  It does not merge, reject,
# rank, or otherwise modify any FTRCS.
# ============================================================

function diagnose_ftrcs_overlap(
    C::IDLCandidateResult,
    R::EpisodeFTRCSResult;
    verbose::Bool = true
)

    selected = [r for r in R.records if r.pass]
    n = length(selected)

    if verbose
        println()
        println("FTRCS overlap / redundancy diagnostics")
        println("--------------------------------------")
        println("selected FTRCS = ",n)
    end

    n < 2 && return nothing

    if verbose
        println()
        println(
            " obj.i  ep.i   obj.j  ep.j   common",
            "    mean J     max J",
            "     mean C     min C"
        )
    end

    for i in 1:n-1

        Arec = selected[i]

        for j in i+1:n

            Brec = selected[j]

            k0 = max(Arec.birth_idx,Brec.birth_idx)
            k1 = min(Arec.death_idx,Brec.death_idx)

            k0 <= k1 || continue

            Jvals = Float64[]
            Cvals = Float64[]

            for k in k0:k1

                A = @view C.masks[:,:,k,Arec.seba_index]
                B = @view C.masks[:,:,k,Brec.seba_index]

                nA = count(A)
                nB = count(B)

                # Both episodes should be active over the common
                # interval, but guard against an empty slice.
                (nA > 0 && nB > 0) || continue

                nAB = count(A .& B)
                nAU = nA + nB - nAB

                J = nAU > 0 ?
                    nAB / nAU :
                    0.0

                containment =
                    nAB / min(nA,nB)

                push!(Jvals,J)
                push!(Cvals,containment)

            end

            isempty(Jvals) && continue

            if verbose
                @printf(
                    "%6d %5d   %6d %5d   %6d   %8.3f  %8.3f   %8.3f  %8.3f\n",
                    Arec.seba_index,
                    Arec.episode,
                    Brec.seba_index,
                    Brec.episode,
                    length(Jvals),
                    mean(Jvals),
                    maximum(Jvals),
                    mean(Cvals),
                    minimum(Cvals)
                )
            end
        end
    end

    return nothing
end


# ============================================================
# FTRCS CLASSIFICATION
#
# An IDL-SEBA candidate is rotationally enriched when
#
#       E_LAVD =
#
#         mean LAVD inside A_j(t0)
#         -----------------------
#         mean LAVD over IDL domain
#
# satisfies
#
#       E_LAVD >= lavd_ratio_min.
# ============================================================

struct FTRCSResult

    objects::Vector{Int}

    rho::Vector{Float64}
    area0::Vector{Int}

    E_LAVD::Vector{Float64}
    pass::BitVector

    lavd_idl::Matrix{Float64}
    lavd_domain::Float64

    ratio_min::Float64

end


function classify_ftrcs(
    O::AbstractIDLOperator,
    C::IDLCandidateResult,
    L::LAVDField;
    lavd_ratio_min::Real = 1.0
)

    #
    # Final accumulated LAVD is a scalar field on the
    # trajectory/CG initial-condition grid.
    #

    lavd_final =
        @view L.lavd[:,:,end]

    #
    # Map the physical IDL grid directly onto the LAVD grid.
    #
    # The IDL grid is an exact spatial subset of the finer
    # trajectory/LAVD grid, but its stored index maps need not
    # coincide with the physical LAVD indexing after grid
    # transformations. Determine the matching indices from
    # the physical coordinates instead.
    #

    ixL =
        [
            argmin(abs.(L.x .- xi))
            for xi in O.x
        ]

    iyL =
        [
            argmin(abs.(L.y .- yi))
            for yi in O.y
        ]

    dxerr =
        maximum(
            abs.(
                L.x[ixL] .- O.x
            )
        )

    dyerr =
        maximum(
            abs.(
                L.y[iyL] .- O.y
            )
        )

    tol =
        100.0 * eps(Float64) *
        max(
            1.0,
            maximum(abs.(O.x)),
            maximum(abs.(O.y))
        )

    dxerr <= tol ||
        error(
            "IDL x grid does not coincide with LAVD grid"
        )

    dyerr <= tol ||
        error(
            "IDL y grid does not coincide with LAVD grid"
        )

    lavd_idl =
        Matrix(
            lavd_final[ixL,iyL]
        )

    #
    # For the rotational-enrichment diagnostic, unavailable
    # accumulated LAVD contributes zero rather than being
    # omitted. Keep lavd_idl unchanged for output.
    #

    lavd_score =
        copy(lavd_idl)

    lavd_score[.!isfinite.(lavd_score)] .= 0.0

    #
    # Reference LAVD mean over the complete physical IDL domain.
    #

    if O isa MaskedIDLOperator

        domain_mask =
            copy(O.wet_mask)

    else

        domain_mask =
            trues(
                O.Nx,
                O.Ny
            )

    end

    domain_vals =
        lavd_score[domain_mask]

    isempty(domain_vals) &&
        error(
            "No IDL-domain points available for LAVD classification"
        )

    lavd_domain =
        mean(domain_vals)

    #
    # Candidates are already ordered by increasing
    # Rayleigh quotient.
    #

    objects =
        copy(C.order)

    nobj =
        length(objects)

    rho =
        Vector{Float64}(undef,nobj)

    area0 =
        zeros(Int,nobj)

    E =
        fill(NaN,nobj)

    pass =
        falses(nobj)

    println()
    println("Classifying FTRCS by LAVD enrichment")
    println("candidates      = ",nobj)
    println("domain LAVD     = ",lavd_domain)
    println("E_LAVD minimum  = ",lavd_ratio_min)

    for io in 1:nobj

        m =
            objects[io]

        #
        # Complete initial IDL-SEBA support A_j(t0), restricted
        # only to the physical IDL domain.
        #

        mask =
            @view C.masks[:,:,1,m]

        candidate_mask =
            mask .&
            domain_mask

        area0[io] =
            count(candidate_mask)

        rho[io] =
            C.rho[m]

        if area0[io] < C.min_area
            continue
        end

        E[io] =
            mean(
                lavd_score[candidate_mask]
            ) /
            lavd_domain

        pass[io] =
            E[io] >=
            Float64(lavd_ratio_min)

    end

    return FTRCSResult(
        objects,
        rho,
        area0,
        E,
        pass,
        lavd_idl,
        lavd_domain,
        Float64(lavd_ratio_min)
    )

end

# ============================================================
# NETCDF OUTPUT
#
# Save the scientific products of the FTRCS calculation.
# Large numerical intermediates (flow maps, Cauchy-Green
# tensors, FEM matrices) are intentionally not included.
# ============================================================

function save_ftrcs_netcdf(
    filename::AbstractString,
    O::AbstractIDLOperator,
    eig_results::Vector{IDLEigenResult},
    crossover::IDLCrossoverResult,
    C::IDLCandidateResult;
    ftrcs::Union{Nothing,FTRCSResult} = nothing,
    episode_ftrcs::Union{Nothing,EpisodeFTRCSResult} = nothing,
    t0::Real,
    t1::Real
)

    Na = length(eig_results)
    Nm = length(eig_results[1].mu)

    #
    # Store all localized IDL-SEBA candidates in increasing
    # Rayleigh-quotient order.  Rayleigh screening is stored
    # separately as a candidate flag.
    #

    order = C.order
    Nc = length(order)

    #
    # Selected-a diagnostics.
    #

    Dsel = crossover.diagnostics[crossover.selected_index]

    #
    # Sweep diagnostics.
    #

    rho_space_med =
        [D.rho_space_median for D in crossover.diagnostics]

    rho_a2material_med =
        [D.rho_a2material_median for D in crossover.diagnostics]

    rho_total_med =
        [D.rho_total_median for D in crossover.diagnostics]

    eigenvalues =
        hcat([E.mu for E in eig_results]...)

    #
    # SEBA and masks in candidate-rank order.
    #

    S4 = expand_idl_vectors(O,C.S)

    seba_out =
        Array{Float64}(undef,O.Nx,O.Ny,O.Nt,Nc)

    mask_out =
        Array{Int8}(undef,O.Nx,O.Ny,O.Nt,Nc)

    threshold_out = Vector{Float64}(undef,Nc)
    rho_out       = Vector{Float64}(undef,Nc)
    rho_space_out = Vector{Float64}(undef,Nc)
    rho_mat_out   = Vector{Float64}(undef,Nc)
    r_space_out   = Vector{Float64}(undef,Nc)
    r_mat_out     = Vector{Float64}(undef,Nc)
    active_out    = Vector{Int32}(undef,Nc)

    for ic in 1:Nc

        m = order[ic]

        seba_out[:,:,:,ic] .= S4[:,:,:,m]

        mask_out[:,:,:,ic] .=
            Int8.(C.masks[:,:,:,m])

        threshold_out[ic] = C.thresholds[m]
        rho_out[ic]       = Dsel.rho[m]
        rho_space_out[ic] = Dsel.rho_space[m]

        #
        # Unweighted material Rayleigh contribution.
        #

        rho_mat_out[ic] =
            Dsel.rho_material[m]

        r_space_out[ic] =
            Dsel.r_space[m]

        r_mat_out[ic] =
            Dsel.r_material[m]

        active_out[ic] =
            count(
                k -> any(@view C.masks[:,:,k,m]),
                1:O.Nt
            )
    end

    #
    # Write NetCDF.
    #

    ds = NCDataset(filename,"c")

    try

        defDim(ds,"x",O.Nx)
        defDim(ds,"y",O.Ny)
        defDim(ds,"time",O.Nt)
        defDim(ds,"candidate",Nc)
        defDim(ds,"a",Na)
        defDim(ds,"mode",Nm)

        if episode_ftrcs !== nothing
            defDim(ds,"episode",length(episode_ftrcs.records))
        end

        vx = defVar(ds,"x",Float64,("x",))
        vy = defVar(ds,"y",Float64,("y",))
        vt = defVar(ds,"time",Float64,("time",))

        vx.attrib["units"] = "km"
        vy.attrib["units"] = "km"
        vt.attrib["units"] = "h"

        vx[:] = O.x
        vy[:] = O.y
        vt[:] = O.t

        if O isa MaskedIDLOperator
            vwet = defVar(ds,"wet_mask",Int8,("x","y"))
            vwet[:,:] = Int8.(O.wet_mask)
            vwet.attrib["description"] =
                "1 for active ocean FEM nodes; 0 outside the masked IDL domain"
        end

        va = defVar(ds,"a_values",Float64,("a",))
        va[:] = [E.a for E in eig_results]

        vmu = defVar(
            ds,
            "eigenvalue",
            Float64,
            ("mode","a")
        )
        vmu[:,:] = eigenvalues

        vrs = defVar(
            ds,
            "rho_space_median",
            Float64,
            ("a",)
        )

        vrm = defVar(
            ds,
            "rho_a2material_median",
            Float64,
            ("a",)
        )

        vrt = defVar(
            ds,
            "rho_total_median",
            Float64,
            ("a",)
        )

        vcs = defVar(
            ds,
            "crossover_score",
            Float64,
            ("a",)
        )

        vrs[:] = rho_space_med
        vrm[:] = rho_a2material_med
        vrt[:] = rho_total_med
        vcs[:] = crossover.crossover_score

        vseba = defVar(
            ds,
            "seba",
            Float32,
            ("x","y","time","candidate")
        )

        vmask = defVar(
            ds,
            "candidate_mask",
            Int8,
            ("x","y","time","candidate")
        )

        vseba[:,:,:,:] = Float32.(seba_out)
        vmask[:,:,:,:] = mask_out

        vthr = defVar(
            ds,
            "candidate_threshold",
            Float64,
            ("candidate",)
        )

        vrho = defVar(
            ds,
            "candidate_rho",
            Float64,
            ("candidate",)
        )

        vrhos = defVar(
            ds,
            "candidate_rho_space",
            Float64,
            ("candidate",)
        )

        vrhom = defVar(
            ds,
            "candidate_rho_material",
            Float64,
            ("candidate",)
        )

        vfrs = defVar(
            ds,
            "candidate_r_space",
            Float64,
            ("candidate",)
        )

        vfrm = defVar(
            ds,
            "candidate_r_material",
            Float64,
            ("candidate",)
        )

        vactive = defVar(
            ds,
            "candidate_active_slices",
            Int32,
            ("candidate",)
        )

        voriginal = defVar(
            ds,
            "candidate_seba_index",
            Int32,
            ("candidate",)
        )

        vthr[:]      = threshold_out
        vrho[:]      = rho_out
        vrhos[:]     = rho_space_out
        vrhom[:]     = rho_mat_out
        vfrs[:]      = r_space_out
        vfrm[:]      = r_mat_out
        vactive[:]   = active_out
        voriginal[:] = Int32.(order)

        #
        # Finite-lifetime episode-level FTRCS classification.
        #

        if episode_ftrcs !== nothing

            R = episode_ftrcs.records
            Ne = length(R)

            vobj = defVar(ds,"episode_seba_index",Int32,("episode",))
            veno = defVar(ds,"episode_number",Int32,("episode",))
            vbi  = defVar(ds,"episode_birth_index",Int32,("episode",))
            vdi  = defVar(ds,"episode_death_index",Int32,("episode",))
            vbt  = defVar(ds,"episode_birth_time",Float64,("episode",))
            vdt  = defVar(ds,"episode_death_time",Float64,("episode",))
            vns  = defVar(ds,"episode_active_slices",Int32,("episode",))
            vab  = defVar(ds,"episode_birth_area_cells",Int32,("episode",))
            vLA  = defVar(ds,"episode_LAVD_candidate_mean",Float64,("episode",))
            vLB  = defVar(ds,"episode_LAVD_background_mean",Float64,("episode",))
            vEE  = defVar(ds,"episode_E_LAVD",Float64,("episode",))
            vFF  = defVar(ds,"episode_FTRCS",Int8,("episode",))

            vobj[:] = Int32.([r.seba_index for r in R])
            veno[:] = Int32.([r.episode for r in R])
            vbi[:]  = Int32.([r.birth_idx for r in R])
            vdi[:]  = Int32.([r.death_idx for r in R])
            vbt[:]  = [r.birth_time for r in R]
            vdt[:]  = [r.death_time for r in R]
            vns[:]  = Int32.([r.active_slices for r in R])
            vab[:]  = Int32.([r.area_birth for r in R])
            vLA[:]  = [r.lavd_candidate for r in R]
            vLB[:]  = [r.lavd_background for r in R]
            vEE[:]  = [r.E_LAVD for r in R]
            vFF[:]  = Int8.([r.pass ? 1 : 0 for r in R])

            vbt.attrib["units"] = "h"
            vdt.attrib["units"] = "h"
            vFF.attrib["description"] =
                "0 finite-lifetime FTCS episode is nonrotational; 1 rotational FTRCS"

            ds.attrib["finite_lifetime_lavd_ratio_min"] =
                episode_ftrcs.ratio_min

            ds.attrib["number_of_retained_episodes"] =
                Ne

            ds.attrib["number_of_lifespan_groups"] =
                episode_ftrcs.n_groups
        end

        #
        # Legacy global LAVD/FTRCS postprocessing.
        #

        if ftrcs !== nothing

            #
            # All IDL-SEBA candidates are written to the NetCDF
            # file, but FTRCS classification is performed only
            # for candidates passing the Rayleigh screening.
            #
            # FTRCS status:
            #
            #   -1   not evaluated
            #    0   evaluated, not FTRCS
            #    1   evaluated, FTRCS
            #

            Eout = fill(NaN,Nc)
            Fout = fill(Int8(-1),Nc)

            for ic in 1:Nc

                m = order[ic]

                jf = findfirst(==(m),ftrcs.objects)

                if jf === nothing
                    continue
                end

                Eout[ic] =
                    ftrcs.E_LAVD[jf]

                Fout[ic] =
                    ftrcs.pass[jf] ? Int8(1) : Int8(0)

            end

            vE = defVar(
                ds,
                "E_LAVD",
                Float64,
                ("candidate",)
            )

            vF = defVar(
                ds,
                "FTRCS",
                Int8,
                ("candidate",)
            )

            vE[:] = Eout
            vF[:] = Fout

            vF.attrib["description"] =
                "-1 not evaluated; 0 evaluated non-FTRCS; 1 evaluated FTRCS"

            ds.attrib["lavd_ratio_min"] =
                ftrcs.ratio_min
        end

        #
        # Global metadata.
        #

        ds.attrib["title"] =
            "Finite-time rotational coherent structures"

        ds.attrib["method"] =
            "Inflated Dynamic Laplacian with localized eigenbasis"

        ds.attrib["time_window_start_h"] =
            Float64(t0)

        ds.attrib["time_window_end_h"] =
            Float64(t1)

        ds.attrib["a_selected"] =
            crossover.a_selected

        ds.attrib["candidate_support_method"] =
            "SEBA subpartition of unity"

        ds.attrib["candidate_subpartition_threshold"] =
            C.tau_pu

        ds.attrib["candidate_seba_normalization"] =
            "positive part; each complete space-time SEBA vector normalized to global maximum 1"

        ds.attrib["candidate_connected_component_pruning"] =
            "none"

        ds.attrib["number_of_candidates"] =
            Nc

    finally

        close(ds)

    end

    println()
    println("Saved finite-lifetime FTRCS output:")
    println(filename)

    return filename
end

# ============================================================
# EXPORTS
# ============================================================

export
    VelocityField,
    FlowMap,
    CauchyGreenField,
    AbstractIDLOperator,
    IDLOperator,
    MaskedIDLOperator,
    build_interpolants,
    compute_flowmap,
    compute_cauchy_green,
    assemble_idl_operator,
    inflated_stiffness,
    IDLEigenResult,
    solve_idl_eigenproblem,
    sweep_idl_eigenproblems,
    IDLRayleighResult,
    IDLCrossoverResult,
    localize_eigenbasis,
    idl_rayleigh_diagnostics,
    select_idl_crossover,
    IDLCandidateResult,
    CandidateEpisode,
    CandidateEpisodeGroup,
    CandidateEpisodeGrouping,
    EpisodeFTRCSRecord,
    EpisodeFTRCSResult,
    largest_component,
    extract_idl_candidates,
    extract_candidate_episodes,
    group_candidate_episodes,
    LAVDField,
    FTRCSResult,
    compute_vorticity,
    compute_lavd,
    classify_ftrcs,
	 classify_ftrcs_episodes,
    diagnose_ftrcs_overlap,
	 material_survival_mask,
    save_ftrcs_netcdf

end
