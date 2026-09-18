include(joinpath(@__DIR__, "..", "src", "FTRCS.jl"))

using .FTRCS
using NCDatasets
using Statistics
using LinearAlgebra
using Printf

# ============================================================
# FTRCS APPLICATION: HINODE PHOTOSPHERIC FLOW
#
# Run from the repository root with, for example,
#
#     julia --project=. --threads=auto scripts/run_FTRCS_hinode.jl
#
# All application-specific choices are made in this file.
# FTRCS.jl contains the reusable numerical implementation.
#
# External NetCDF velocity arrays are stored as
#
#     u(y,x,t), v(y,x,t)  ->  [Ny Nx Nt].
#
# Before constructing VelocityField, this script converts them
# once to the internal FTRCS convention
#
#     u(x,y,t), v(x,y,t)  ->  [Nx Ny Nt].
#
# Spatial coordinates are in km, time is in h, and internal
# velocities are in km/h.
# ============================================================

# ============================================================
# INPUT DATA
# ============================================================

data_file = joinpath(
    @__DIR__,
    "..",
    "data",
    "hinode_synthetic.nc"
)

# ============================================================
# ANALYSIS SPACE/TIME WINDOW
#
# Hinode photospheric application.
#
# User-facing analysis choices are specified in physical units.
# Integer grid strides used internally are derived below.
# ============================================================

analysis_t0_h = 0.0
analysis_hours = 6.0

# Spatial analysis rectangle [km].
#
# This specifies where the initial conditions for the flow map,
# Cauchy--Green tensor, and IDL calculation are placed.
#
# To use the complete Hinode velocity domain:
#
#     analysis_bounds = nothing
#
# To restrict the calculation to a rectangular subdomain, use:
#
#     analysis_bounds = (
#         (xmin,xmax),   # x bounds [km]
#         (ymin,ymax)    # y bounds [km]
#     )
#
# For example:
#
#     analysis_bounds = (
#         (10_000.0,40_000.0),
#         (15_000.0,35_000.0)
#     )
#
# restricts the initial-condition grid to
#
#     10,000 <= x <= 40,000 km,
#     15,000 <= y <= 35,000 km.
#
# The velocity field itself is NOT restricted to this rectangle.
# Trajectories may leave the analysis rectangle and continue to
# be integrated as long as they remain inside the complete
# velocity-data domain.
# ============================================================

analysis_bounds = nothing

# Flow-map / Cauchy-Green spatial resolution.

cg_space_km = 464.0

# IDL/FEM spatial and temporal resolution.

idl_space_km = 464.0
idl_time_h = 0.25

# ============================================================
# OUTPUT
# ============================================================

output_dir = joinpath(
    @__DIR__,
    "..",
    "runs"
)

mkpath(output_dir)

output_dir = joinpath(
    @__DIR__,
    "..",
    "runs"
)

mkpath(output_dir)


# ============================================================
# PARALLEL COMPUTING
# ============================================================

thread_fraction = 0.75

# Flow-map and other thread-safe parallel calculations.

nchunks = max(
    1,
    floor(
        Int,
        thread_fraction * Threads.nthreads()
    )
)

# Number of trajectories integrated in each vectorized
# flow-map batch.

flow_batch_size = 5_000

# ARPACK eigensolves are run sequentially.

eig_workers = 1


# ============================================================
# NUMERICAL PARAMETERS
# ============================================================

# Trajectory integrator.
#
# Use :DP5 for adaptive Dormand-Prince integration or :RK4
# for fixed-step classical Runge-Kutta integration.
#
# Time is in hours. rk4_dt_h is used only when :RK4 is selected.

flow_integrator = :DP5
rk4_dt_h = 0.01

reltol = 1e-6
abstol = 1e-8

# ============================================================
# IDL EIGENVALUE SWEEP
# ============================================================

a_values = [
    0.005,
    0.01,
    0.02,
    0.03,
    0.05,
    0.10
]

nmodes = 32

eig_tol     = 1e-5
eig_maxiter = 300
eig_sigma   = 1e-10

# ============================================================
# IDL-SEBA CANDIDATES
#
# Candidate supports are defined by Gary Froyland's
# subpartition-of-unity postprocessing.  There are no plateau
# parameters and no physical area thresholds.
# ============================================================

# ============================================================
# LOAD VELOCITY
#
# NetCDF variables:
#
#   x, y      km
#   t         h
#   u, v      km/h
#
# External file layout is [Ny Nx Nt] = [y x time].
# Convert once here to the internal FTRCS layout [Nx Ny Nt].
# ============================================================

isfile(data_file) ||
    error("Hinode velocity file not found: $data_file")

println()
println("Loading Hinode photospheric velocity")

x,y,t,u,v =
    NCDataset(data_file,"r") do ds

        x_all =
            Float64.(ds["x"][:])

        y_all =
            Float64.(ds["y"][:])

        t_all =
            Float64.(ds["t"][:])

        t1_requested =
           isfinite(analysis_hours) ?
           analysis_t0_h + analysis_hours :
           t_all[end]

        t1_requested <= t_all[end] ||
           error(
           "Requested analysis window extends beyond velocity record: " *
           "requested t1=$(t1_requested) h, available t1=$(t_all[end]) h"
           )

        it =
            findall(
                (t_all .>= analysis_t0_h) .&
                (t_all .<= t1_requested)
            )

        isempty(it) &&
            error(
                "Requested analysis window is outside the velocity record"
            )

        u_yx =
            Float64.(ds["u"][:,:,it])

        v_yx =
            Float64.(ds["v"][:,:,it])

        size(u_yx) ==
            (
                length(y_all),
                length(x_all),
                length(it)
            ) ||
            error(
                "Hinode NetCDF u must be stored as [Ny Nx Nt]"
            )

        size(v_yx) == size(u_yx) ||
            error(
                "Hinode u and v dimensions do not agree"
            )

        #
        # External [Ny Nx Nt] -> internal [Nx Ny Nt].
        #

        u_xy =
            permutedims(
                u_yx,
                (2,1,3)
            )

        v_xy =
            permutedims(
                v_yx,
                (2,1,3)
            )

        return (
            x_all,
            y_all,
            t_all[it],
            u_xy,
            v_xy
        )

    end

velocity_dx_km =
    mean(diff(x))

velocity_dy_km =
    mean(diff(y))

velocity_dt_h =
    mean(diff(t))

vel =
    VelocityField(
        x,
        y,
        t,
        u,
        v,
        velocity_dx_km,
        velocity_dy_km,
        velocity_dt_h
    )

println()
println("Hinode photospheric velocity field")
println("----------------------------------")
println("size(u) = ",size(vel.u)," = [Nx Ny Nt]")
println("size(v) = ",size(vel.v)," = [Nx Ny Nt]")
println(
    "x = ",
    first(vel.x),
    " ... ",
    last(vel.x),
    " km"
)
println(
    "y = ",
    first(vel.y),
    " ... ",
    last(vel.y),
    " km"
)
println(
    "t = ",
    first(vel.t),
    " ... ",
    last(vel.t),
    " h"
)
println(
    "dx = ",
    velocity_dx_km,
    " km"
)
println(
    "dy = ",
    velocity_dy_km,
    " km"
)
println(
    "dt = ",
    velocity_dt_h,
    " h"
)


# ============================================================
# DERIVED ANALYSIS GRID PARAMETERS
# ============================================================

seed_dx = cg_space_km
seed_dy = cg_space_km

t0 = analysis_t0_h

t1 =
    isfinite(analysis_hours) ?
    analysis_t0_h + analysis_hours :
    vel.t[end]

idl_space_stride =
    max(
        1,
        round(
            Int,
            idl_space_km /
            mean((seed_dx,seed_dy))
        )
    )

idl_time_stride =
    max(
        1,
        round(
            Int,
            idl_time_h /
            velocity_dt_h
        )
    )

println()
println("Analysis resolution")
println("-------------------")
println(
    "analysis window             = ",
    t0,
    " -- ",
    t1,
    " h"
)
println(
    "velocity spacing            = ",
    velocity_dx_km,
    " x ",
    velocity_dy_km,
    " km"
)
println(
    "velocity cadence            = ",
    velocity_dt_h,
    " h"
)
println(
    "requested CG spacing        = ",
    cg_space_km,
    " km"
)
println(
    "actual CG spacing           = ",
    seed_dx,
    " x ",
    seed_dy,
    " km"
)
println(
    "requested IDL spacing       = ",
    idl_space_km,
    " km"
)
println(
    "actual IDL spacing          = ",
    idl_space_stride * seed_dx,
    " x ",
    idl_space_stride * seed_dy,
    " km"
)
println(
    "requested IDL time spacing  = ",
    idl_time_h,
    " h"
)
println(
    "actual IDL time spacing     = ",
    idl_time_stride * velocity_dt_h,
    " h"
)

# ============================================================
# FLOW MAP
# ============================================================

flow = compute_flowmap(
    vel;
    seed_dx = seed_dx,
    seed_dy = seed_dy,
    window = (t0,t1),
    seed_bounds = analysis_bounds,
    integrator = flow_integrator,
    rk4_dt = rk4_dt_h,
    reltol = reltol,
    abstol = abstol,
    batch_size = flow_batch_size,
    nchunks = nchunks
)

println()
println("Flow-map size:")
println("PhiX = ", size(flow.PhiX))
println("PhiY = ", size(flow.PhiY))

# ============================================================
# FLOW-MAP IDENTITY CHECK AT t0
# ============================================================

X0_expected = repeat(flow.xseed, 1, length(flow.yseed))
Y0_expected = repeat(flow.yseed', length(flow.xseed), 1)

errX = flow.PhiX[:,:,1] .- X0_expected
errY = flow.PhiY[:,:,1] .- Y0_expected

println()
println("Flow-map identity check at t0")
println("-----------------------------")
println("max |PhiX(t0)-x| = ", maximum(abs.(errX)), " km")
println("max |PhiY(t0)-y| = ", maximum(abs.(errY)), " km")
println("median |PhiX(t0)-x| = ", median(abs.(errX)), " km")
println("median |PhiY(t0)-y| = ", median(abs.(errY)), " km")

println()
println("First 5 xseed = ", flow.xseed[1:5])
println("First 5 yseed = ", flow.yseed[1:5])

println()
println("PhiX[1:5,1,1] = ", flow.PhiX[1:5,1,1])
println("PhiY[1:5,1,1] = ", flow.PhiY[1:5,1,1])

println()
println("PhiX[1,1:5,1] = ", flow.PhiX[1,1:5,1])
println("PhiY[1,1:5,1] = ", flow.PhiY[1,1:5,1])

# ============================================================
# STEP 5A: MATERIAL SURVIVAL DOMAIN
#
# For the selected analysis interval, retain only initial CG
# nodes whose complete trajectories remain inside the observed
# Hinode velocity domain.  This mask defines the spatial domain
# used by the masked IDL/FEM assembly below.
# ============================================================

survival_mask,exit_time =
    material_survival_mask(
        flow,
        vel;
        verbose = true
    )

# ============================================================
# CAUCHY-GREEN
# ============================================================

cg = compute_cauchy_green(flow)

valid0 = isfinite.(cg.detDF[:,:,1])

println()
println("Initial Cauchy-Green diagnostics")
println("--------------------------------")

println(
    "median detDF(t0) = ",
    median(cg.detDF[:,:,1][valid0])
)

println(
    "min/max detDF(t0) = ",
    extrema(cg.detDF[:,:,1][valid0])
)

println(
    "median lambda_max(t0) = ",
    median(cg.lam_max[:,:,1][valid0])
)

k = length(cg.times)

valid = isfinite.(cg.detDF[:,:,k])

println()
println("Final Cauchy-Green diagnostics")
println("------------------------------")

println(
    "valid final C points = ",
    count(valid),
    " / ",
    length(valid)
)

println(
    "median detDF = ",
    median(cg.detDF[:,:,k][valid])
)

println(
    "min/max detDF = ",
    extrema(cg.detDF[:,:,k][valid])
)

println(
    "median lambda_max = ",
    median(cg.lam_max[:,:,k][valid])
)

# ============================================================
# INFLATED DYNAMIC LAPLACIAN
# ============================================================

idl = assemble_idl_operator(
    cg,
    vel;
    space_stride = idl_space_stride,
    time_stride = idl_time_stride,
    wet_mask = survival_mask
)

# ============================================================
# IDL SANITY CHECKS
# ============================================================

println()
println("IDL sanity checks")
println("-----------------")

one_st = ones(idl.Nst)

println(
    "||Kspace*1|| / ||1|| = ",
    norm(idl.Kspace*one_st)/norm(one_st)
)

println(
    "||Kmaterial*1|| / ||1|| = ",
    norm(idl.Kmaterial*one_st)/norm(one_st)
)

for r in 1:3

    z = randn(idl.Nst)

    println(
        "z'Kspace z    = ",
        dot(z,idl.Kspace*z)
    )

    println(
        "z'Kmaterial z = ",
        dot(z,idl.Kmaterial*z)
    )

end

# ============================================================
# IDL EIGENVALUE SWEEP
# ============================================================

eig_results = sweep_idl_eigenproblems(
    idl,
    a_values;
    nmodes = nmodes,
    nworkers = eig_workers,
    tol = eig_tol,
    maxiter = eig_maxiter
)

println()
println("IDL leading eigenvalues")
println("-----------------------")

for R in eig_results

    println()
    println(
        "a = ",
        R.a,
        "   time = ",
        round(R.elapsed,digits=2),
        " s"
    )

    println(
        "mu = ",
        R.mu
    )

end

# ============================================================
# IDL SPATIAL-MATERIAL CROSSOVER
# ============================================================

crossover = select_idl_crossover(
    idl,
    eig_results
)

println()
println("IDL spatial-material crossover")
println("------------------------------")
println(
    "       a        rho_space        a^2 rho_material      rho_total        ratio"
)

for D in crossover.diagnostics

    ratio =
        D.rho_space_median /
        D.rho_a2material_median

    @printf(
        "%9.4g   %14.6e   %14.6e   %14.6e   %10.4f\n",
        D.a,
        D.rho_space_median,
        D.rho_a2material_median,
        D.rho_total_median,
        ratio
    )
end

println()
println(
    "Selected crossover a = ",
    crossover.a_selected
)

# ============================================================
# EXTRACT IDL-SEBA CANDIDATES
# ============================================================

# ============================================================
# FTRCS / LAVD CLASSIFICATION
# ============================================================

lavd_ratio_min = 1.0




candidates = extract_idl_candidates(
    idl,
    crossover
)

println()
println("IDL-SEBA candidates")
println("-------------------")
println("selected a           = ",candidates.a)
println("localized objects    = ",length(candidates.order))

println()
println(" object        rho              tau_pu         active slices")

for m in candidates.order

    active =
        count(
            k -> any(@view candidates.masks[:,:,k,m]),
            1:idl.Nt
        )

    @printf(
        "%6d   %14.6e   %14.6e   %6d / %d\n",
        m,
        candidates.rho[m],
        candidates.thresholds[m],
        active,
        idl.Nt
    )
end


# ============================================================
# FINITE-LIFETIME IDL-SEBA EPISODES
#
# Step 1 of the lifespan-aware FTRCS revision.
#
# A candidate is active whenever its thresholded SEBA mask is
# nonempty.  If it disappears and later reappears, each contiguous
# active period is returned as a separate CandidateEpisode.
#
# No LAVD or FTRCS logic is changed yet.
# ============================================================

episodes =
    extract_candidate_episodes(
        candidates
    )

println()
println("IDL-SEBA episode times")
println("----------------------")
println(
    " rank   SEBA object   episode       birth [h]       death [h]    duration [h]"
)

for ep in episodes

    tb = idl.t[ep.birth_idx]
    td = idl.t[ep.death_idx]

    @printf(
        "%5d   %11d   %7d   %13.6f   %13.6f   %12.6f\n",
        ep.rank,
        ep.seba_index,
        ep.episode,
        tb,
        td,
        td-tb
    )

end

println()
println(
    "Total finite-lifetime episodes = ",
    length(episodes)
)

println(
    "Candidates with more than one episode = ",
    count(
        m -> count(ep -> ep.seba_index == m, episodes) > 1,
        candidates.order
    )
)

# ============================================================
# FILTER AND GROUP FINITE-LIFETIME EPISODES
#
# Raw episode extraction records every contiguous active interval,
# including isolated one-slice threshold crossings.
#
# For the lifespan-aware FTRCS development, require at least two
# consecutive active IDL slices before promoting an interval to a
# finite-lifetime coherent episode.
#
# Retained episodes are then grouped by their exact
#
#     (birth_idx, death_idx)
#
# pair. Later, each unique lifespan group can reuse one
# flow-map/LAVD calculation.
# ============================================================

min_episode_slices = 2

episode_grouping =
    group_candidate_episodes(
        episodes;
        min_episode_slices = min_episode_slices,
        verbose = true
    )

# ============================================================
# STEP 4: FINITE-LIFETIME LAVD CLASSIFICATION
#
# Each retained FTCS episode is evaluated over its own lifespan.
# Episodes sharing exactly the same birth/death indices reuse one
# lifespan-specific flow-map/LAVD calculation.
# ============================================================

episode_ftrcs =
    classify_ftrcs_episodes(
        vel,
        idl,
        candidates,
        episode_grouping,
        flow;
        lavd_ratio_min = lavd_ratio_min,
        verbose = true
    )

# ============================================================
# FTRCS OVERLAP / REDUNDANCY DIAGNOSTICS
# ============================================================

diagnose_ftrcs_overlap(
    candidates,
    episode_ftrcs
)	 

# ============================================================
# LEGACY GLOBAL LAVD / CLASSIFICATION
#
# Disabled in Step 4. The old 0--12 h classifier is inconsistent
# with finite-lifetime episode classification and is retained below
# only as commented reference code.
# ============================================================

#=
# ============================================================
# LAVD
#
# NOTE:
# The calculation below is still the LEGACY/global LAVD over the
# complete 0--12 h interval. It is intentionally retained for
# regression comparison while the lifespan-aware classifier is
# developed.
# ============================================================

lavd = compute_lavd(
    vel,
    flow
)

# ============================================================
# VORTICITY DIAGNOSTIC
# ============================================================

omega,omega_mean =
    compute_vorticity(vel)


# ============================================================
# SAVE VORTICITY
# ============================================================

vorticity_file =
    joinpath(
        output_dir,
        "vorticity_hinode_00h_12h.nc"
    )

NCDataset(
    vorticity_file,
    "c"
) do ds

    defDim(ds,"x",length(vel.x))
    defDim(ds,"y",length(vel.y))
    defDim(ds,"time",length(vel.t))

    defVar(
        ds,
        "x",
        Float64,
        ("x",)
    )[:] = vel.x

    defVar(
        ds,
        "y",
        Float64,
        ("y",)
    )[:] = vel.y

    defVar(
        ds,
        "t",
        Float64,
        ("time",)
    )[:] = vel.t

    #
    # FTRCS internal convention:
    #
    #     omega = [Nx Ny Nt]
    #
    # External NetCDF convention:
    #
    #     omega = [Ny Nx Nt]
    #

    omega_yxt =
        permutedims(
            omega,
            (2,1,3)
        )

    defVar(
        ds,
        "omega",
        Float64,
        ("y","x","time")
    )[:,:,:] = omega_yxt

    defVar(
        ds,
        "omega_mean",
        Float64,
        ("time",)
    )[:] = omega_mean

    ds["x"].attrib["units"] = "km"
    ds["y"].attrib["units"] = "km"
    ds["t"].attrib["units"] = "h"

    ds["omega"].attrib["units"] =
        "h^-1"

    ds["omega_mean"].attrib["units"] =
        "h^-1"

end

println()
println(
    "Saved vorticity field: ",
    vorticity_file
)

# ============================================================
# FTRCS CLASSIFICATION
# ============================================================

ftrcs = classify_ftrcs(
    idl,
    candidates,
    lavd;
    lavd_ratio_min = lavd_ratio_min
)

println()
println("FTRCS classification")
println("--------------------")
println(
    " object          rho       area0      E_LAVD    FTRCS"
)

for io in eachindex(ftrcs.objects)

    @printf(
        "%6d   %12.5e   %6d     %8.3f     %s\n",
        ftrcs.objects[io],
        ftrcs.rho[io],
        ftrcs.area0[io],
        ftrcs.E_LAVD[io],
        ftrcs.pass[io] ? "yes" : "no"
    )

end

println()
println(
    "FTRCS selected = ",
    count(ftrcs.pass),
    " / ",
    length(ftrcs.pass)
)

=#

# ============================================================
# SAVE FTRCS OUTPUT
# ============================================================

t0_tag = @sprintf("%02.0f",t0)

t1_tag =
    isapprox(t1,round(t1); atol=1e-10) ?
    @sprintf("%02.0f",t1) :
    replace(@sprintf("%.2f",t1),"."=>"p")

outfile =
    joinpath(
        output_dir,
        "FTRCS_hinode_synthetic_$(t0_tag)h_$(t1_tag)h_output.nc"
    )

save_ftrcs_netcdf(
    outfile,
    idl,
    eig_results,
    crossover,
    candidates;
    episode_ftrcs = episode_ftrcs,
    t0 = t0,
    t1 = t1
)

println()
println("Saved finite-lifetime FTRCS output:")
println(outfile)

println()
println("--------------------------------")
println("Hinode FTRCS run complete")
println("--------------------------------")
println("analysis window = ",t0," -- ",t1," h")
println("output file = ",outfile)
