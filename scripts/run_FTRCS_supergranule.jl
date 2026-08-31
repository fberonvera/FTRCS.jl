include(joinpath(@__DIR__, "..", "src", "FTRCS.jl"))

using .FTRCS
using NCDatasets
using Statistics
using LinearAlgebra
using Printf

# ============================================================
# FTRCS APPLICATION: SYNTHETIC SUPERGRANULE-LIKE FLOW
#
# Run from the repository root with, for example,
#
#     julia --project=. --threads=18 scripts/run_FTRCS_supergranule.jl
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
    "supergranule_synthetic.nc"
)

# ============================================================
# ANALYSIS SPACE/TIME WINDOW
#
# Synthetic supergranule-like application.
#
# User-facing analysis choices are specified in physical units.
# Integer grid strides used internally are derived below.
# ============================================================

analysis_t0_h = 0.0
analysis_hours = 6.0

# Flow-map / Cauchy-Green spatial resolution.

cg_space_km = 464.0

# IDL/FEM spatial and temporal resolution.

idl_space_km = 928.0
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

output_file =
    joinpath(
        output_dir,
        "FTRCS_supergranule_synthetic_output.nc"
    )


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

nmodes = 24

eig_tol     = 1e-5
eig_maxiter = 300
eig_sigma   = 1e-10

# ============================================================
# IDL-SEBA CANDIDATES
# ============================================================

candidate_quantile       = 0.90
candidate_min_area       = 20

# ============================================================
# FTRCS / LAVD CLASSIFICATION
# ============================================================

lavd_ratio_min = 1.0

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
    error("Synthetic supergranule velocity file not found: $data_file")

println()
println("Loading synthetic supergranule-like velocity")

x,y,t,u,v =
    NCDataset(data_file,"r") do ds

        x_all =
            Float64.(ds["x"][:])

        y_all =
            Float64.(ds["y"][:])

        t_all =
            Float64.(ds["t"][:])

        t1_requested =
            analysis_t0_h +
            analysis_hours

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
                "Supergranule NetCDF u must be stored as [Ny Nx Nt]"
            )

        size(v_yx) == size(u_yx) ||
            error(
                "Supergranule u and v dimensions do not agree"
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
println("Supergranule velocity field")
println("---------------------------")
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
t1 = analysis_t0_h + analysis_hours

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
    time_stride = idl_time_stride
)

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

candidates = extract_idl_candidates(
    idl,
    crossover;
    qlevel = candidate_quantile,
    min_area = candidate_min_area
)

println()
println("IDL-SEBA candidates")
println("-------------------")
println("selected a           = ",candidates.a)
println("localized objects    = ",length(candidates.order))

println()
println(" object        rho              threshold      active slices")

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
# LAVD
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
        "vorticity_supergranule.nc"
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

# ============================================================
# SAVE FTRCS OUTPUT
# ============================================================

save_ftrcs_netcdf(
    output_file,
    idl,
    eig_results,
    crossover,
    candidates;
    ftrcs = ftrcs,
    t0 = t0,
    t1 = t1
)

