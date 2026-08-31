include(joinpath(@__DIR__, "..", "src", "FTRCS.jl"))

using .FTRCS
using NCDatasets
using Statistics
using LinearAlgebra
using Printf
using Dates

# ============================================================
# FTRCS APPLICATION: GULF OF MEXICO ALTIMETRY
#
# Run from the repository root with, for example,
#
#     julia --project=. --threads=18 scripts/run_FTRCS_gom.jl
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
#
# The distributed AVISO subset contains longitude, latitude,
# time, and horizontal velocity components. The run converts
# the selected regional data to the common internal FTRCS units
# and Cartesian coordinates described above.
# ============================================================

data_file = joinpath(
    @__DIR__,
    "..",
    "data",
    "aviso_20130518_20140421.nc"
)

# ============================================================
# ANALYSIS SPACE/TIME WINDOW
#
# Gulf of Mexico application.
#
# User-facing analysis choices are specified here in physical
# units. Integer grid strides used internally are derived below.
# ============================================================

lon_bounds = (-98.0, -80.0)
lat_bounds = ( 18.0,  31.0)

analysis_t0 =
    DateTime(2013,6,1)

analysis_days = 30.0

# Flow-map / Cauchy-Green spatial resolution.

cg_space_km = 1.0

# IDL/FEM spatial and temporal resolution.

idl_space_km = 4.0
idl_time_days = 15.0

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
        "FTRCS_gom_output.nc"
    )

# ============================================================
# PARALLEL COMPUTING
# ============================================================

thread_fraction = 0.75

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
# LOAD ALTIMETRY
#
# NetCDF variables:
#
#   lon       degrees east
#   lat       degrees north
#   time      decoded by NCDatasets as DateTime
#   u, v      m/s
#
# The selected regional lon-lat grid is mapped to a local
# Cartesian grid using an equirectangular projection.
#
# Internally FTRCS uses:
#
#   x,y       km
#   t         h
#   u,v       km/h
# ============================================================

isfile(data_file) ||
    error("Altimetry file not found: $data_file")

println()
println("Loading Gulf of Mexico altimetry")

lon,lat,t,u,v,wet =
    NCDataset(data_file,"r") do ds

        lon_all =
            Float64.(ds["lon"][:])

        lat_all =
            Float64.(ds["lat"][:])

        time_datetime =
            ds["time"][:]

        #
        # Convert longitude to [-180,180) if needed.
        #

        lon_all =
            mod.(lon_all .+ 180.0,360.0) .- 180.0

        ix = findall(
            (lon_all .>= lon_bounds[1]) .&
            (lon_all .<= lon_bounds[2])
        )

        iy = findall(
            (lat_all .>= lat_bounds[1]) .&
            (lat_all .<= lat_bounds[2])
        )

        isempty(ix) &&
            error("No longitude points in requested GoM box")

        isempty(iy) &&
            error("No latitude points in requested GoM box")

        #
        # Select the requested finite-time analysis window.
        #

        analysis_t1 =
            analysis_t0 +
            Millisecond(
                round(
                    Int,
                    analysis_days * 24.0 * 3600.0 * 1000.0
                )
            )

        it = findall(
            (time_datetime .>= analysis_t0) .&
            (time_datetime .<= analysis_t1)
        )

        isempty(it) &&
            error("Requested analysis window is outside the data record")

        lon =
            lon_all[ix]

        lat =
            lat_all[iy]

        #
        # Internal time in hours from analysis_t0.
        #

        t =
            Dates.value.(
                time_datetime[it] .- analysis_t0
            ) ./ (1000.0 * 3600.0)

        #
        # NCDatasets may represent NetCDF missing values as
        # `missing`; convert them to NaN explicitly.
        #

        u =
            Float64.(
                coalesce.(
                    ds["u"][ix,iy,it],
                    NaN
                )
            )

        v =
            Float64.(
                coalesce.(
                    ds["v"][ix,iy,it],
                    NaN
                )
            )

        size(u) ==
            (length(lon),length(lat),length(t)) ||
            error(
                "AVISO u dimensions are not (lon,lat,time)"
            )

        size(v) == size(u) ||
            error(
                "AVISO u and v dimensions do not agree"
            )

        #
        # Preserve the original ocean/land mask before replacing
        # missing velocities by zero. A grid point is regarded as
        # ocean only if both velocity components are finite over
        # the complete analysis interval.
        #

        wet =
            dropdims(
                all(
                    isfinite.(u) .&
                    isfinite.(v);
                    dims = 3
                );
                dims = 3
            )

        return lon,lat,t,u,v,wet

    end

# ============================================================
# LAND MASK
#
# Retain the full rectangular Gulf of Mexico domain. Missing
# AVISO velocities over land are set to zero only so that the
# rectangular velocity interpolant remains defined. The original
# ocean/land mask is retained separately for the masked IDL/FEM
# assembly.
# ============================================================

u[.!isfinite.(u)] .= 0.0
v[.!isfinite.(v)] .= 0.0

# ============================================================
# LOCAL CARTESIAN GRID
# ============================================================

Re = 6371.0                 # km

lat0 = mean(lat)

x =
    Re * cosd(lat0) *
    deg2rad.(lon .- lon[1])

y =
    Re *
    deg2rad.(lat .- lat[1])

dx = mean(diff(x))
dy = mean(diff(y))
dt = mean(diff(t))

# m/s -> km/h

u .*= 3.6
v .*= 3.6

vel = VelocityField(
    collect(x),
    collect(y),
    collect(t),
    u,
    v,
    Float64(dx),
    Float64(dy),
    Float64(dt)
)

# ============================================================
# AXIS ADAPTATION FOR THE EXISTING FTRCS CORE
#
# The current FTRCS.jl reproduces the spatial-derivative
# convention used by the validated supergranule calculation.
# AVISO is stored naturally as (lon,lat,time) = (x,y,t).
#
# For derivative-based diagnostics only, swap the spatial axes
# in the run script, call the unchanged core routines, and map
# the resulting scalar/tensor fields back to the physical
# (x,y,t) orientation.
# ============================================================

function swap_xy_velocity(V::VelocityField)

    VelocityField(
        copy(V.y),
        copy(V.x),
        copy(V.t),
        permutedims(V.v,(2,1,3)),
        permutedims(V.u,(2,1,3)),
        V.dy,
        V.dx,
        V.dt
    )

end


function swap_xy_flow(F::FlowMap)

    FlowMap(
        permutedims(F.PhiY,(2,1,3)),
        permutedims(F.PhiX,(2,1,3)),
        copy(F.yseed),
        copy(F.xseed),
        copy(F.times),
        F.seed_dy,
        F.seed_dx
    )

end


function restore_xy_cauchy_green(
    C::CauchyGreenField,
    F::FlowMap
)

    CauchyGreenField(
        permutedims(C.C22,(2,1,3)),
        permutedims(C.C12,(2,1,3)),
        permutedims(C.C11,(2,1,3)),
        permutedims(C.detC,(2,1,3)),
        permutedims(C.detDF,(2,1,3)),
        permutedims(C.lam_min,(2,1,3)),
        permutedims(C.lam_max,(2,1,3)),
        copy(F.xseed),
        copy(F.yseed),
        copy(C.times)
    )

end


function restore_xy_lavd(
    L::LAVDField,
    F::FlowMap
)

    LAVDField(
        permutedims(L.lavd,(2,1,3)),
        copy(L.omega_mean),
        copy(F.xseed),
        copy(F.yseed),
        copy(L.times)
    )

end

# ============================================================
# FLOW MAP / CAUCHY-GREEN GRID
# ============================================================

# Interpolate the native altimetry velocity field onto the
# requested trajectory grid. The Cauchy-Green tensor is computed
# from this flow map.

seed_dx = cg_space_km
seed_dy = cg_space_km

t0 = first(t)
t1 = last(t)

# Trajectory integrator.
#
# Use :DP5 for adaptive Dormand-Prince integration or :RK4
# for fixed-step classical Runge-Kutta integration.
#
# Time is in hours. For altimetry, 2.4 h = 0.1 day.

flow_integrator = :RK4
rk4_dt_h = 2.4

reltol = 1e-6
abstol = 1e-8


# ============================================================
# IDL / FEM GRID
# ============================================================

# Convert the requested physical IDL/FEM resolution to integer
# subsampling strides on the Cauchy-Green space/time grid.

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
            24.0 * idl_time_days / dt
        )
    )

println()
println("Analysis resolution")
println("-------------------")
println(
    "native velocity spacing    = ",
    dx,
    " x ",
    dy,
    " km"
)
println(
    "requested CG spacing       = ",
    cg_space_km,
    " km"
)
println(
    "actual CG spacing          = ",
    seed_dx,
    " x ",
    seed_dy,
    " km"
)
println(
    "requested IDL spacing      = ",
    idl_space_km,
    " km"
)
println(
    "actual IDL spacing         = ",
    idl_space_stride * seed_dx,
    " x ",
    idl_space_stride * seed_dy,
    " km"
)
println(
    "requested IDL time spacing = ",
    idl_time_days,
    " days"
)
println(
    "actual IDL time spacing    = ",
    idl_time_stride * dt / 24.0,
    " days"
)


# ============================================================
# IDL EIGENVALUE SWEEP
#
# a has the units required to balance the spatial and material
# Rayleigh contributions.  Start broad; the crossover is
# selected automatically.
# ============================================================

a_values = [
    0.05,
    0.10,
    0.20,
    0.50,
    1.00,
    2.00
]

nmodes = 24

eig_tol     = 1e-5
eig_maxiter = 300
eig_sigma   = 1e-10

# ============================================================
# IDL-SEBA CANDIDATES
# ============================================================

candidate_quantile = 0.90
candidate_min_area = 20

# ============================================================
# FTRCS / LAVD CLASSIFICATION
# ============================================================

lavd_ratio_min = 1.0

# ============================================================
# FLOW MAP
# ============================================================

flow = compute_flowmap(
    vel;
    seed_dx = cg_space_km,
    seed_dy = cg_space_km,
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
println("PhiX = ",size(flow.PhiX))
println("PhiY = ",size(flow.PhiY))

# ============================================================
# FLOW-MAP IDENTITY CHECK AT t0
# ============================================================

X0_expected =
    repeat(
        flow.xseed,
        1,
        length(flow.yseed)
    )

Y0_expected =
    repeat(
        flow.yseed',
        length(flow.xseed),
        1
    )

errX =
    flow.PhiX[:,:,1] .- X0_expected

errY =
    flow.PhiY[:,:,1] .- Y0_expected

println()
println("Flow-map identity check at t0")
println("-----------------------------")
println(
    "max |PhiX(t0)-x| = ",
    maximum(abs.(errX)),
    " km"
)
println(
    "max |PhiY(t0)-y| = ",
    maximum(abs.(errY)),
    " km"
)

# ============================================================
# CAUCHY-GREEN
# ============================================================

flow_xy = swap_xy_flow(flow)

cg_xy = compute_cauchy_green(
    flow_xy
)

cg = restore_xy_cauchy_green(
    cg_xy,
    flow
)

valid0 =
    isfinite.(cg.detDF[:,:,1])

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

valid =
    isfinite.(cg.detDF[:,:,k])

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
    wet_mask = wet
)

println()
println("IDL sanity checks")
println("-----------------")

one_st =
    ones(idl.Nst)

println(
    "||Kspace*1|| / ||1|| = ",
    norm(idl.Kspace*one_st) /
    norm(one_st)
)

println(
    "||Kmaterial*1|| / ||1|| = ",
    norm(idl.Kmaterial*one_st) /
    norm(one_st)
)

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
println(
    "selected a        = ",
    candidates.a
)
println(
    "localized objects = ",
    length(candidates.order)
)

println()
println(
    " object        rho              threshold      active slices"
)

for m in candidates.order

    active =
        count(
            k -> any(
                @view candidates.masks[:,:,k,m]
            ),
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

vel_xy = swap_xy_velocity(vel)

lavd_xy = compute_lavd(
    vel_xy,
    flow_xy
)

lavd = restore_xy_lavd(
    lavd_xy,
    flow
)

# ============================================================
# SAVE FULL LAVD FIELD
# ============================================================

lavd_file =
    joinpath(
        output_dir,
        "LAVD_gom.nc"
    )

ds =
    NCDataset(
        lavd_file,
        "c"
    )

defDim(
    ds,
    "x",
    length(lavd.x)
)

defDim(
    ds,
    "y",
    length(lavd.y)
)

defVar(
    ds,
    "x",
    Float64,
    ("x",)
)[:] = lavd.x

defVar(
    ds,
    "y",
    Float64,
    ("y",)
)[:] = lavd.y

defVar(
    ds,
    "LAVD",
    Float64,
    ("x","y")
)[:,:] =
    lavd.lavd[:,:,end]

close(ds)

println()
println(
    "Saved LAVD field: ",
    lavd_file
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