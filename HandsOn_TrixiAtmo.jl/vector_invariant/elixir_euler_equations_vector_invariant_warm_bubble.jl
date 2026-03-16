using OrdinaryDiffEqLowStorageRK
using Plots

include("equations/compressible_euler_vectorinvariant_2d.jl")

# Initial condition
function initial_condition_warm_bubble(
    x,
    t,
    equations::CompressibleEulerVectorInvariantEquations2D,
)
    g = equations.g
    c_p = equations.c_p
    c_v = equations.c_v
    # center of perturbation
    center_x = 10000.0
    center_z = 2000.0
    # radius of perturbation
    radius = 2000.0
    # distance of current x to center of perturbation
    r = sqrt((x[1] - center_x)^2 + (x[2] - center_z)^2)

    # perturbation in potential temperature
    potential_temperature_ref = 300.0
    potential_temperature_perturbation = 0.0
    if r <= radius
        potential_temperature_perturbation = 2 * cospi(0.5 * r / radius)^2
    end
    potential_temperature = potential_temperature_ref + potential_temperature_perturbation

    # Exner pressure, solves hydrostatic equation for x[2]
    exner = 1 - g / (c_p * potential_temperature) * x[2]

    # pressure
    p_0 = 100_000.0  # reference pressure
    R = c_p - c_v    # gas constant (dry air)
    p = p_0 * exner^(c_p / R)

    # temperature
    T = potential_temperature * exner

    # density
    rho = p / (R * T)

    v1 = 20.0
    v2 = 0.0
    return SVector(rho, v1, v2, rho * potential_temperature, g * x[2])
end

###############################################################################
# semidiscretization of the compressible Euler equations
equations = CompressibleEulerVectorInvariantEquations2D(c_p = 1007, c_v = 717, gravity = 9.81)

boundary_conditions = (y_neg = boundary_condition_slip_wall, y_pos = boundary_condition_slip_wall)

polydeg = 3
basis = LobattoLegendreBasis(polydeg)

## Here are examples of different solutions, from the slowest version to the fastest.
## Check the file `solutions/numerical_fluxes_2d.jl` for their implementation after you gave it try implementing your own version
#
# include("solution/numerical_fluxes_2d.jl")
#
#surface_flux = (flux_conservative_surface, flux_nonconservative_surface)
#volume_flux = (flux_conservative_volume, flux_nonconservative_volume)

#surface_flux = flux_surface_combined
#volume_flux = flux_volume_combined

# In order to use the turbo version, please uncomment the next line:
 include("solver/noncons_kernel_2d.jl")
surface_flux = flux_surface_combined
volume_flux = flux_volume_combined_turbo

## TODO: Define surface and volume numerical fluxes for the vector invariant formulation.

volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (0.0, 0.0)
coordinates_max = (20_000.0, 10_000.0)

trees_per_dimension = (32, 16)

mesh = P4estMesh(
    trees_per_dimension,
    polydeg = polydeg,
    coordinates_min = coordinates_min,
    coordinates_max = coordinates_max,
    periodicity = (true, false),
    initial_refinement_level = 0,
)

semi = SemidiscretizationHyperbolic(
    mesh,
    equations,
    initial_condition_warm_bubble,
    solver,
    boundary_conditions = boundary_conditions,
)

###############################################################################
# ODE solvers, callbacks etc.
tspan = (0.0, 1000.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000

analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)

###############################################################################
# run the simulation
sol = solve(ode,
            RDPK3SpFSAL49(thread = Trixi.True());
            abstol = 1.0e-5, reltol = 1.0e-5, ode_default_options()...,
            callback = callbacks);
