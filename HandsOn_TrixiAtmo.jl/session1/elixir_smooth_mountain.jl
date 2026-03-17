###############################################################################
# DGSEM for the linear advection equation on a prismed icosahedral grid
###############################################################################

using OrdinaryDiffEq, Trixi, TrixiAtmo, ForwardDiff, LinearAlgebra

###############################################################################
# Auxiliary variables for covariant equations with bottom topography in source terms

# Add up the total number of auxiliary variables for equations in covariant form
@inline function TrixiAtmo.n_aux_node_vars(
    ::TrixiAtmo.AbstractCovariantEquations{NDIMS,NDIMS_AMBIENT},
) where {NDIMS,NDIMS_AMBIENT}
    nvars_basis_covariant = NDIMS_AMBIENT * NDIMS
    nvars_basis_contravariant = NDIMS * NDIMS_AMBIENT
    nvars_area_element = 1
    nvars_metric_covariant = NDIMS * (NDIMS + 1) ÷ 2
    nvars_metric_contravariant = NDIMS * (NDIMS + 1) ÷ 2
    nvars_bottom_topography = 1
    nvars_christoffel = NDIMS * NDIMS * (NDIMS + 1) ÷ 2
    nvars_bottom_topography_derivatives = NDIMS

    return nvars_basis_covariant +
           nvars_basis_contravariant +
           nvars_area_element +
           nvars_metric_covariant +
           nvars_metric_contravariant +
           nvars_bottom_topography +
           nvars_christoffel +
           nvars_bottom_topography_derivatives
end

function TrixiAtmo.init_auxiliary_node_variables!(
    aux_values,
    mesh::DGMultiMesh,
    equations::TrixiAtmo.AbstractCovariantEquations{2,3},
    dg::DGMulti{<:Any,<:Tri},
    bottom_topography,
)
    rd = dg.basis
    (; xyz) = mesh.md
    md = mesh.md
    n_aux = TrixiAtmo.n_aux_node_vars(equations)

    # Identify the vertices corresponding to the corners of the reference element. rd.V1 is not useful,
    # since the physical nodes are projected onto the sphere and thus the corner nodes would be slightly off.
    # Instead, we identify the corner nodes by their reference coordinates and build a mask to access them directly
    VMask = []
    for corner in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)]
        for j = 1:size(rd.rst[1], 1)
            r, s = rd.rst[1][j], rd.rst[2][j]
            if all(isapprox.((r, s), corner))
                push!(VMask, j)
            end
        end
    end

    # Compute the radius of the sphere from the first element's fourth vertex, such that we can use it
    # throughout the computation. We assume that each Wedge element's last three corner vertices lie
    # on the simulated sphere.
    VX, VY, VZ = map(coords -> coords[VMask, 1], xyz)
    v_outer = getindex.([VX, VY, VZ], 1)
    radius = norm(v_outer)

    for element in TrixiAtmo.eachelement(mesh, dg)
        # Compute corner vertices of the element
        VX, VY, VZ = map(coords -> coords[VMask, element], xyz)
        v1, v2, v3 = map(i -> getindex.([VX, VY, VZ], i), 1:3)

        aux_node = Vector{eltype(aux_values[1, 1])}(undef, n_aux)

        # Compute the auxiliary metric information at each node
        for i = 1:Trixi.nnodes(dg)
            r, s = rd.rst[1][i], rd.rst[2][i]
            # Covariant basis in the desired global coordinate system as columns of a matrix
            basis_covariant = TrixiAtmo.calc_basis_covariant(
                v1,
                v2,
                v3,
                r,
                s,
                radius,
                equations.global_coordinate_system,
            )

            aux_node[1:6] = SVector(basis_covariant)


            # Covariant metric tensor G := basis_covariant' * basis_covariant
            metric_covariant = basis_covariant' * basis_covariant

            # Contravariant metric tensor inv(G)
            metric_contravariant = inv(metric_covariant)

            # Contravariant basis vectors as rows of a matrix
            basis_contravariant = metric_contravariant * basis_covariant'


            aux_node[7:12] = SVector(basis_contravariant)
            # Area element
            aux_node[13] = sqrt(det(metric_covariant))

            # Covariant metric tensor components
            aux_node[14:16] = SVector(
                metric_covariant[1, 1],
                metric_covariant[1, 2],
                metric_covariant[2, 2],
            )

            # Contravariant metric tensor components
            aux_node[17:19] = SVector(
                metric_contravariant[1, 1],
                metric_contravariant[1, 2],
                metric_contravariant[2, 2],
            )
            # Bottom topography
            if !isnothing(bottom_topography)
                x_node = map(coords -> coords[i, element], xyz)
                aux_node[20] = bottom_topography(x_node)
                # TODO: Compute the derivatives of the bottom topography and store them in aux_node[27] and aux_node[28]
            else
                aux_node[20] = zero(eltype(aux_node))
                aux_node[27] = zero(eltype(aux_node))
                aux_node[28] = zero(eltype(aux_node))
            end
            if aux_node[27] == Inf
                aux_node[27] = zero(eltype(aux_node))
            end
            if aux_node[28] == Inf
                aux_node[28] = zero(eltype(aux_node))
            end
            aux_values[i, element] = SVector{n_aux}(aux_node)
        end
        # Christoffel symbols of the second kind (aux_values[21:26, :, :, element])
        TrixiAtmo.calc_christoffel_symbols!(aux_values, mesh, equations, dg, element)
    end

    return nothing
end

# Extract the derivatives of the bottom topography ∂hₛ/∂r and ∂hₛ/∂s from the auxiliary variables
@inline function bottom_topography_derivatives(
    aux_vars,
    ::TrixiAtmo.AbstractCovariantEquations{2},
)
    return SVector{2}(aux_vars[27], aux_vars[28])
end


##############################################################################
# Initial condition and bottom topography

@inline function initial_condition_smooth_mountain(x, t, equations)
    RealT = eltype(x)
    a = sqrt(x[1]^2 + x[2]^2 + x[3]^2)  # radius of the sphere
    lat = asin(clamp(x[3] / a, -one(RealT), one(RealT)))
    h_0 = 5960.0f0
    v_0 = 20.0f0

    # compute zonal and meridional components of the velocity
    vlon, vlat = v_0 * cos(lat), zero(eltype(x))

    # compute geopotential height 
    h =
        h_0 -
        1 / EARTH_GRAVITATIONAL_ACCELERATION *
        (a * EARTH_ROTATION_RATE * v_0 + 0.5f0 * v_0^2) *
        (sin(lat))^2

    # Convert primitive variables from spherical coordinates to the chosen global 
    # coordinate system, which depends on the equation type
    return TrixiAtmo.spherical2global(
        SVector(h, vlon, vlat, zero(RealT), bottom_topography_smooth_mountain(x)),
        x,
        equations,
    )
end

# Bottom topography function to pass as auxiliary_field keyword argument in constructor for 
# SemidiscretizationHyperbolic, used with initial_condition_smooth_mountain
@inline function bottom_topography_smooth_mountain(x)
    RealT = eltype(x)
    a = sqrt(x[1]^2 + x[2]^2 + x[3]^2)  # radius of the sphere
    lon, lat = atan(x[2], x[1]), asin(clamp(x[3] / a, -one(RealT), one(RealT)))

    # Position and height of mountain, noting that latitude is λ = -π/2 and not λ = 3π/2 
    # because atan(y,x) is in [-π, π]
    lon_0, lat_0 = convert(RealT, -π / 2), convert(RealT, π / 6)
    b_0 = 2000.0f0

    R = convert(RealT, π / 9)
    return b_0 * exp(-((lon - lon_0)^2 + (lat - lat_0)^2) / R^2)
end

equations = CovariantShallowWaterEquations2D(
    EARTH_GRAVITATIONAL_ACCELERATION,
    EARTH_ROTATION_RATE,
    global_coordinate_system = GlobalCartesianCoordinates(),
)

###############################################################################
# Build DG solver.

polydeg = 4

dg = DGMulti(
    element_type = Tri(),
    approximation_type = Polynomial(),
    surface_flux = flux_lax_friedrichs,
    polydeg = polydeg,
)

###############################################################################
# Build mesh.

initial_refinement_level = 4

mesh = DGMultiMeshTriIcosahedron2D(
    dg,
    EARTH_RADIUS;
    initial_refinement = initial_refinement_level,
)

# Transform the initial condition to the proper set of conservative variables
initial_condition_transformed = transform_initial_condition(initial_condition, equations)

# Standard geometric and Coriolis source terms for a rotating sphere
@inline function source_terms_geometric_coriolis_bottom_topography(
    u,
    x,
    t,
    aux_vars,
    equations::CovariantShallowWaterEquations2D,
)
    # Geometric variables
    Gcon = TrixiAtmo.metric_contravariant(aux_vars, equations)
    Gamma1, Gamma2 = TrixiAtmo.christoffel_symbols(aux_vars, equations)
    J = TrixiAtmo.area_element(aux_vars, equations)

    # Physical variables
    h = waterheight(u, equations)
    h_vcon = TrixiAtmo.momentum_contravariant(u, equations)
    v_con = TrixiAtmo.velocity_contravariant(u, equations)

    # Doubly-contravariant flux tensor
    momentum_flux = h_vcon * v_con' + 0.5f0 * equations.gravity * h^2 * Gcon

    # Coriolis parameter
    f = 2 * equations.rotation_rate * x[3] / sqrt(x[1]^2 + x[2]^2 + x[3]^2)  # 2Ωsinθ

    # Geometric source term
    s_geo = SVector(sum(Gamma1 .* momentum_flux), sum(Gamma2 .* momentum_flux))

    # Combined source terms
    source_1 = s_geo[1] + f * J * (Gcon[1, 2] * h_vcon[1] - Gcon[1, 1] * h_vcon[2])
    source_2 = s_geo[2] + f * J * (Gcon[2, 2] * h_vcon[1] - Gcon[2, 1] * h_vcon[2])

    # TODO: Add bottom topography source term

    # Do not scale by Jacobian since apply_jacobian! is called before this
    return SVector(zero(eltype(u)), -source_1, -source_2)
end

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(
    mesh,
    equations,
    initial_condition_transformed,
    dg,
    source_terms = source_terms_geometric_coriolis_bottom_topography,
    auxiliary_field = bottom_topography_smooth_mountain,
)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0 to T
ode = semidiscretize(semi, (0.0, 15 * SECONDS_PER_DAY))

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation 
# setup and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the 
# results
analysis_callback = AnalysisCallback(
    semi,
    interval = 100,
    save_analysis = true,
    extra_analysis_errors = (:conservation_error,),
    uEltype = real(dg),
)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution =
    SaveSolutionCallback(interval = 300, solution_variables = cons2prim_and_vorticity)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl = 0.7)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE 
# solver
callbacks =
    CallbackSet(summary_callback, analysis_callback, save_solution, stepsize_callback)

###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed 
# callbacks
sol = solve(
    ode,
    CarpenterKennedy2N54(williamson_condition = false),
    dt = 1.0,
    save_everystep = false,
    callback = callbacks,
)
