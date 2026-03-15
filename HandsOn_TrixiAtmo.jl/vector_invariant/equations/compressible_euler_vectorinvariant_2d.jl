using Trixi
using Trixi: AbstractCompressibleEulerEquations, @muladd, norm
import Trixi:
    varnames,
    cons2cons,
    cons2prim,
    cons2entropy,
    entropy,
    FluxLMARS,
    boundary_condition_slip_wall,
    flux,
    max_abs_speeds,
    max_abs_speed,
    max_abs_speed_naive,
    have_nonconservative_terms,
    True, prim2cons, False
# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

struct CompressibleEulerVectorInvariantEquations2D{RealT<:Real} <:
       AbstractCompressibleEulerEquations{2,5}
    p_0::RealT # reference pressure in Pa
    c_p::RealT # specific heat at constant pressure in J/(kg K)
    c_v::RealT # specific heat at constant volume in J/(kg K)
    g::RealT # gravitational acceleration in m/s²
    R::RealT # gas constant in J/(kg K)
    gamma::RealT # ratio of specific heats 
    inv_gamma_minus_one::RealT # = inv(gamma - 1); can be used to write slow divisions as fast multiplications
    K::RealT # = p_0 * (R / p_0)^gamma; scaling factor between pressure and weighted potential temperature
    stolarsky_factor::RealT # = (gamma - 1) / gamma; used in the stolarsky mean
    function CompressibleEulerVectorInvariantEquations2D(; c_p, c_v, gravity)
        c_p, c_v, g = promote(c_p, c_v, gravity)
        p_0 = 100_000
        R = c_p - c_v
        gamma = c_p / c_v
        inv_gamma_minus_one = inv(gamma - 1)
        K = p_0 * (R / p_0)^gamma
        stolarsky_factor = (gamma - 1) / gamma
        return new{typeof(c_p)}(
            p_0,
            c_p,
            c_v,
            g,
            R,
            gamma,
            inv_gamma_minus_one,
            K,
            stolarsky_factor,
        )
    end
end

function varnames(::typeof(cons2cons), ::CompressibleEulerVectorInvariantEquations2D)
    ("rho", "v1", "v2", "rho_theta", "phi")
end

have_nonconservative_terms(::CompressibleEulerVectorInvariantEquations2D) = True()

varnames(::typeof(cons2prim), ::CompressibleEulerVectorInvariantEquations2D) =
    ("rho", "v1", "v2", "p", "phi")


@inline function Trixi.boundary_condition_slip_wall(
    u_inner,
    normal_direction::AbstractVector,
    x,
    t,
    surface_flux_function,
    equations::CompressibleEulerVectorInvariantEquations2D
)

    Trixi.boundary_condition_slip_wall(u_inner, normal_direction, x, t, surface_flux_function, equations, Trixi.combine_conservative_and_nonconservative_fluxes(surface_flux_function, equations))
end    

@inline function Trixi.boundary_condition_slip_wall(
    u_inner,
    normal_direction::AbstractVector,
    x,
    t,
    surface_flux_function,
    equations::CompressibleEulerVectorInvariantEquations2D, combine_conservative_and_nonconservative_fluxes::True
)

    # normalize the outward pointing direction
    normal = normal_direction / norm(normal_direction)

    # compute the normal velocity
    u_normal = normal[1] * u_inner[2] + normal[2] * u_inner[3]

    # create the "external" boundary solution state
    u_boundary = SVector(
        u_inner[1],
        u_inner[2] - 2 * u_normal * normal[1],
        u_inner[3] - 2 * u_normal * normal[2],
        u_inner[4],
        u_inner[5],
    )

    #   calculate the boundary flux
    flux, _ = surface_flux_function(u_inner, u_boundary, normal_direction, equations)
    return flux
end

@inline function Trixi.boundary_condition_slip_wall(
    u_inner,
    normal_direction::AbstractVector,
    x,
    t,
    surface_flux_functions,
    equations::CompressibleEulerVectorInvariantEquations2D, combine_conservative_and_nonconservative_fluxes::False
)
    surface_flux_function, nonconservative_flux_function = surface_flux_functions
    # normalize the outward pointing direction
    normal = normal_direction / norm(normal_direction)

    # compute the normal velocity
    u_normal = normal[1] * u_inner[2] + normal[2] * u_inner[3]

    # create the "external" boundary solution state
    u_boundary = SVector(
        u_inner[1],
        u_inner[2] - 2 * u_normal * normal[1],
        u_inner[3] - 2 * u_normal * normal[2],
        u_inner[4],
        u_inner[5],
    )

    #   calculate the boundary flux
    flux = surface_flux_function(u_inner, u_boundary, normal_direction, equations)
    noncons_flux = nonconservative_flux_function(u_inner, u_boundary, normal_direction, equations)

    return flux, noncons_flux
end


# Less "cautious", i.e., less overestimating `λ_max` compared to `max_abs_speed_naive`
@inline function max_abs_speed(
    u_ll,
    u_rr,
    normal_direction::AbstractVector,
    equations::CompressibleEulerVectorInvariantEquations2D,
)
    rho_ll, v1_ll, v2_ll, p_ll = cons2prim(u_ll, equations)
    rho_rr, v1_rr, v2_rr, p_rr = cons2prim(u_rr, equations)

    # Calculate normal velocities and sound speeds
    # left
    v_ll = (v1_ll * normal_direction[1] + v2_ll * normal_direction[2])
    c_ll = sqrt(equations.gamma * p_ll / rho_ll)
    # right
    v_rr = (v1_rr * normal_direction[1] + v2_rr * normal_direction[2])
    c_rr = sqrt(equations.gamma * p_rr / rho_rr)

    norm_ = norm(normal_direction)
    return max(abs(v_ll) + c_ll * norm_, abs(v_rr) + c_rr * norm_)
end

@inline function max_abs_speeds(
    u,
    equations::CompressibleEulerVectorInvariantEquations2D,
)
    rho, v1, v2, p = cons2prim(u, equations)
    c = sqrt(equations.gamma * p / rho)

    return abs(v1) + c, abs(v2) + c
end

# Convert conservative variables to primitive
@inline function cons2prim(u, equations::CompressibleEulerVectorInvariantEquations2D)
    rho, v1, v2, rho_theta, phi = u

    p = equations.K * rho_theta^equations.gamma

    return SVector(rho, v1, v2, p, phi)
end

# Convert primitive to conservative variables
@inline function prim2cons(prim, equations::CompressibleEulerVectorInvariantEquations2D)
    rho, v1, v2, p, phi = prim
    rho_theta = (p / equations.p_0)^(1 / equations.gamma) * equations.p_0 / equations.R
    return SVector(rho, v1, v2, rho_theta, phi)
end

@inline function density(u, equations::CompressibleEulerVectorInvariantEquations2D)
    rho = u[1]
    return rho
end

@inline function velocity(u, equations::CompressibleEulerVectorInvariantEquations2D)
    v1 = u[2]
    v2 = u[3]
    return SVector(v1, v2)
end

@inline function pressure(u, equations::CompressibleEulerVectorInvariantEquations2D)
    rho, v1, v2, rho_theta = u
    p = equations.K * rho_theta^equations.gamma
    return p
end

@inline function cons2primexner(
    u,
    equations::CompressibleEulerVectorInvariantEquations2D,
)

    rho, v1, v2, rho_theta, phi = u

    exner = (rho_theta * equations.R / equations.p_0)^(equations.R / equations.c_v)
    return SVector(rho, v1, v2, exner, phi)
end

@inline function exner_pressure(
    u,
    equations::CompressibleEulerVectorInvariantEquations2D,
)
    _, _, _, rho_theta = u

    exner = (rho_theta * equations.R / equations.p_0)^(equations.R / equations.c_v)
    return exner
end

@inline function cons2entropy(u, equations::CompressibleEulerVectorInvariantEquations2D)
    rho, v1, v2, rho_theta = u
	# TODO:
    return SVector(0, 0, 0, 0, 0)
end

@inline function well_balanced_v1(
    u,
    equations::CompressibleEulerVectorInvariantEquations2D,
)
    rho, v1, v2, rho_theta, _ = u
    return abs(v1)
end

@inline function well_balanced_v2(
    u,
    equations::CompressibleEulerVectorInvariantEquations2D,
)
    rho, v1, v2, rho_theta, _ = u
    return abs(v2)
end
end # @muladd
