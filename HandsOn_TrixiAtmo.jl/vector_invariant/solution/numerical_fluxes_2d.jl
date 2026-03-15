@muladd begin

@inline function flux_surface_combined(
    u_ll,
    u_rr,
    normal_direction::AbstractVector,
    equations::CompressibleEulerVectorInvariantEquations2D,
)

    rho_ll, v1_ll, v2_ll, rho_theta_ll = u_ll
    rho_rr, v1_rr, v2_rr, rho_theta_rr = u_rr
    _, _, _, exner_ll = cons2primexner(u_ll, equations)
    _, _, _, exner_rr = cons2primexner(u_rr, equations)
    theta_ll = rho_theta_ll / rho_ll
    theta_rr = rho_theta_rr / rho_rr

    v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
    v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]

    # Average each factor of products in flux
    rho_avg = 0.5f0 * (rho_ll + rho_rr)
    kin_avg = 0.5f0 * (v1_rr * v1_rr + v2_rr * v2_rr + v1_ll * v1_ll + v2_ll * v2_ll)
    theta_avg = 0.5f0 * (theta_ll + theta_rr)
    v1_avg = 0.5f0 * (v1_ll + v1_rr)
    v2_avg = 0.5f0 * (v2_ll + v2_rr)

    jump_v1 = v1_rr - v1_ll
    jump_v2 = v2_rr - v2_ll

    rho_v_ll = v1_ll * rho_ll * normal_direction[1] + v2_ll * rho_ll * normal_direction[2]
    rho_v_rr = v1_rr * rho_rr * normal_direction[1] + v2_rr * rho_rr * normal_direction[2]
    c = 340
    c_adv = 0.5f0 * abs((v_dot_n_ll + v_dot_n_rr)) / norm(normal_direction)
    diss1 = 0.5f0 * c / rho_avg * (rho_v_rr - rho_v_ll) * normal_direction[1] / norm(normal_direction)
    diss2 = 0.5f0 * c / rho_avg * (rho_v_rr - rho_v_ll) * normal_direction[2] / norm(normal_direction)
    f1 = rho_avg * 0.5f0 * (v_dot_n_ll + v_dot_n_rr)
    f2 =
        kin_avg * 0.5f0 * normal_direction[1] - diss1 -
        0.5f0 * c_adv / rho_avg *
        (rho_rr * v1_rr - rho_ll * v1_ll) *
        norm(normal_direction)
    f3 =
        kin_avg * 0.5f0 * normal_direction[2] - diss2 -
        0.5f0 * c_adv / rho_avg *
        (rho_rr * v2_rr - rho_ll * v2_ll) *
        norm(normal_direction)
    if f1 >= 0
        f4 = f1 * theta_ll
    else
        f4 = f1 * theta_rr
    end
    g2 =
        v2_avg * jump_v1 * normal_direction[2] -
        v2_avg * jump_v2 * normal_direction[1] +
        equations.c_p * theta_avg * (exner_rr - exner_ll) * normal_direction[1]
    g3 =
        v1_avg * jump_v2 * normal_direction[1] -
        v1_avg * jump_v1 * normal_direction[2] +
        equations.c_p * theta_avg * (exner_rr - exner_ll) * normal_direction[2]
    return SVector(f1, f2 + 0.5f0 * g2, f3 + 0.5f0 * g3, f4, zero(eltype(u_ll))),
           SVector(f1, f2 - 0.5f0 * g2, f3 - 0.5f0 * g3, f4, zero(eltype(u_ll)))
end

@inline Trixi.combine_conservative_and_nonconservative_fluxes(::typeof(flux_surface_combined), equations::CompressibleEulerVectorInvariantEquations2D) = Trixi.True()

@inline function flux_volume_combined(
    u_ll,
    u_rr,
    normal_direction::AbstractVector,
    equations::CompressibleEulerVectorInvariantEquations2D,
)

    rho_ll, v1_ll, v2_ll, rho_theta_ll, phi_ll = u_ll
    rho_rr, v1_rr, v2_rr, rho_theta_rr, phi_rr = u_rr
    _, _, _, exner_ll = cons2primexner(u_ll, equations)
    _, _, _, exner_rr = cons2primexner(u_rr, equations)
    theta_ll = rho_theta_ll / rho_ll
    theta_rr = rho_theta_rr / rho_rr

    v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
    v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]

    # Average each factor of products in flux
    rho_avg = 0.5f0 * (rho_ll + rho_rr)
    kin_avg = 0.5f0 * (v1_rr * v1_rr + v2_rr * v2_rr + v1_ll * v1_ll + v2_ll * v2_ll)
    theta_avg = 0.5f0 * (theta_ll + theta_rr)
    v1_avg = 0.5f0 * (v1_ll + v1_rr)
    v2_avg = 0.5f0 * (v2_ll + v2_rr)

    jump_v1 = v1_rr - v1_ll
    jump_v2 = v2_rr - v2_ll

    f1 = rho_avg * 0.5f0 * (v_dot_n_ll + v_dot_n_rr)
    f2 =
        kin_avg * 0.5f0 * normal_direction[1] 
    f3 =
        kin_avg * 0.5f0 * normal_direction[2]
    f4 = f1 *theta_avg
    gravity = phi_rr - phi_ll
    g2 =
        v2_avg * jump_v1 * normal_direction[2] -
        v2_avg * jump_v2 * normal_direction[1] +
        equations.c_p * theta_avg * (exner_rr - exner_ll) * normal_direction[1] + gravity * normal_direction[1]
    g3 =
        v1_avg * jump_v2 * normal_direction[1] -
        v1_avg * jump_v1 * normal_direction[2] +
        equations.c_p * theta_avg * (exner_rr - exner_ll) * normal_direction[2] + gravity * normal_direction[2]
    return SVector(f1, f2 + 0.5f0 * g2, f3 + 0.5f0 * g3, f4, zero(eltype(u_ll))),
           SVector(f1, f2 - 0.5f0 * g2, f3 - 0.5f0 * g3, f4, zero(eltype(u_ll)))
end

@inline Trixi.combine_conservative_and_nonconservative_fluxes(::typeof(flux_volume_combined), equations::CompressibleEulerVectorInvariantEquations2D) = Trixi.True()

@inline function flux_conservative_surface(
    u_ll,
    u_rr,
    normal_direction::AbstractVector,
    equations::CompressibleEulerVectorInvariantEquations2D,
)

    rho_ll, v1_ll, v2_ll, rho_theta_ll = u_ll
    rho_rr, v1_rr, v2_rr, rho_theta_rr = u_rr
    theta_ll = rho_theta_ll / rho_ll
    theta_rr = rho_theta_rr / rho_rr

    v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
    v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]

    # Average each factor of products in flux
    rho_avg = 0.5f0 * (rho_ll + rho_rr)
    kin_avg = 0.5f0 * (v1_rr * v1_rr + v2_rr * v2_rr + v1_ll * v1_ll + v2_ll * v2_ll)

    rho_v_ll = v1_ll * rho_ll * normal_direction[1] + v2_ll * rho_ll * normal_direction[2]
    rho_v_rr = v1_rr * rho_rr * normal_direction[1] + v2_rr * rho_rr * normal_direction[2]
    c = 340
    c_adv = 0.5f0 * abs((v_dot_n_ll + v_dot_n_rr)) / norm(normal_direction)
    diss1 = 0.5f0 * c / rho_avg * (rho_v_rr - rho_v_ll) * normal_direction[1] / norm(normal_direction)
    diss2 = 0.5f0 * c / rho_avg * (rho_v_rr - rho_v_ll) * normal_direction[2] / norm(normal_direction)
    f1 = rho_avg * 0.5f0 * (v_dot_n_ll + v_dot_n_rr)
    f2 = kin_avg * 0.5f0 * normal_direction[1] - diss1 - 0.5f0 * c_adv / rho_avg * (rho_rr * v1_rr - rho_ll * v1_ll) * norm(normal_direction)
    f3 = kin_avg * 0.5f0 * normal_direction[2] - diss2 - 0.5f0 * c_adv / rho_avg * (rho_rr * v2_rr - rho_ll * v2_ll) * norm(normal_direction)
    if f1 >= 0
        f4 = f1 * theta_ll
    else
        f4 = f1 * theta_rr
    end
    return SVector(f1, f2, f3, f4, zero(eltype(u_ll)))
end


@inline function flux_conservative_volume(
    u_ll,
    u_rr,
    normal_direction::AbstractVector,
    equations::CompressibleEulerVectorInvariantEquations2D,
)

    rho_ll, v1_ll, v2_ll, rho_theta_ll = u_ll
    rho_rr, v1_rr, v2_rr, rho_theta_rr = u_rr
    theta_ll = rho_theta_ll / rho_ll
    theta_rr = rho_theta_rr / rho_rr

    v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
    v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]

    # Average each factor of products in flux
    rho_avg = 0.5f0 * (rho_ll + rho_rr)
    kin_avg = 0.5f0 * (v1_rr * v1_rr + v2_rr * v2_rr + v1_ll * v1_ll + v2_ll * v2_ll)
    theta_avg = 0.5f0 * (theta_ll + theta_rr)

    f1 = rho_avg * 0.5f0 * (v_dot_n_ll + v_dot_n_rr) 
    f2 = kin_avg * 0.5f0 * normal_direction[1]
    f3 = kin_avg * 0.5f0 * normal_direction[2]         
    f4 = f1 * theta_avg
    
    return SVector(f1, f2, f3, f4, zero(eltype(u_ll)))
end

@inline function flux_nonconservative_surface(
    u_ll,
    u_rr,
    normal_direction::AbstractVector,
    equations::CompressibleEulerVectorInvariantEquations2D,
)

    rho_ll, v1_ll, v2_ll, rho_theta_ll = u_ll
    rho_rr, v1_rr, v2_rr, rho_theta_rr = u_rr
    _, _, _, exner_ll = cons2primexner(u_ll, equations)
    _, _, _, exner_rr = cons2primexner(u_rr, equations)
    theta_ll = rho_theta_ll / rho_ll
    theta_rr = rho_theta_rr / rho_rr

    # Average each factor of products in flux
    rho_avg = 0.5f0 * (rho_ll + rho_rr)
    theta_avg = 0.5f0 * (theta_ll + theta_rr)
    v1_avg = 0.5f0 * (v1_ll + v1_rr)
    v2_avg = 0.5f0 * (v2_ll + v2_rr)

    jump_v1 = v1_rr - v1_ll
    jump_v2 = v2_rr - v2_ll
    g2 = v2_avg * jump_v1 * normal_direction[2] - v2_avg * jump_v2 * normal_direction[1] + equations.c_p * theta_avg * (exner_rr - exner_ll) * normal_direction[1] 
    g3 = v1_avg * jump_v2 * normal_direction[1] - v1_avg * jump_v1 * normal_direction[2] + equations.c_p * theta_avg * (exner_rr - exner_ll) * normal_direction[2]
	return SVector(zero(eltype(u_ll)), g2, g3, zero(eltype(u_ll)), zero(eltype(u_ll)))
end

@inline function flux_nonconservative_volume(
    u_ll,
    u_rr,
    normal_direction::AbstractVector,
    equations::CompressibleEulerVectorInvariantEquations2D,
)

    rho_ll, v1_ll, v2_ll, rho_theta_ll, phi_ll = u_ll
    rho_rr, v1_rr, v2_rr, rho_theta_rr, phi_rr = u_rr
    _, _, _, exner_ll = cons2primexner(u_ll, equations)
    _, _, _, exner_rr = cons2primexner(u_rr, equations)
    theta_ll = rho_theta_ll / rho_ll
    theta_rr = rho_theta_rr / rho_rr

    # Average each factor of products in flux
    rho_avg = 0.5f0 * (rho_ll + rho_rr)
    theta_avg = 0.5f0 * (theta_ll + theta_rr)
    v1_avg = 0.5f0 * (v1_ll + v1_rr)
    v2_avg = 0.5f0 * (v2_ll + v2_rr)

    jump_v1 = v1_rr - v1_ll
    jump_v2 = v2_rr - v2_ll
gravity = phi_rr - phi_ll
    g2 = v2_avg * jump_v1 * normal_direction[2] - v2_avg * jump_v2 * normal_direction[1] + equations.c_p * theta_avg * (exner_rr - exner_ll) * normal_direction[1] + gravity * normal_direction[1]
    g3 = v1_avg * jump_v2 * normal_direction[1] - v1_avg * jump_v1 * normal_direction[2] + equations.c_p * theta_avg * (exner_rr - exner_ll) * normal_direction[2] + gravity * normal_direction[2]
	return SVector(zero(eltype(u_ll)), g2, g3, zero(eltype(u_ll)), zero(eltype(u_ll)))
end

end
