using Trixi
using IntelITT

EXAMPLES_DIR = joinpath(examples_dir(), "p4est_2d_dgsem")

trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
	      sol = nothing, initial_refinement_level=5)


solve(ode, CarpenterKennedy2N54(williamson_condition = false);
          dt = 1.0, ode_default_options()...,
		  callback = callbacks)


IntelITT.resume()

IntelITT.@task "ode" solve(ode, CarpenterKennedy2N54(williamson_condition = false);
          dt = 1.0, ode_default_options()...,
		  callback = callbacks)
		  # callback = CallbackSet(stepsize_callback))

IntelITT.pause()
