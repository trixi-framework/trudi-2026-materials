using Trixi

EXAMPLES_DIR = joinpath(examples_dir(), "p4est_2d_dgsem")

trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"))
