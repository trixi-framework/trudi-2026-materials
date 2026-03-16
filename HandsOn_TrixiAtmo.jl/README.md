# Hand-on TrixiAtmo.jl

Initiate a Julia project using the files in the current directory.

```shell
julia --project -e 'using Pkg; Pkg.initiate()'
```

## SWE with bottom topography

## Vector invariant form

The code is located in the subfolder `vector_invariant`.

The task is to add fluxes to `equations/compressible_euler_vectorinvariant_2d.jl` and use those in `elixir_euler_equations_vector_invariant_warm_bubble.jl`.

```
julia> include("vector_invariant/elixir_euler_equations_vector_invariant_warm_bubble.jl")
```
