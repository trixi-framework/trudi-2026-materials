# Hands-on: 🌎 TrixiAtmo.jl 🌍

Instantiate a Julia project using the files in the current directory.

```shell
julia --project -e 'using Pkg; Pkg.instantiate()'
```

## SWE with bottom topography

## Vector invariant form

The code is located in the subfolder `vector_invariant`.

The task is to add fluxes to `equations/compressible_euler_vectorinvariant_2d.jl` and use them
 in `elixir_euler_equations_vector_invariant_warm_bubble.jl`.

```julia
julia> include("vector_invariant/elixir_euler_equations_vector_invariant_warm_bubble.jl")
```

## Unified equations

This is just a proof of concept and living in a branch.

- Get a local checkout of `TrixiAtmo.jl`, e.g. by using
  ```julia
  (run) pkg> dev --local TrixiAtmo
  ```
- Go to `dev/TrixiAtmo`
- Change the branch:
  ```shell
  git checkout bg/primitive-equations
  ```
- Go back to your Julia project and update TrixiAtmo:
   ```julia
  (run) pkg> update TrixiAtmo
  ```

1. Run the elixir:
    ```julia
    julia> using Trixi; trixi_include("dev/TrixiAtmo/examples/euler/moist_air/buoyancy_unified/elixir_bubble_bryan_fritsch.jl")
    ```
    You should see some upwards moving structure.

2. Change the thermodynamic equation:
    ```julia
    td_equation = TotalEnergy(td_state)
    ```
    and run again. Your are now solving a different equation set, but the result should be similar.

3.  Now add one condensate and change the thermodynics as well:
    ```julia
    n_condens = 1
    [...]
    td_state = IdealGasesAndLiquids(; parameters, n_gas = 2, n_condens = 1)
    ```
    You have added vapor and cloud water, but the bubble will not rise anymore.

4.  Implement phase conversion terms in `src/parametrization/microphysics.jl`, or just look for `saturation_factor` in the elixir, set it to `1.0`, and run again.
