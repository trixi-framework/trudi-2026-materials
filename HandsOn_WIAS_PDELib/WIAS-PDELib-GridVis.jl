### A Pluto.jl notebook ###
# v0.20.23

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 614b2d29-5bae-47c7-a349-26f3796c12bb
using ExtendableGrids: ExtendableGrids, simplexgrid

# ╔═╡ 155b8646-faca-4b9f-bce6-47fb29139e6f
using ExtendableGrids: NodeCells, CellEdges, Coordinates, CellNodes

# ╔═╡ 4faca367-1f09-4cbf-811d-f249d4d397f8
begin
    using GridVisualize: gridplot, default_plotter!
    import PlutoVista
    default_plotter!(PlutoVista)
end;

# ╔═╡ f6dce047-b283-4e92-855d-c85b11dc474e
using ExtendableGrids: cellmask!, bfacemask!

# ╔═╡ 369d96ec-eb77-4729-863c-2e8f3f823084
using ExtendableGrids: glue, geomspace

# ╔═╡ 806e23f0-cbe5-48aa-bf10-98cfac4d826a
using GridVisualize: GridVisualizer, scalarplot!, reveal, scalarplot, gridplot!

# ╔═╡ 794558f5-48cc-49dc-93bd-c596560848f6
begin
    using SimplexGridFactory: SimplexGridBuilder, point!, facetregion!, facet!, cellregion!, regionpoint!, options!, maxvolume!, holepoint!
    import Triangulate
end

# ╔═╡ d8a97d1e-f689-4aca-8bdc-0e08d160ba2a
using Gmsh: gmsh

# ╔═╡ 391f4cd3-fd8e-48df-b1f8-d806607dc619
using ExtendableGrids: partition, RecursiveMetisPartitioning, PlainMetisPartitioning

# ╔═╡ 5f9c4f4c-91a1-4c92-abc1-781e4c8898fb
md"""
We introduce the WIAS-PDELib Julia package ecosystem for numerical PDE solving. This presentation focuses on grid management with ExtendableGrids.jl, covering 1D, 2D, and 3D tensor product and unstructured simplex grids. We demonstrate mesh generation using Triangle, TetGen, and Gmsh backends, visualization via GridVisualize.jl with PlutoVista and CairoMakie, and grid partitioning with Metis.
"""

# ╔═╡ 4e6e769b-0878-40fb-bf12-1aaf9fb34b93
md"""
## Intro
"""

# ╔═╡ 505433d7-e34e-4cd1-94ef-bf7b67f19f0a
md"""
__Top level packages__
- [VoronoiFVM.jl](https://github.com/WIAS-PDELib/VoronoiFVM.jl): Nonlinear multiphysics PDE systems using Voronoi finite volumes
- [ExtendableFEM.jl](https://github.com/WIAS-PDELib/ExtendableFEM.jl): High Level API FEM based on [ExtendableGrids](https://github.com/WIAS-PDELib/ExtendableGrids.jl) and [ExtendableFEMBase](https://github.com/WIAS-PDELib/ExtendableFEMBase.jl)
- [ExtendableASGFEM.jl](https://github.com/WIAS-PDELib/ExtendableASGFEM.jl): Adaptive Stochastic Galerkin FEM for parametric PDEs

__Support packages:__
- [ExtendableGrids.jl](https://github.com/WIAS-PDELib/ExtendableGrids.jl): Grid management for FEM & FVM computations
- [ExtendableFEMBase.jl](https://github.com/WIAS-PDELib/ExtendableFEMBase.jl):Basic structures for FEM based on [ExtendableGrids](https://github.com/WIAS-PDELib/ExtendableGrids.jl)
- [ExtendableSparse.jl](https://github.com/WIAS-PDELib/ExtendableSparse.jl): Sparse matrix class with efficient successive insertion of entries
- [AMGCLWrap.jl](https://github.com/j-fu/AMGCLWrap.jl): Wrapper around AMGCL multigrid solver package
- [SimplexGridFactory.jl](https://github.com/WIAS-PDELib/SimplexGridFactory.jl): Convenience interfaces to [Triangulate.jl](https://github.com/JuliaGeometry/Triangulate.jl) and [TetGen.jl](https://github.com/JuliaGeometry/TetGen.jl)
- [GridVisualize.jl](https://github.com/WIAS-PDELib/GridVisualize.jl): Grid & function visualization with [ExtendableGrids](https://github.com/WIAS-PDELib/ExtendableGrids.jl)
- [GridVisualizeTools.jl](https://github.com/WIAS-PDELib/GridVisualizeTools.jl): Marching triangles, tetrahedra & Co
- [PlutoVista.jl](https://github.com/j-fu/PlutoVista.jl): Visualization in Pluto notebooks using plotly.js and vtk.js
- [ExampleJuggler.jl](https://github.com/j-fu/ExampleJuggler.jl): Manage script and notebook examples for CI and docs
- [LessUnitful.jl](https://github.com/j-fu/LessUnitful.jl): Less painful API around [Unitful.jl](https://github.com/JuliaPhysics/Unitful.jl)

__Application packages:__
- [ChargeTransport.jl](https://github.com/WIAS-PDELib/ChargeTransport.jl): Drift-diffusion simulator for semiconductor devices 
- [ElectroMechanicsFEM.jl](https://github.com/WIAS-PDELib/ElectroMechanicsFEM.jl): Finite-strain bent nanowires and bimetallic beams
- [LiquidElectrolytes.jl](https://github.com/j-fu/LiquidElectrolytes.jl): Electrolyte models with finite ion sizes and solvation
"""

# ╔═╡ 47103e8f-a4ff-46ed-a632-572a2e194a50
md"""
## 1D grids

1D grids are created just from arrays of montonically increasing  coordinate values.
"""

# ╔═╡ 93a0c45e-d6a3-415a-a82c-e4f7e2a09d22
X1 = range(0, 1; length = 11)

# ╔═╡ 4622a1fc-fda7-4211-9cc0-4eb1a1584aa6
g1 = simplexgrid(X1)

# ╔═╡ bb3bd2ed-6ca4-4c87-a72b-b5a3b2ec43d2
md"""
The ExtendableGrid struct just wraps a Dict which contains types as keys. This allows to use multiple dispatch on the `getindex`  method to implement lazy creation of higher order adjacencies.
"""

# ╔═╡ 9e6dfe3f-0c1d-4225-bc15-362004bd8e27
g1.components

# ╔═╡ 36498181-6d35-40c6-8a46-93a4070ab87b
g1[NodeCells]

# ╔═╡ 135d7271-3a4f-4275-9578-08d90561b503
md"""
## 1D Grid plots
"""

# ╔═╡ 5520b8c0-0874-4790-a956-224e6c43d9cf
gridplot(g1; resolution = (500, 150), legend = :rt)

# ╔═╡ 13aef2a1-5745-4fe5-9659-6b7c0e7267fc
md"""
We see some  additional information:

- `cellregion`: grid cells (interval, triangle, tetrahedron) have a region marker attached 
- `bfaceregion`: boundary faces (points, lines, triangles) have a boundary region marker attached
"""

# ╔═╡ 2d392a98-bf32-4145-ae93-a8e218367277
md"""
## Modifying region markers

The `simplexgrid` method provides a default distribution of markers, but we would like to be able to change them as core support for multiphysics problems. This can be done by putting masks on cells or faces (points in 1D):
"""

# ╔═╡ 88247350-aa6c-4876-82c3-3534036d5702
begin
    g2 = deepcopy(g1)
    cellmask!(g2, [0.0], [0.5], 2)
    bfacemask!(g2, [0.5], [0.5], 3)
end


# ╔═╡ 74641f73-2efe-4df8-bebb-ed97e77d869e
gridplot(g2; resolution = (500, 175), legend = :rt)

# ╔═╡ 14fb7977-93cf-4f74-a8ec-b6ee25dbdf86
md"""
## Creating locally refined grids

For this purpose, we just need to create arrays with the corresponding coordinate values. This can be done programmatically.
"""

# ╔═╡ 2d5cb9e1-2d14-415e-b792-c3124901011d
hmin = 0.01 ; hmax = 0.1;

# ╔═╡ b1f903b3-29d7-4909-b7e2-8ef3528c9965
md"""
The `geomspace` method creates an array such that the smallest interval size is `hmin` and the largest interval size is not larger but close to `hmax`, and the interval sizes constitute a geometric sequence.
"""

# ╔═╡ 19125aed-5c46-4968-bcfc-0b469628b68e
begin
    X2L = geomspace(0, 0.5, hmax, hmin)
    X2R = geomspace(0.5, 1, hmin, hmax)
    X2 = glue(X2L, X2R)
end

# ╔═╡ dd7d8e17-f825-4047-9386-d9e2bfd0a48d
gridplot(simplexgrid(X2); resolution = (500, 150), legend = :rt)

# ╔═╡ 3d7db57f-2864-4984-a979-609e1d838a9f
md"""
## Plotting of functions

We assume that functions can be represented by their node values an plotted via their piecewise linear interpolants. E.g. they could come from some simulation.
"""

# ╔═╡ 4cf840e6-86c5-4af1-a780-6fc78b60716b
g1d2 = simplexgrid(range(-10, 10, length = 201));

# ╔═╡ 151cc4b8-c5ed-4f5e-8d5f-2f708c9c7fae
fsin = map(sin, g1d2); fcos = map(cos, g1d2);

# ╔═╡ 71af99ab-612a-4821-8e9c-efc8766e2e3e
let
    vis = GridVisualizer(; resolution = (600, 175), legend = :lt)
    scalarplot!(vis, g1d2, fcos, label = "cos", markershape = :xcross, color = :green, linestyle = :dash, clear = false, markevery = 20)
    scalarplot!(vis, g1d2, fsin, label = "sin", markershape = :none, color = :blue, linestyle = :dot, clear = false, markevery = 20)
    reveal(vis)
end

# ╔═╡ 6dfa1d73-8baa-4589-b2e6-547834c9e444
md"""
## 2D Tensor product grids

We can again use the `simplexgrid` method
and apply the mask methods for modifying cell and boundary region markers.
"""

# ╔═╡ f9599246-8238-432c-a315-300d74abfa2c
begin
    g2d1 = simplexgrid(X1, X2)
    cellmask!(g2d1, [0.0, 0.0], [0.5, 0.5], 2)
    cellmask!(g2d1, [0.5, 0.5], [1.0, 1.0], 3)
    bfacemask!(g2d1, [0.0, 0.0], [0.0, 0.5], 5)
end;

# ╔═╡ ba5af291-fb16-4f21-8b74-664284bf7bd9
gridplot(g2d1, resolution = (500, 300), linewidth = 0.5, legend = :lt)

# ╔═╡ dfbaacea-4cb4-4147-9f1b-4424d7a7e89b
md"""
To interact with the PlutoVista plot, on can use the mouse wheel or double touch to zoom,
"shift-mouse-left" to pan, and "alt-mouse-left" or "ctrl-mouse-left" to reset.
"""

# ╔═╡ 7d1698dd-3bb7-4b38-9c6d-a88652749eee
md"""
We can also have a look into the components of a 2D grid:
"""

# ╔═╡ 81249f7c-abdf-43cc-b57f-2915b09da009
g2d1.components

# ╔═╡ 0765641a-8ed9-4579-bd9b-90bb02a55792
md"""
## 2D Unstructured grids
Leveraging Jonathan Shewchuk's mesh generator [Triangle](https://www.cs.cmu.edu/~quake/triangle.html) via its Julia wrapper package  [Triangulate.jl](https://github.com/JuliaGeometry/Triangulate.jl).

"""

# ╔═╡ 884a11a2-15cf-40fc-a1ca-66ea23c6094e
grid2d2 = let
    b = SimplexGridBuilder(Generator = Triangulate)
    p1 = point!(b, 0, 0);    p2 = point!(b, 1, 0); p3 = point!(b, 1, 1)
    facetregion!(b, 1);     facet!(b, p1, p2)
    facetregion!(b, 2);    facet!(b, p2, p3)
    facetregion!(b, 3);    facet!(b, p3, p1)
    cellregion!(b, 1);    regionpoint!(b, 0.75, 0.25)
    simplexgrid(b; maxvolume = 0.001)
end;

# ╔═╡ 8f7d958e-5dc5-4324-af21-ad829d7d77eb
gridplot(grid2d2, resolution = (400, 300), linewidth = 0.5)

# ╔═╡ 4a289b23-46b9-495d-b19c-42b3da71b242
md"""
## 2D holes & local refinement etc.
"""

# ╔═╡ b12838f0-fe9c-11ea-2939-155ed907322d
const refinement_center = [0.8, 0.2]

# ╔═╡ d5d8a1d6-fe9d-11ea-0fd8-df6e81492cb5
md"""
Define  a function, which is able to tell if a triangle is to be refined ("unsuitable") or can be kept as it is.
"""

# ╔═╡ aae2e82a-fe9c-11ea-0427-593f8d2c7746
function unsuitable(x1, y1, x2, y2, x3, y3, area)
    bary_x = (x1 + x2 + x3) / 3.0
    bary_y = (y1 + y2 + y3) / 3.0
    dx = bary_x - refinement_center[1]
    dy = bary_y - refinement_center[2]
    qdist = dx^2 + dy^2
    return area > 0.1 * max(1.0e-2, qdist)
end;

# ╔═╡ 1ae86964-fe9e-11ea-303b-65bb128384a5
md"""
- Interior boundaries are described in a similar as exterior ones - just by facets connecting points.
- Subregions are defined as regions surrounded by interior boundaries.  By placing a "region point" into such a region and specifying a "region number", we can set the cell region marker for all triangles created in the subregion.
- Holes are defined in a similar way as subregions, but a "hole point" is places into the place which shall become the hole.
"""

# ╔═╡ 511b26c6-f920-11ea-1228-51c3750f495c
grid2d3 = let
    b = SimplexGridBuilder(Generator = Triangulate; tol = 1.0e-10)
    p1 = point!(b, 0, 0); p2 = point!(b, 1, 0); p3 = point!(b, 1, 1); p4 = point!(b, 0, 0.7)
    facetregion!(b, 1); facet!(b, p1, p2)
    facetregion!(b, 2); facet!(b, p2, p3)
    facetregion!(b, 3); facet!(b, p3, p4)
    facetregion!(b, 4); facet!(b, p1, p4)

    options!(b, unsuitable = unsuitable)

    facetregion!(b, 5); facet!(b, p1, p3)

    # Coarse elements in upper left region #1
    cellregion!(b, 1); maxvolume!(b, 0.1);  regionpoint!(b, 0.1, 0.5)

    # Fine elements in lower right region #2
    cellregion!(b, 2);  maxvolume!(b, 0.01); regionpoint!(b, 0.9, 0.5)

    # Hole
    hp1 = point!(b, 0.4, 0.1); hp2 = point!(b, 0.6, 0.1); hp3 = point!(b, 0.5, 0.3); holepoint!(b, 0.5, 0.2)
    facetregion!(b, 6);  facet!(b, hp1, hp2);  facet!(b, hp2, hp3);   facet!(b, hp3, hp1)
    simplexgrid(b)
end;

# ╔═╡ 59a6c8b5-25aa-47aa-9489-a803672013df
gridplot(grid2d3, legend = :lt, resolution = (400, 400))

# ╔═╡ 4c99c40f-cf93-4cba-bef1-0c4ffcbf6833
md"""
## 2D function plot
"""

# ╔═╡ a375c23f-6b8c-4b2c-a8b5-d38e6b5a8f6d
fsin3 = map((x, y) -> sin(3y) * x, grid2d3);

# ╔═╡ 4dfb2e0f-3e3a-4053-8a76-765546e96992
scalarplot(grid2d3, fsin3, label = "grid2d3", size = (500, 400), colormap = :hot, levels = 11, backend = :plotly)

# ╔═╡ afeca036-2014-4ab4-b128-8b8851a448d7
md"""
## Makie backend for GridVisualize
"""

# ╔═╡ f02ef7d1-2940-42c2-bb51-92504ef0819f
import CairoMakie

# ╔═╡ 64f9ed75-4edf-4fec-ada0-9749e04280dc
scalarplot(grid2d3, fsin3, colormap = :hot, Plotter = CairoMakie, size = (500, 400), levels = 11)

# ╔═╡ 2682df92-5955-4b17-ae4f-8e99c5b17980
md"""
## 3D Tensor product grids
"""

# ╔═╡ 265fe6c7-d1cc-48a6-8295-f8f55acf677c
X3 = range(0.0, 10.1, length = 21)

# ╔═╡ b357395f-2a6e-476f-b008-02802c85a541
grid3d1 = simplexgrid(X3, X3, X3);

# ╔═╡ 38e2b4a8-2480-40e7-bde3-6d1775201aae
p3dg = GridVisualizer(dim = 3, size = (500, 400))

# ╔═╡ 8927fbba-f5ea-4ab3-8249-d9a3d5a46c3e
md"""
## 3D Function plot
"""

# ╔═╡ 86fb7e23-efc0-423a-b024-ac41adc446a1
func3 = map((x, y, z) -> sin(x / 2) * cos(y / 2) * z / 10, grid3d1);

# ╔═╡ ef1fde48-fe90-4714-ac86-614ae3451aa7
p3ds = GridVisualizer(dim = 3, resolution = (400, 400))

# ╔═╡ 04041481-0f03-41e1-a7de-1b3fd033c952
mean(x) = sum(x) / length(x)

# ╔═╡ 6cad87eb-1c59-4000-b688-a6f6d41f9413
md"""
## 3D Unstructured grids


The SimplexGridBuilder API supports creation of three-dimensional grids in way very similar to the 2D case. Just define points with three coordinates and planar (!) facets with at least three points to describe the geometry.

The backend for mesh generation in this case is the [TetGen](http://tetgen.org) mesh generator by Hang Si from WIAS Berlin and its Julia wrapper [TetGen.jl](https://github.com/JuliaGeometry/TetGen.jl) (maintained together with Simon Danisch).
"""

# ╔═╡ 75e9629b-fdd9-4d10-b09d-7c0f83ea7e40
import TetGen

# ╔═╡ fefc7587-8e25-4080-b934-90c0e1afc56a
builder3d = let

    b = SimplexGridBuilder(Generator = TetGen)

    p1 = point!(b, 0, 0, 0)
    p2 = point!(b, 1, 0, 0)
    p3 = point!(b, 1, 1, 0)
    p4 = point!(b, 0, 1, 0)
    p5 = point!(b, 0, 0, 1)
    p6 = point!(b, 1, 0, 1)
    p7 = point!(b, 1, 1, 1)
    p8 = point!(b, 0, 1, 1)

    facetregion!(b, 1)
    facet!(b, p1, p2, p3, p4)
    facetregion!(b, 2)
    facet!(b, p5, p6, p7, p8)
    facetregion!(b, 3)
    facet!(b, p1, p2, p6, p5)
    facetregion!(b, 4)
    facet!(b, p2, p3, p7, p6)
    facetregion!(b, 5)
    facet!(b, p3, p4, p8, p7)
    facetregion!(b, 6)
    facet!(b, p4, p1, p5, p8)


    hp1 = point!(b, 0.4, 0.4, 0.4)
    hp2 = point!(b, 0.6, 0.4, 0.4)
    hp3 = point!(b, 0.6, 0.6, 0.4)
    hp4 = point!(b, 0.4, 0.6, 0.4)
    hp5 = point!(b, 0.4, 0.4, 0.6)
    hp6 = point!(b, 0.6, 0.4, 0.6)
    hp7 = point!(b, 0.6, 0.6, 0.6)
    hp8 = point!(b, 0.4, 0.6, 0.6)

    facetregion!(b, 7)
    facet!(b, hp1, hp2, hp3, hp4)
    facet!(b, hp5, hp6, hp7, hp8)
    facet!(b, hp1, hp2, hp6, hp5)
    facet!(b, hp2, hp3, hp7, hp6)
    facet!(b, hp3, hp4, hp8, hp7)
    facet!(b, hp4, hp1, hp5, hp8)
    holepoint!(b, 0.5, 0.5, 0.5)

    b

end;

# ╔═╡ 065735f7-c799-4284-bd59-fe6383bb987c
grid3d2 = simplexgrid(builder3d, maxvolume = 0.0001)

# ╔═╡ 329992a0-e352-468b-af8b-0b190315fc61
gridplot(grid3d2, zplane = 0.1, azim = 20, elev = 20, linewidth = 0.5, outlinealpha = 0.3)

# ╔═╡ d2c71574-c168-4840-9892-1a3c49bc378d
md"""
## GMSH Grids
"""

# ╔═╡ f1ce470f-983b-46be-b0d6-993eefef8c2d
function gmsh_t4()
    gmsh.initialize()

    gmsh.model.add("t4")

    cm = 1.0e-2
    e1 = 4.5 * cm; e2 = 6 * cm / 2; e3 = 5 * cm / 2
    h1 = 5 * cm; h2 = 10 * cm; h3 = 5 * cm; h4 = 2 * cm; h5 = 4.5 * cm
    R1 = 1 * cm; R2 = 1.5 * cm; r = 1 * cm
    Lc1 = 0.01
    Lc2 = 0.003

    function hypot(a, b)
        return sqrt(a * a + b * b)
    end

    ccos = (-h5 * R1 + e2 * hypot(h5, hypot(e2, R1))) / (h5 * h5 + e2 * e2)
    ssin = sqrt(1 - ccos * ccos)

    factory = gmsh.model.geo
    factory.addPoint(-e1 - e2, 0, 0, Lc1, 1)
    factory.addPoint(-e1 - e2, h1, 0, Lc1, 2)
    factory.addPoint(-e3 - r, h1, 0, Lc2, 3)
    factory.addPoint(-e3 - r, h1 + r, 0, Lc2, 4)
    factory.addPoint(-e3, h1 + r, 0, Lc2, 5)
    factory.addPoint(-e3, h1 + h2, 0, Lc1, 6)
    factory.addPoint(e3, h1 + h2, 0, Lc1, 7)
    factory.addPoint(e3, h1 + r, 0, Lc2, 8)
    factory.addPoint(e3 + r, h1 + r, 0, Lc2, 9)
    factory.addPoint(e3 + r, h1, 0, Lc2, 10)
    factory.addPoint(e1 + e2, h1, 0, Lc1, 11)
    factory.addPoint(e1 + e2, 0, 0, Lc1, 12)
    factory.addPoint(e2, 0, 0, Lc1, 13)

    factory.addPoint(R1 / ssin, h5 + R1 * ccos, 0, Lc2, 14)
    factory.addPoint(0, h5, 0, Lc2, 15)
    factory.addPoint(-R1 / ssin, h5 + R1 * ccos, 0, Lc2, 16)
    factory.addPoint(-e2, 0.0, 0, Lc1, 17)

    factory.addPoint(-R2, h1 + h3, 0, Lc2, 18)
    factory.addPoint(-R2, h1 + h3 + h4, 0, Lc2, 19)
    factory.addPoint(0, h1 + h3 + h4, 0, Lc2, 20)
    factory.addPoint(R2, h1 + h3 + h4, 0, Lc2, 21)
    factory.addPoint(R2, h1 + h3, 0, Lc2, 22)
    factory.addPoint(0, h1 + h3, 0, Lc2, 23)

    factory.addPoint(0, h1 + h3 + h4 + R2, 0, Lc2, 24)
    factory.addPoint(0, h1 + h3 - R2, 0, Lc2, 25)

    factory.addLine(1, 17, 1)
    factory.addLine(17, 16, 2)

    factory.addCircleArc(14, 15, 16, 3)
    factory.addLine(14, 13, 4)
    factory.addLine(13, 12, 5)
    factory.addLine(12, 11, 6)
    factory.addLine(11, 10, 7)
    factory.addCircleArc(8, 9, 10, 8)
    factory.addLine(8, 7, 9)
    factory.addLine(7, 6, 10)
    factory.addLine(6, 5, 11)
    factory.addCircleArc(3, 4, 5, 12)
    factory.addLine(3, 2, 13)
    factory.addLine(2, 1, 14)
    factory.addLine(18, 19, 15)
    factory.addCircleArc(21, 20, 24, 16)
    factory.addCircleArc(24, 20, 19, 17)
    factory.addCircleArc(18, 23, 25, 18)
    factory.addCircleArc(25, 23, 22, 19)
    factory.addLine(21, 22, 20)

    factory.addCurveLoop([17, -15, 18, 19, -20, 16], 21)
    factory.addPlaneSurface([21], 22)
    factory.addCurveLoop([11, -12, 13, 14, 1, 2, -3, 4, 5, 6, 7, -8, 9, 10], 23)

    factory.addPlaneSurface([23, 21], 24)

    factory.synchronize()


    gmsh.model.setColor([(2, 22)], 127, 127, 127)
    gmsh.model.setColor([(2, 24)], 160, 32, 240)
    gmsh.model.setColor([(1, i) for i in 1:14], 255, 0, 0)
    gmsh.model.setColor([(1, i) for i in 15:20], 255, 255, 0)

    gmsh.model.mesh.generate(2)
    grid = ExtendableGrids.simplexgrid_from_gmsh(gmsh.model)
    gmsh.finalize()
    return grid
end;

# ╔═╡ d751ab29-e818-4511-a595-2b3fdf3888fb
gridplot(gmsh_t4())

# ╔═╡ b2e78cb6-aacc-44d8-8b52-84440040540c
md"""
## Grid Partitioning
"""

# ╔═╡ 3b286da6-d3ed-4a00-8b6b-11440395bf4c
import Metis

# ╔═╡ 8d5cc8db-04f9-4b9e-9e9a-2cb15cc7667b
gp = let

    b = SimplexGridBuilder(Generator = Triangulate; tol = 1.0e-10)

    #  Specify points
    p1 = point!(b, 0, 0)
    p2 = point!(b, 1, 0)
    p3 = point!(b, 1, 1)
    p4 = point!(b, 0, 0.7)

    # Specify outer boundary
    facetregion!(b, 1)
    facet!(b, p1, p2)
    facetregion!(b, 2)
    facet!(b, p2, p3)
    facetregion!(b, 3)
    facet!(b, p3, p4)
    facetregion!(b, 4)
    facet!(b, p1, p4)


    # Specify interior boundary
    facetregion!(b, 5)
    facet!(b, p1, p3)

    # Coarse elements in upper left region #1
    cellregion!(b, 1)
    maxvolume!(b, 0.1)
    regionpoint!(b, 0.1, 0.5)

    # Fine elements in lower right region #2
    cellregion!(b, 2)
    maxvolume!(b, 0.01)
    regionpoint!(b, 0.9, 0.5)

    # Hole
    hp1 = point!(b, 0.4, 0.1)
    hp2 = point!(b, 0.6, 0.1)
    hp3 = point!(b, 0.5, 0.3)
    holepoint!(b, 0.5, 0.2)
    facetregion!(b, 6)
    facet!(b, hp1, hp2)
    facet!(b, hp2, hp3)
    facet!(b, hp3, hp1)
    simplexgrid(b, maxvolume = 0.00025)

end;

# ╔═╡ 58cf8068-297b-4a72-8900-efbd363c80b2
gridplot(gp)

# ╔═╡ 601b29fa-170c-46e3-964b-cebe5aa6969e
gp1 = partition(gp, PlainMetisPartitioning(npart = 20))

# ╔═╡ 18b4e6d1-f35d-485a-b946-11561b5f3cc6
gridplot(gp1, cellcoloring = :pcolors)

# ╔═╡ 217fc42d-ec97-4ca2-bad8-8d9fa2b03ef8
gp2 = partition(gp, RecursiveMetisPartitioning(npart = 6))

# ╔═╡ 2a03aec9-2d22-4297-a4f8-0bb4be93d701
gridplot(gp2, cellcoloring = :pcolors)

# ╔═╡ a7965a6e-2e83-47eb-aee2-d366246a8637
html"""<hr>"""

# ╔═╡ 2659180e-b6a6-413c-a06c-3b4827f2bfa8
html"""<style> main { max-width: 95%; } </style>"""

# ╔═╡ fcce7bec-59a5-4125-976c-87223cf4ceea
import PlutoUI

# ╔═╡ 940b1996-fe9d-11ea-2fa4-8b72bee62b76
md"""
# WIAS-PDELib I: Grids & Visualization

__Jürgen Fuhrmann__, WIAS Berlin

with Dilara Abdel, Zeina Amer, Patricio Farrell, Yannis Hadjimichael, Patrick Jaap, Liam Johnen, Christian Merdon, Marieke Osewold, Daniel Runge, Laura Prieto Saavedra, Jan Philipp Thiele

__Trixi User and Developer Interaction (TRUDI) 2026__

Köln, 2026-03-18

$(PlutoUI.Resource("https://www.fv-berlin.de/fileadmin/user_upload/Institute/Logos/WIAS/WIAS_ohne.svg",:width=>150))
$(PlutoUI.Resource("https://wias-berlin.de/people/fuhrmann/blobs/pdelib-logo.png", :width => 150))
"""

# ╔═╡ a3844fda-5725-4d95-894b-051a5f6c2faa
md"""
f=$(@bind flevel PlutoUI.Slider(range(extrema(func3)...,length=20),default=mean(func3),show_value=true))

x=$(@bind xplane PlutoUI.Slider(X3[1]:0.1:X3[end],default=X3[end],show_value=true))

y=$(@bind yplane PlutoUI.Slider(X3[1]:0.1:X3[end],default=X3[end],show_value=true))

z=$(@bind zplane PlutoUI.Slider(X3[1]:0.1:X3[end],default=X3[end],show_value=true))

"""

# ╔═╡ f97d085c-e7bf-4561-8183-673912bdeab6
gridplot!(p3dg, grid3d1, zplanes = [zplane], yplanes = [yplane], xplanes = [xplane], resolution = (200, 200), show = true)

# ╔═╡ d73d18e7-bcf9-4cc1-9154-b70dc1ff5524
scalarplot!(p3ds, grid3d1, func3, zplanes = [zplane], yplanes = [yplane], xplanes = [xplane], levels = [flevel], colormap = :spring, resolution = (200, 200), show = true, levelalpha = 0.5, outlinealpha = 0.1)

# ╔═╡ 7ad541b1-f40f-4cdd-b7b5-b792a8e63d71
PlutoUI.TableOfContents(depth = 4, aside = true)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
ExtendableGrids = "cfc395e8-590f-11e8-1f13-43a2532b2fa8"
Gmsh = "705231aa-382f-11e9-3f0c-b7cb4346fdeb"
GridVisualize = "5eed8a63-0fb0-45eb-886d-8d5a387d12b8"
Metis = "2679e427-3c69-5b7f-982b-ece356f1e94b"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
PlutoVista = "646e1f28-b900-46d7-9d87-d554eb38a413"
SimplexGridFactory = "57bfcd06-606e-45d6-baf4-4ba06da0efd5"
TetGen = "c5d3f3f7-f850-59f6-8a2e-ffc6dc1317ea"
Triangulate = "f7e6ffb2-c36d-4f8f-a77e-16e897189344"

[compat]
CairoMakie = "~0.15.8"
ExtendableGrids = "~1.16.0"
Gmsh = "~0.3.1"
GridVisualize = "~1.17.0"
Metis = "~1.5.0"
PlutoUI = "~0.7.79"
PlutoVista = "~1.2.2"
SimplexGridFactory = "~2.6.1"
TetGen = "~2.0.1"
Triangulate = "~3.0.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.5"
manifest_format = "2.0"
project_hash = "fb275d0172762a380404c50407bee1a0a20e3e5f"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "35ea197a51ce46fcd01c4a44befce0578a1aaeca"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.5.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AdaptivePredicates]]
git-tree-sha1 = "7e651ea8d262d2d74ce75fdf47c4d63c07dba7a6"
uuid = "35492f91-a3bd-45ad-95db-fcad7dcfedb7"
version = "1.2.0"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e092fa223bf66a3c41f9c022bd074d916dc303e7"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "d57bd3762d308bded22c3b82d033bff85f6195c6"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.4.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Automa]]
deps = ["PrecompileTools", "SIMD", "TranscodingStreams"]
git-tree-sha1 = "a8f503e8e1a5f583fbef15a8440c8c7e32185df2"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "1.1.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "4126b08903b777c88edf1754288144a0492c05ad"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.8"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BaseDirs]]
git-tree-sha1 = "bca794632b8a9bbe159d56bf9e31c422671b35e0"
uuid = "18cc8868-cbac-4acf-b575-c8ff214dc66f"
version = "1.3.2"

[[deps.Bijections]]
git-tree-sha1 = "a2d308fcd4c2fb90e943cf9cd2fbfa9c32b69733"
uuid = "e2ed5e7c-b2de-5872-ae92-c73ca462fb04"
version = "0.2.2"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"
version = "1.11.0"

[[deps.CRlibm]]
deps = ["CRlibm_jll"]
git-tree-sha1 = "66188d9d103b92b6cd705214242e27f5737a1e5e"
uuid = "96374032-68de-5a5b-8d9e-752f78720389"
version = "1.0.2"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[deps.Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "71aa551c5c33f1a4415867fe06b7844faadb0ae9"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.1.1"

[[deps.CairoMakie]]
deps = ["CRC32c", "Cairo", "Cairo_jll", "Colors", "FileIO", "FreeType", "GeometryBasics", "LinearAlgebra", "Makie", "PrecompileTools"]
git-tree-sha1 = "5017d6849aff775febd36049f7d926a5fb6677ec"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.15.8"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "a21c5464519504e41e0cbc91f0188e8ca23d7440"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.5+1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "e4c6a16e77171a5f5e25e9646617ab1c276c5607"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON"]
git-tree-sha1 = "07da79661b919001e6863b81fc572497daa58349"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.ComputePipeline]]
deps = ["Observables", "Preferences"]
git-tree-sha1 = "3b4be73db165146d8a88e47924f464e55ab053cd"
uuid = "95dc2771-c249-4cd0-9c9f-1f3b4330693c"
version = "0.1.7"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "e357641bb3e0638d353c4b29ea0e40ea644066a6"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.3"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DelaunayTriangulation]]
deps = ["AdaptivePredicates", "EnumX", "ExactPredicates", "Random"]
git-tree-sha1 = "c55f5a9fd67bdbc8e089b5a3111fe4292986a8e8"
uuid = "927a84f5-c5f4-47a5-9785-b46e178433df"
version = "1.6.6"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "fbcc7610f6d8348428f722ecbe0e6cfe22e672c6"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.123"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.ElasticArrays]]
deps = ["Adapt"]
git-tree-sha1 = "75e5697f521c9ab89816d3abeea806dfc5afb967"
uuid = "fdbdab4c-e67f-52f5-8c3f-e7b388dad3d4"
version = "1.2.12"

[[deps.EnumX]]
git-tree-sha1 = "c49898e8438c828577f04b92fc9368c388ac783c"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.7"

[[deps.ExactPredicates]]
deps = ["IntervalArithmetic", "Random", "StaticArrays"]
git-tree-sha1 = "83231673ea4d3d6008ac74dc5079e77ab2209d8f"
uuid = "429591f6-91af-11e9-00e2-59fbe8cec110"
version = "2.2.9"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "27af30de8b5445644e8ffe3bcb0d72049c089cf1"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.7.3+0"

[[deps.ExtendableGrids]]
deps = ["AbstractTrees", "Bijections", "Compat", "Dates", "DocStringExtensions", "ElasticArrays", "Graphs", "InteractiveUtils", "LinearAlgebra", "Printf", "Random", "SparseArrays", "StaticArrays", "StatsBase", "UUIDs", "WriteVTK"]
git-tree-sha1 = "f0d353d1c84a367f96a5703919266ab3b6a03869"
uuid = "cfc395e8-590f-11e8-1f13-43a2532b2fa8"
version = "1.16.0"
weakdeps = ["Gmsh", "Metis", "TetGen", "Triangulate"]

    [deps.ExtendableGrids.extensions]
    ExtendableGridsGmshExt = "Gmsh"
    ExtendableGridsMetisExt = "Metis"
    ExtendableGridsTetGenExt = "TetGen"
    ExtendableGridsTriangulateExt = "Triangulate"

[[deps.Extents]]
git-tree-sha1 = "b309b36a9e02fe7be71270dd8c0fd873625332b4"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.6"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "66381d7059b5f3f6162f28831854008040a4e905"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "8.0.1+1"

[[deps.FFTA]]
deps = ["AbstractFFTs", "DocStringExtensions", "LinearAlgebra", "MuladdMacro", "Primes", "Random", "Reexport"]
git-tree-sha1 = "65e55303b72f4a567a51b174dd2c47496efeb95a"
uuid = "b86e33f2-c0db-4aa1-a6e0-ab43e668529e"
version = "0.3.1"

[[deps.FLTK_jll]]
deps = ["Artifacts", "Fontconfig_jll", "FreeType2_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "Xorg_libXft_jll", "Xorg_libXinerama_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "72a4842f93e734f378cf381dae2ca4542f019d23"
uuid = "4fce6fc7-ba6a-5f4c-898f-77e99806d6f8"
version = "1.3.8+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "6522cfb3b8fe97bec632252263057996cbd3de20"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.18.0"

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

    [deps.FileIO.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"

[[deps.FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport"]
git-tree-sha1 = "a1b2fbfe98503f15b665ed45b3d149e5d8895e4c"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.9.0"

    [deps.FilePaths.extensions]
    FilePathsGlobExt = "Glob"
    FilePathsURIParserExt = "URIParser"
    FilePathsURIsExt = "URIs"

    [deps.FilePaths.weakdeps]
    Glob = "c27321d9-0574-5035-807b-f59d2c89b15c"
    URIParser = "30578b45-9adc-5946-b283-645ec420af67"
    URIs = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "3bab2c5aa25e7840a4b065805c0cdfc01f3068d2"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.24"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "2f979084d1e13948a3352cf64a25df6bd3b4dca3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.16.0"
weakdeps = ["PDMats", "SparseArrays", "StaticArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStaticArraysExt = "StaticArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "907369da0f8e80728ab49c1c7e09327bf0d6d999"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.1"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "2c5512e11c791d1baed2049c5652441b28fc6a31"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.4+0"

[[deps.FreeTypeAbstraction]]
deps = ["BaseDirs", "ColorVectorSpace", "Colors", "FreeType", "GeometryBasics", "Mmap"]
git-tree-sha1 = "4ebb930ef4a43817991ba35db6317a05e59abd11"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.8"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.GLU_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg"]
git-tree-sha1 = "65af046f4221e27fb79b28b6ca89dd1d12bc5ec7"
uuid = "bd17208b-e95e-5925-bf81-e2f59b3e5c61"
version = "9.0.1+0"

[[deps.GMP_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "781609d7-10c4-51f6-84f2-b8444358ff6d"
version = "6.3.0+2"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "Extents", "IterTools", "LinearAlgebra", "PrecompileTools", "Random", "StaticArrays"]
git-tree-sha1 = "1f5a80f4ed9f5a4aada88fc2db456e637676414b"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.5.10"

    [deps.GeometryBasics.extensions]
    GeometryBasicsGeoInterfaceExt = "GeoInterface"

    [deps.GeometryBasics.weakdeps]
    GeoInterface = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Giflib_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6570366d757b50fabae9f4315ad74d2e40c0560a"
uuid = "59f7168a-df46-5410-90c8-f2779963d0ec"
version = "5.2.3+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "24f6def62397474a297bfcec22384101609142ed"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.86.3+0"

[[deps.Gmsh]]
deps = ["gmsh_jll"]
git-tree-sha1 = "6d815101e62722f4e323514c9fc704007d4da2e3"
uuid = "705231aa-382f-11e9-3f0c-b7cb4346fdeb"
version = "0.3.1"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "a641238db938fff9b2f60d08ed9030387daf428c"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.3"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "DataStructures", "Inflate", "LinearAlgebra", "Random", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "7eb45fe833a5b7c51cf6d89c5a841d5967e44be3"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.14.0"
weakdeps = ["Distributed", "SharedArrays"]

    [deps.Graphs.extensions]
    GraphsSharedArraysExt = "SharedArrays"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "93d5c27c8de51687a2c70ec0716e6e76f298416f"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.11.2"

[[deps.GridVisualize]]
deps = ["ColorSchemes", "Colors", "DocStringExtensions", "ElasticArrays", "ExtendableGrids", "GeometryBasics", "GridVisualizeTools", "HypertextLiteral", "Interpolations", "IntervalSets", "LinearAlgebra", "Observables", "OrderedCollections", "Printf", "StaticArrays"]
git-tree-sha1 = "27189023b9042cdfd4cddd12ad525b07da758a0c"
uuid = "5eed8a63-0fb0-45eb-886d-8d5a387d12b8"
version = "1.17.0"

    [deps.GridVisualize.extensions]
    GridVisualizeMakieExt = "Makie"
    GridVisualizeMeshCatExt = "MeshCat"
    GridVisualizePlotsExt = "Plots"
    GridVisualizePlutoVistaExt = "PlutoVista"
    GridVisualizePyPlotExt = "PyPlot"
    GridVisualizePythonPlotExt = "PythonPlot"
    GridVisualizeVTKViewExt = "VTKView"

    [deps.GridVisualize.weakdeps]
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    MeshCat = "283c5d60-a78f-5afe-a0af-af636b173e11"
    Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
    PlutoVista = "646e1f28-b900-46d7-9d87-d554eb38a413"
    PyPlot = "d330b81b-6aea-500a-939a-2ce795aea3ee"
    PythonPlot = "274fc56d-3b97-40fa-a1cd-1b4a50311bf9"
    Triangulate = "f7e6ffb2-c36d-4f8f-a77e-16e897189344"
    VTKView = "955f2c64-5fd0-11e9-0ad0-3332e913311a"

[[deps.GridVisualizeTools]]
deps = ["ColorSchemes", "Colors", "DocStringExtensions", "StaticArrays", "StaticArraysCore"]
git-tree-sha1 = "7cfc079442c7bd2904bbfa32b76975054b06a639"
uuid = "5573ae12-3b76-41d9-b48c-81d0b6e61cc5"
version = "3.0.2"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HDF5_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "LibCURL_jll", "Libdl", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "OpenSSL_jll", "TOML", "Zlib_jll", "libaec_jll"]
git-tree-sha1 = "e94f84da9af7ce9c6be049e9067e511e17ff89ec"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.14.6+0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "92f65c4d78ce8cdbb6b68daf88889950b0a99d11"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.12.1+0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "68c173f4f449de5b438ee67ed0c9c748dc31a2ec"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.28"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "e12629406c6c4442539436581041d372d69c55ba"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.12"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "8c193230235bbcee22c8066b0374f63b5683c2d3"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.5"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs", "WebP"]
git-tree-sha1 = "696144904b76e1ca433b886b4e7edd067d76cbf7"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.9"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "2a81c3897be6fbcde0802a0ebe6796d0562f63ec"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.10"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "dcc8d0cd653e55213df9b75ebc6fe4a8d3254c65"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.2.2+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "4c1acff2dc6b6967e7e750633c50bc3b8d83e617"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.3"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "65d505fa4c0d7072990d659ef3fc086eb6da8208"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.16.2"

    [deps.Interpolations.extensions]
    InterpolationsForwardDiffExt = "ForwardDiff"
    InterpolationsUnitfulExt = "Unitful"

    [deps.Interpolations.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.IntervalArithmetic]]
deps = ["CRlibm", "MacroTools", "OpenBLASConsistentFPCSR_jll", "Printf", "Random", "RoundingEmulator"]
git-tree-sha1 = "02b61501dbe6da3b927cc25dacd7ce32390ee970"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "1.0.2"

    [deps.IntervalArithmetic.extensions]
    IntervalArithmeticArblibExt = "Arblib"
    IntervalArithmeticDiffRulesExt = "DiffRules"
    IntervalArithmeticForwardDiffExt = "ForwardDiff"
    IntervalArithmeticIntervalSetsExt = "IntervalSets"
    IntervalArithmeticLinearAlgebraExt = "LinearAlgebra"
    IntervalArithmeticRecipesBaseExt = "RecipesBase"
    IntervalArithmeticSparseArraysExt = "SparseArrays"

    [deps.IntervalArithmetic.weakdeps]
    Arblib = "fb37089c-8514-4489-9461-98f9c8763369"
    DiffRules = "b552c78f-8df3-52c6-915a-8e097449b14b"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.IntervalSets]]
git-tree-sha1 = "d966f85b3b7a8e49d034d27a189e9a4874b4391a"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.13"

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

    [deps.IntervalSets.weakdeps]
    Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "b3ad4a0255688dcb895a52fafbaae3023b588a90"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.4.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "9496de8fb52c224a2e3f9ff403947674517317d9"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.6"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6893345fd6658c8e475d40155789f4860ac3b21"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.4+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTA", "Interpolations", "StatsBase"]
git-tree-sha1 = "4260cfc991b8885bf747801fb60dd4503250e478"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.11"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "97bbca976196f2a1eb9607131cb108c69ec3f8a6"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.41.3+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "f04133fe05eff1667d2054c53d59f9122383fe05"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.2+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d0205286d9eceadc518742860bf23f703779a3d6"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.41.3+0"

[[deps.LightXML]]
deps = ["Libdl", "XML2_jll"]
git-tree-sha1 = "aa971a09f0f1fe92fe772713a564aa48abe510df"
uuid = "9c8b4983-aa76-5018-a973-4c85ecc9e179"
version = "0.9.3"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LinearElasticity_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "71e8ee0f9fe0e86a8f8c7f28361e5118eab2f93f"
uuid = "18c40d15-f7cd-5a6d-bc92-87468d86c5db"
version = "5.0.0+0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.METIS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2eefa8baa858871ae7770c98c3c2a7e46daba5b4"
uuid = "d00139f3-1899-568f-a2f0-47f597d42d70"
version = "5.1.3+0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MMG_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "LinearElasticity_jll", "Pkg", "SCOTCH_jll"]
git-tree-sha1 = "70a59df96945782bb0d43b56d0fbfdf1ce2e4729"
uuid = "86086c02-e288-5929-a127-40944b0018b7"
version = "5.6.0+0"

[[deps.MPICH_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "9341048b9f723f2ae2a72a5269ac2f15f80534dc"
uuid = "7cb0a576-ebde-5e09-9194-50597f1243b4"
version = "4.3.2+0"

[[deps.MPIPreferences]]
deps = ["Libdl", "Preferences"]
git-tree-sha1 = "c105fe467859e7f6e9a852cb15cb4301126fac07"
uuid = "3da0fdf6-3ccc-4f1b-acd9-58baa6c99267"
version = "0.1.11"

[[deps.MPItrampoline_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "e214f2a20bdd64c04cd3e4ff62d3c9be7e969a59"
uuid = "f1f71cc9-e9ae-5b93-9b94-4fe0e1ad3748"
version = "5.5.4+0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Makie]]
deps = ["Animations", "Base64", "CRC32c", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "ComputePipeline", "Contour", "Dates", "DelaunayTriangulation", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG_jll", "FileIO", "FilePaths", "FixedPointNumbers", "Format", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageBase", "ImageIO", "InteractiveUtils", "Interpolations", "IntervalSets", "InverseFunctions", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "Markdown", "MathTeXEngine", "Observables", "OffsetArrays", "PNGFiles", "Packing", "Pkg", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Scratch", "ShaderAbstractions", "Showoff", "SignedDistanceFields", "SparseArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun", "Unitful"]
git-tree-sha1 = "d1b974f376c24dad02c873e951c5cd4e351cd7c2"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.24.8"

    [deps.Makie.extensions]
    MakieDynamicQuantitiesExt = "DynamicQuantities"

    [deps.Makie.weakdeps]
    DynamicQuantities = "06fc5a27-2a28-4c7c-a15d-362465fb6821"

[[deps.MappedArrays]]
git-tree-sha1 = "0ee4497a4e80dbd29c058fcee6493f5219556f40"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.3"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "UnicodeFun"]
git-tree-sha1 = "7eb8cdaa6f0e8081616367c10b31b9d9b34bb02a"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.6.7"

[[deps.MeshIO]]
deps = ["ColorTypes", "FileIO", "GeometryBasics", "Printf"]
git-tree-sha1 = "c009236e222df68e554c7ce5c720e4a33cc0c23f"
uuid = "7269a6da-0436-5bbc-96c2-40638cbb6118"
version = "0.5.3"

[[deps.Metis]]
deps = ["CEnum", "LinearAlgebra", "METIS_jll", "SparseArrays"]
git-tree-sha1 = "54aca4fd53d39dcd2c3f1bef367b6921e8178628"
uuid = "2679e427-3c69-5b7f-982b-ece356f1e94b"
version = "1.5.0"

    [deps.Metis.extensions]
    MetisGraphs = "Graphs"
    MetisLightGraphs = "LightGraphs"
    MetisSimpleWeightedGraphs = ["SimpleWeightedGraphs", "Graphs"]

    [deps.Metis.weakdeps]
    Graphs = "86223c79-3864-5bf0-83f7-82e725a168b6"
    LightGraphs = "093fc24a-ae57-5d10-9952-331d41423f4d"
    SimpleWeightedGraphs = "47aef6b3-ad0c-573a-a1e2-d07658019622"

[[deps.MicrosoftMPI_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bc95bf4149bf535c09602e3acdf950d9b4376227"
uuid = "9237b28f-5490-5468-be7b-bb81f5f5e6cf"
version = "10.1.4+3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.MuladdMacro]]
git-tree-sha1 = "cac9cc5499c25554cba55cd3c30543cff5ca4fab"
uuid = "46d2c3a1-f734-5fdb-9937-b9b9aeba4221"
version = "0.2.4"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.OCCT_jll]]
deps = ["Artifacts", "FreeType2_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "Xorg_libXft_jll", "Xorg_libXinerama_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "f130ae166e244c5791d211d482d153360c34a94a"
uuid = "baad4e97-8daa-5946-aac2-2edac59d34e1"
version = "7.9.2+0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "117432e406b5c023f665fa73dc26e79ec3630151"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.17.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLASConsistentFPCSR_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f2b3b9e52a5eb6a3434c8cca67ad2dde011194f4"
uuid = "6cdc7f73-28fd-5e50-80fb-958a8875b1af"
version = "0.3.30+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "97db9e07fe2091882c765380ef58ec553074e9c7"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.3"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "df9b7c88c2e7a2e77146223c526bf9e236d5f450"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.4.4+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenMPI_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML", "Zlib_jll"]
git-tree-sha1 = "ab6596a9d8236041dcd59b5b69316f28a8753592"
uuid = "fe0851c0-eecd-5654-98d4-656369965a5c"
version = "5.0.9+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e2bb57a313a74b8104064b7efd01406c0a50d2ff"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.6.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.44.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "e4cff168707d441cd6bf3ff7e4832bdf34278e4a"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.37"
weakdeps = ["StatsBase"]

    [deps.PDMats.extensions]
    StatsBaseExt = "StatsBase"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "cf181f0b1e6a18dfeb0ee8acc4a9d1672499626c"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.4"

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "bc5bf2ea3d5351edf285a06b0016788a121ce92c"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.5.1"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0662b083e11420952f2e62e17eddae7fc07d5997"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.57.0+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "db76b1ecd5e9715f3d043cec13b2ec93ce015d53"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.44.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "26ca162858917496748aad52bb5d3be4d26a228a"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.4"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "3ac7038a98ef6977d44adeadc73cc6f596c08109"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.79"

[[deps.PlutoVista]]
deps = ["AbstractPlutoDingetjes", "ColorSchemes", "Colors", "DocStringExtensions", "GridVisualizeTools", "HypertextLiteral", "UUIDs"]
git-tree-sha1 = "d64875384d29bd97f331de1d9eca1e37c3f72d14"
uuid = "646e1f28-b900-46d7-9d87-d554eb38a413"
version = "1.2.2"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "07a921781cab75691315adc645096ed5e370cb77"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.3"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "25cdd1d20cd005b52fc12cb6be3f75faaf59bb9b"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.7"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "fbb92c6c56b34e1a2c4c36058f68f332bec840e7"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "472daaa816895cb7aee81658d4e7aec901fa1106"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9da16da70037ba9d701192e27befedefb91ec284"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.2"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "5b3d50eb374cea306873b371d3f8d3915a018f0b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.9.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[deps.SCOTCH_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "7110b749766853054ce8a2afaa73325d72d32129"
uuid = "a8d0f55d-b80e-548d-aff6-1a04c175f0f9"
version = "6.1.3+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "e24dc23107d426a096d3eae6c165b921e74c18e4"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.7.2"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.ShaderAbstractions]]
deps = ["ColorTypes", "FixedPointNumbers", "GeometryBasics", "LinearAlgebra", "Observables", "StaticArrays"]
git-tree-sha1 = "818554664a2e01fc3784becb2eb3a82326a604b6"
uuid = "65257c39-d410-5151-9873-9b3e5be5013e"
version = "0.5.0"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SignedDistanceFields]]
deps = ["Statistics"]
git-tree-sha1 = "3949ad92e1c9d2ff0cd4a1317d5ecbba682f4b92"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.1"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "be8eeac05ec97d379347584fa9fe2f5f76795bcb"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.5"

[[deps.SimplexGridFactory]]
deps = ["DocStringExtensions", "ElasticArrays", "ExtendableGrids", "FileIO", "GridVisualize", "LinearAlgebra", "MeshIO", "Printf"]
git-tree-sha1 = "d9200b7cd7b9b029695c1b1a751c4429b1f9bd98"
uuid = "57bfcd06-606e-45d6-baf4-4ba06da0efd5"
version = "2.6.1"
weakdeps = ["TetGen", "Triangulate"]

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "0494aed9501e7fb65daba895fb7fd57cc38bc743"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.5"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "5acc6a41b3082920f79ca3c759acbcecf18a8d78"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.7.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "4f96c596b8c8258cc7d3b19797854d368f243ddc"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.4"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "be1cf4eb0ac528d96f5115b4ed80c26a8d8ae621"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.2"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "246a8bb2e6667f832eea063c3a56aef96429a3db"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.18"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "aceda6f4e598d331548e04cc6b2124a6148138e3"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.10"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "91f091a8716a6bb38417a6e6f274602a19aaa685"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.5.2"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "a2c37d815bf00575332b7bd0389f771cb7987214"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.2"

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

    [deps.StructArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "28145feabf717c5d65c1d5e09747ee7b1ff3ed13"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.6.3"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TetGen]]
deps = ["DocStringExtensions", "GeometryBasics", "LinearAlgebra", "Printf", "StaticArrays", "TetGen_jll"]
git-tree-sha1 = "ef2dc4d446a66dd5a84f36428d4bb51595ad229f"
uuid = "c5d3f3f7-f850-59f6-8a2e-ffc6dc1317ea"
version = "2.0.1"

[[deps.TetGen_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9ceedd691bce040e24126a56354f20d71554a495"
uuid = "b47fdcd6-d2c1-58e9-bbba-c1cee8d8c179"
version = "1.5.3+0"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "PrecompileTools", "ProgressMeter", "SIMD", "UUIDs"]
git-tree-sha1 = "08c10bc34f4e7743f530793d0985bf3c254e193d"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.11.8"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Triangle_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "bc4c7bb314cd0ac8bb36152f637fcd764e95748e"
uuid = "5639c1d2-226c-5e70-8d55-b3095415a16a"
version = "1.6.3+0"

[[deps.Triangulate]]
deps = ["DocStringExtensions", "Triangle_jll"]
git-tree-sha1 = "fd348d50587253dff8efb7a34b997effccf44427"
uuid = "f7e6ffb2-c36d-4f8f-a77e-16e897189344"
version = "3.0.1"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "57e1b2c9de4bd6f40ecb9de4ac1797b81970d008"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.28.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    ForwardDiffExt = "ForwardDiff"
    InverseFunctionsUnitfulExt = "InverseFunctions"
    LatexifyExt = ["Latexify", "LaTeXStrings"]
    NaNMathExt = "NaNMath"
    PrintfExt = "Printf"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"
    LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
    Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
    NaNMath = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
    Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.VTKBase]]
git-tree-sha1 = "c2d0db3ef09f1942d08ea455a9e252594be5f3b6"
uuid = "4004b06d-e244-455f-a6ce-a5f9919cc534"
version = "1.0.1"

[[deps.WebP]]
deps = ["CEnum", "ColorTypes", "FileIO", "FixedPointNumbers", "ImageCore", "libwebp_jll"]
git-tree-sha1 = "aa1ca3c47f119fbdae8770c29820e5e6119b83f2"
uuid = "e3aaa7dc-3e4b-44e0-be63-ffb868ccd7c1"
version = "0.1.3"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "248a7031b3da79a127f14e5dc5f417e26f9f6db7"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.1.0"

[[deps.WriteVTK]]
deps = ["Base64", "CodecZlib", "FillArrays", "LightXML", "TranscodingStreams", "VTKBase"]
git-tree-sha1 = "a329e0b6310244173690d6a4dfc6d1141f9b9370"
uuid = "64499a7a-5c06-52f2-abe2-ccb03c286192"
version = "1.21.2"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "5c959b708667b34cb758e8d7c6f8e69b94c32deb"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.15.1+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "9cce64c0fdd1960b597ba7ecda2950b5ed957438"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.2+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "808090ede1d41644447dd5cbafced4731c56bd2f"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.13+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "1a4a26870bf1e5d26cd585e38038d399d7e65706"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.8+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXft_jll]]
deps = ["Artifacts", "Fontconfig_jll", "JLLWrappers", "Libdl", "Xorg_libXrender_jll"]
git-tree-sha1 = "d893c27836da7986c3248997a2a9535e5e4d8a95"
uuid = "2c808117-e144-5220-80d1-69d4eaa9352c"
version = "2.3.9+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "0ba01bc7396896a4ace8aab67db31403c71628f4"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.7+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "4909eb8f1cbf6bd4b1c30dd18b2ead9019ef2fad"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.18.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.gmsh_jll]]
deps = ["Artifacts", "Cairo_jll", "CompilerSupportLibraries_jll", "FLTK_jll", "FreeType2_jll", "GLU_jll", "GMP_jll", "HDF5_jll", "JLLWrappers", "JpegTurbo_jll", "LLVMOpenMP_jll", "Libdl", "Libglvnd_jll", "METIS_jll", "MMG_jll", "OCCT_jll", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "Xorg_libXft_jll", "Xorg_libXinerama_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fdca60995cf57a52a571b3eb13284703a0b79f93"
uuid = "630162c2-fc9b-58b3-9910-8442a8a132e6"
version = "4.15.0+0"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaec_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1aa23f01927b2dac46db77a56b31088feee0a491"
uuid = "477f73a3-ac25-53e9-8cc3-50b2fa2566f0"
version = "1.1.4+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "371cc681c00a3ccc3fbc5c0fb91f58ba9bec1ecf"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.13.1+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "125eedcb0a4a0bba65b657251ce1d27c8714e9d6"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.4+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdrm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "63aac0bcb0b582e11bad965cef4a689905456c03"
uuid = "8e53e030-5e6c-5a89-a30b-be5b7263a166"
version = "2.4.125+1"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e015f211ebb898c8180887012b938f3851e719ac"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.55+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "libpng_jll"]
git-tree-sha1 = "c1733e347283df07689d71d61e14be986e49e47a"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.5+0"

[[deps.libva_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "libdrm_jll"]
git-tree-sha1 = "7dbf96baae3310fe2fa0df0ccbb3c6288d5816c9"
uuid = "9a156e7d-b971-5f62-b2c9-67348b8fb97c"
version = "2.23.0+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.libwebp_jll]]
deps = ["Artifacts", "Giflib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libglvnd_jll", "Libtiff_jll", "libpng_jll"]
git-tree-sha1 = "4e4282c4d846e11dce56d74fa8040130b7a95cb3"
uuid = "c5f90fcd-3b7e-5836-afba-fc50a0988cb2"
version = "1.6.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"
"""

# ╔═╡ Cell order:
# ╟─940b1996-fe9d-11ea-2fa4-8b72bee62b76
# ╟─5f9c4f4c-91a1-4c92-abc1-781e4c8898fb
# ╟─4e6e769b-0878-40fb-bf12-1aaf9fb34b93
# ╟─505433d7-e34e-4cd1-94ef-bf7b67f19f0a
# ╟─47103e8f-a4ff-46ed-a632-572a2e194a50
# ╠═93a0c45e-d6a3-415a-a82c-e4f7e2a09d22
# ╠═614b2d29-5bae-47c7-a349-26f3796c12bb
# ╠═4622a1fc-fda7-4211-9cc0-4eb1a1584aa6
# ╟─bb3bd2ed-6ca4-4c87-a72b-b5a3b2ec43d2
# ╠═9e6dfe3f-0c1d-4225-bc15-362004bd8e27
# ╠═155b8646-faca-4b9f-bce6-47fb29139e6f
# ╠═36498181-6d35-40c6-8a46-93a4070ab87b
# ╟─135d7271-3a4f-4275-9578-08d90561b503
# ╠═4faca367-1f09-4cbf-811d-f249d4d397f8
# ╠═5520b8c0-0874-4790-a956-224e6c43d9cf
# ╟─13aef2a1-5745-4fe5-9659-6b7c0e7267fc
# ╟─2d392a98-bf32-4145-ae93-a8e218367277
# ╠═f6dce047-b283-4e92-855d-c85b11dc474e
# ╠═88247350-aa6c-4876-82c3-3534036d5702
# ╠═74641f73-2efe-4df8-bebb-ed97e77d869e
# ╟─14fb7977-93cf-4f74-a8ec-b6ee25dbdf86
# ╠═369d96ec-eb77-4729-863c-2e8f3f823084
# ╠═2d5cb9e1-2d14-415e-b792-c3124901011d
# ╟─b1f903b3-29d7-4909-b7e2-8ef3528c9965
# ╠═19125aed-5c46-4968-bcfc-0b469628b68e
# ╠═dd7d8e17-f825-4047-9386-d9e2bfd0a48d
# ╟─3d7db57f-2864-4984-a979-609e1d838a9f
# ╠═806e23f0-cbe5-48aa-bf10-98cfac4d826a
# ╠═4cf840e6-86c5-4af1-a780-6fc78b60716b
# ╠═151cc4b8-c5ed-4f5e-8d5f-2f708c9c7fae
# ╠═71af99ab-612a-4821-8e9c-efc8766e2e3e
# ╟─6dfa1d73-8baa-4589-b2e6-547834c9e444
# ╠═f9599246-8238-432c-a315-300d74abfa2c
# ╠═ba5af291-fb16-4f21-8b74-664284bf7bd9
# ╟─dfbaacea-4cb4-4147-9f1b-4424d7a7e89b
# ╟─7d1698dd-3bb7-4b38-9c6d-a88652749eee
# ╠═81249f7c-abdf-43cc-b57f-2915b09da009
# ╟─0765641a-8ed9-4579-bd9b-90bb02a55792
# ╠═794558f5-48cc-49dc-93bd-c596560848f6
# ╠═884a11a2-15cf-40fc-a1ca-66ea23c6094e
# ╠═8f7d958e-5dc5-4324-af21-ad829d7d77eb
# ╟─4a289b23-46b9-495d-b19c-42b3da71b242
# ╠═b12838f0-fe9c-11ea-2939-155ed907322d
# ╟─d5d8a1d6-fe9d-11ea-0fd8-df6e81492cb5
# ╠═aae2e82a-fe9c-11ea-0427-593f8d2c7746
# ╟─1ae86964-fe9e-11ea-303b-65bb128384a5
# ╠═511b26c6-f920-11ea-1228-51c3750f495c
# ╠═59a6c8b5-25aa-47aa-9489-a803672013df
# ╟─4c99c40f-cf93-4cba-bef1-0c4ffcbf6833
# ╠═a375c23f-6b8c-4b2c-a8b5-d38e6b5a8f6d
# ╠═4dfb2e0f-3e3a-4053-8a76-765546e96992
# ╟─afeca036-2014-4ab4-b128-8b8851a448d7
# ╠═f02ef7d1-2940-42c2-bb51-92504ef0819f
# ╠═64f9ed75-4edf-4fec-ada0-9749e04280dc
# ╟─2682df92-5955-4b17-ae4f-8e99c5b17980
# ╠═265fe6c7-d1cc-48a6-8295-f8f55acf677c
# ╠═b357395f-2a6e-476f-b008-02802c85a541
# ╠═38e2b4a8-2480-40e7-bde3-6d1775201aae
# ╠═f97d085c-e7bf-4561-8183-673912bdeab6
# ╟─8927fbba-f5ea-4ab3-8249-d9a3d5a46c3e
# ╠═86fb7e23-efc0-423a-b024-ac41adc446a1
# ╠═ef1fde48-fe90-4714-ac86-614ae3451aa7
# ╟─a3844fda-5725-4d95-894b-051a5f6c2faa
# ╠═d73d18e7-bcf9-4cc1-9154-b70dc1ff5524
# ╠═04041481-0f03-41e1-a7de-1b3fd033c952
# ╟─6cad87eb-1c59-4000-b688-a6f6d41f9413
# ╠═75e9629b-fdd9-4d10-b09d-7c0f83ea7e40
# ╠═fefc7587-8e25-4080-b934-90c0e1afc56a
# ╠═065735f7-c799-4284-bd59-fe6383bb987c
# ╠═329992a0-e352-468b-af8b-0b190315fc61
# ╟─d2c71574-c168-4840-9892-1a3c49bc378d
# ╠═d8a97d1e-f689-4aca-8bdc-0e08d160ba2a
# ╠═f1ce470f-983b-46be-b0d6-993eefef8c2d
# ╠═d751ab29-e818-4511-a595-2b3fdf3888fb
# ╟─b2e78cb6-aacc-44d8-8b52-84440040540c
# ╠═3b286da6-d3ed-4a00-8b6b-11440395bf4c
# ╠═391f4cd3-fd8e-48df-b1f8-d806607dc619
# ╠═8d5cc8db-04f9-4b9e-9e9a-2cb15cc7667b
# ╠═58cf8068-297b-4a72-8900-efbd363c80b2
# ╠═601b29fa-170c-46e3-964b-cebe5aa6969e
# ╠═18b4e6d1-f35d-485a-b946-11561b5f3cc6
# ╠═217fc42d-ec97-4ca2-bad8-8d9fa2b03ef8
# ╠═2a03aec9-2d22-4297-a4f8-0bb4be93d701
# ╟─a7965a6e-2e83-47eb-aee2-d366246a8637
# ╟─2659180e-b6a6-413c-a06c-3b4827f2bfa8
# ╠═fcce7bec-59a5-4125-976c-87223cf4ceea
# ╠═7ad541b1-f40f-4cdd-b7b5-b792a8e63d71
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
