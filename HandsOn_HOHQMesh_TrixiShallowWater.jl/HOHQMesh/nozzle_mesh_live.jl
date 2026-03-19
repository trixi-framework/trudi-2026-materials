
using HOHQMesh, GLMakie

# Make a new project
p = newProject("nozzle", "out")

# Reset params (optional)

# Set the background with bounds [TOP, LEFT, BOTTOM, RIGHT]
# and desired number of elements in each direction

# This background grid strategy assumes an `OUTER_BOUNDARY` is present
addBackgroundGrid!(p, [0.5, 0.5, 0.0])

# This background grid strategy works without an `OUTER_BOUNADRY` present
# addBackgroundGrid!(p, [6.0, -3.0, -1.0, 12.0], [24, 20, 0])

# Create the outer boundary for the nozzle (must be counter-clockwise)
# Helpful to first the spline data, then build the curves
nozzTop = newSplineCurve("nozzleTop", joinpath(@__DIR__, "NozzleTop.txt"))
nozzBot = new("nozzleBot", joinpath(@__DIR__, "NozzleBottom.txt"))

addCurveToOuterBoundary!(p, nozzTop)

ptTop = nozzTop["SPLINE_DATA"][end, 2:4]
ptBot = nozzBot["SPLINE_DATA"][1, 2:4]

inlet = new("inlet", ptTop, ptBot)
add!(p, inlet)

add!(p, nozzBot)

ptBotEnd = nozzBot["SPLINE_DATA"][end, 2:4]

ext1 = newEndPointsLineCurve("ext1", ptBotEnd, [1.0, 0.5, 0.0])
add!(p, ext1)

ext2 = newEndPointsLineCurve("ext2", [1.0, 0.5, 0.0], [1.0, 0.0, 0.0])
add!(p, ext2)

Bottom = new("Bottom", [1.0, 0.0, 0.0], [9.0, 0.0, 0.0])
add!(p, Bottom)

Right = new("Right", [9.0, 0.0, 0.0], [9.0, 5.0, 0.0])
add!(p, Right)

Top = new("Top", [9.0, 5.0, 0.0], [nozzTop["SPLINE_DATA"][1, 2], 5.0, 0.0])
add!(p, Top)

Left = new("Top", [nozzTop["SPLINE_DATA"][1, 2], 5.0, 0.0], nozzTop["SPLINE_DATA"][1, 2:4])
add!(p, Left)

# Add refinement regions (if desired)

# Plot project command
plotProject!(p, MODEL + GRID + REFINEMENTS)

# Generate the mesh
generate_mesh(p)

# If desired, one can rename the Bottom boundary to make it symmetric.
# The name `:symmetry` is special and informs HOHQMesh that it should
# reflect the mesh over the boundary with this name.
#
# renameCurve(p, "Bottom", ":symmetry")
#
# and regenerate the mesh.
