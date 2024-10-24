module Voronoi

using LinearAlgebra
using Quickhull

import ChainRulesCore: rrule, NoTangent

export boundary, square_points, Delaunay_Triangulate, areas, rrule, corners

# 2D cross product
∧(u, v) = u[1] * v[2] - u[2] * v[1]

"""
    boundary(d)

Returns the boundary vertices of a unit cube in `d` dimensions.
"""
function boundary_corners(d)
    d == 2 && return Float32.([[-1;-1] [-1;+1] [+1;+1] [+1;-1]])

    b = zeros(Float32, d, 2^d)
    for i in 0:(2^d-1)
        for n in 0:(d-1)
            b[n+1,i+1] = (((i >> n) & 1) == 1) ? +1 : -1
        end
    end
    return b
end


function square_points(N, half_side = 1.0)
    N = Int(N)
    b = zeros(Float32, 2, N)
    side_length = half_side * 2
    
    for i in 0:N-1
        position = i * 8 / N
        b[:, i + 1] = if position < 2
            [-half_side + side_length * position / 2, half_side]
        elseif position < 4
            [half_side, half_side - side_length * (position - 2) / 2]
        elseif position < 6
            [half_side - side_length * (position - 4) / 2, -half_side]
        else
            [-half_side, -half_side + side_length * (position - 6) / 2]
        end
    end
    
    return b
end


 # For use with Quickhull
function Delaunay_Triangulate(q)
    tri = delaunay(unique(eachcol(q)))
    faces = sort(sort.(collect(facets(tri))))
    return hcat([faces[i] for i ∈ eachindex(faces)]...) |> collect
end

# For use with DelaunayTriangulate.jl
# function Delaunay_Triangulate(q)
#     tri = triangulate(unique(eachcol(q)))
    
#     # Sort the vertex indices within each triangle
#     sorted_triangles = [sort(collect(f)) for f in tri.triangles if -1 ∉ f]
    
#     # Sort the columns lexicographically to ensure consistent output
#     triangulation = hcat(sorted_triangles...)
#     triangulation = triangulation[:, sortperm(eachcol(triangulation))]
    
#     return triangulation
# end

function areas(x, boundary_corners=nothing)
    q = isnothing(boundary_corners) || isempty(boundary_corners) ? x : hcat(boundary_corners, x)
    triangulation = Delaunay_Triangulate(q)

    a = 0.5 * [
        let
            q[1, t[1]] * (q[2, t[2]] - q[2, t[3]]) +
            q[1, t[2]] * (q[2, t[3]] - q[2, t[1]]) +
            q[1, t[3]] * (q[2, t[1]] - q[2, t[2]])
        end for t in eachcol(triangulation)
    ]
    s = sign.(a)

    return s .* a
end

function rrule(::typeof(areas), x, boundary_corners)
    q = isnothing(boundary_corners) || isempty(boundary_corners) ? x : hcat(boundary_corners, x)
    triangulation = Delaunay_Triangulate(q)

    a = 0.5 * [
        let
            q[1, t[1]] * (q[2, t[2]] - q[2, t[3]]) +
            q[1, t[2]] * (q[2, t[3]] - q[2, t[1]]) +
            q[1, t[3]] * (q[2, t[1]] - q[2, t[2]])
        end for t in eachcol(triangulation)
    ]
    s = sign.(a)

    return s .* a, (∂a) -> let
        ∂x = zeros(size(x))
        NB = isnothing(boundary_corners) || isempty(boundary_corners) ? 0 : size(boundary_corners, 2)


        for (i, t) in enumerate(eachcol(triangulation))
            if t[1] > NB
                ∂x[1, t[1] - NB] += (q[2, t[2]] - q[2, t[3]]) * ∂a[i] * s[i]
                ∂x[2, t[1] - NB] -= (q[1, t[2]] - q[1, t[3]]) * ∂a[i] * s[i]
            end
            if t[2] > NB
                ∂x[1, t[2] - NB] += (q[2, t[3]] - q[2, t[1]]) * ∂a[i] * s[i]
                ∂x[2, t[2] - NB] -= (q[1, t[3]] - q[1, t[1]]) * ∂a[i] * s[i]
            end
            if t[3] > NB
                ∂x[1, t[3] - NB] += (q[2, t[1]] - q[2, t[2]]) * ∂a[i] * s[i]
                ∂x[2, t[3] - NB] -= (q[1, t[1]] - q[1, t[2]]) * ∂a[i] * s[i]
            end
        end

        (NoTangent(), 0.5 * ∂x, NoTangent())
    end
end


"""
    corners(widths)

Returns the corners of a hypercube with side lengths given by `widths`.
"""
function corners(widths::AbstractVector{<:Real})
    dims = [[0.0, w] for w in widths]
    corners = collect(Iterators.product(dims...))
    return Float32.(hcat([collect(corner) for corner in corners]...))
end

affine(simplex) = vcat(simplex, ones(Float32, 1, size(simplex,2)))
volume(simplex) = det(affine(simplex))


function volumes(x)
    d = size(x,1)
    q = hcat(boundary_corners(d), x)

    simplices = Delaunay_Triangulate(q)

    Ω = [ volume(hcat((q[:,i] for i in simplex)...)) for simplex in eachcol(simplices) ]
    s = sign.(Ω)

    return s.*Ω
end

function rrule(::typeof(volumes), x)
    d = size(x,1)
    b = boundary_corners(d)
    q = hcat(b, x)

    v₀ = size(b,2)
    simplices = Delaunay_Triangulate(q)

    Ω = [ volume(hcat((q[:,i] for i in simplex)...)) for simplex in eachcol(simplices) ]
    s = sign.(Ω)

    Z = factorial(d)
    return (s.*Ω) ./ Z, function(∂Ω)
        ∂x = zeros(Float32,size(x))

        for (n,simplex) in enumerate(eachcol(simplices))
            ω = s[n]*Ω[n]*∂Ω[n]
            try
                Q = inv(affine(hcat((q[:,i] for i in simplex)...)))

                for (i,v) in enumerate(simplex)
                    if v > v₀
                        for j in 1:d
                            ∂x[j,v-v₀] += ω*Q[i,j]
                        end
                    end
                end
            catch error
                # XXX: do we need to do something more intelligent here?
                if isa(error, LinearAlgebra.SingularException)
                    continue
                else
                    throw(error)
                end
            end
        end

        return (NoTangent(), ∂x ./ Z)
    end
end

end