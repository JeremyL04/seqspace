module Voronoi

using LinearAlgebra
using DelaunayTriangulation

import ChainRulesCore: rrule, NoTangent

∧(x,y) = x[1,:].*y[2,:] .- x[2,:].*y[1,:]

"""
    boundary(d)

Returns the boundary vertices of a unit cube in `d` dimensions.
"""
function boundary(d)
    d == 2 && return Float32.([[-1;-1] [-1;+1] [+1;+1] [+1;-1]])

    b = zeros(Float32, d, 2^d)
    for i in 0:(2^d-1)
        for n in 0:(d-1)
            b[n+1,i+1] = (((i >> n) & 1) == 1) ? +1 : -1
        end
    end
    return b
end


function boundary_points(N, half_side = 1.0)
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


const NB = 4



 # For use with Quickhull
# function Delaunay_Triangulate(q)
#     tri = delaunay(q)
#     faces = collect(facets(tri))
#     return hcat([faces[i] for i ∈ eachindex(faces)]...) |> collect
# end

function Delaunay_Triangulate(q)
    tri = triangulate(q)
    
    # Sort the vertex indices within each triangle
    sorted_triangles = [sort(collect(f)) for f in tri.triangles if -1 ∉ f]
    
    # Sort the columns lexicographically to ensure consistent output
    triangulation = hcat(sorted_triangles...)
    triangulation = triangulation[:, sortperm(eachcol(triangulation))]
    
    return triangulation
end

function areas(x)
    q = hcat(boundary_points(NB), x)
    triangulation = Delaunay_Triangulate(q)

    a = 0.5*[
        let
            q[1,t[1]]*(q[2,t[2]]-q[2,t[3]]) + 
            q[1,t[2]]*(q[2,t[3]]-q[2,t[1]]) + 
            q[1,t[3]]*(q[2,t[1]]-q[2,t[2]])
        end for t in eachcol(triangulation) 
    ]
    s = sign.(a)

    return s.*a
end

function rrule(::typeof(areas), x)
    q = hcat(boundary_points(NB), x)
    triangulation = Delaunay_Triangulate(q)

    a = 0.5*[ 
        let
            q[1,t[1]]*(q[2,t[2]]-q[2,t[3]]) +
            q[1,t[2]]*(q[2,t[3]]-q[2,t[1]]) +
            q[1,t[3]]*(q[2,t[1]]-q[2,t[2]])
        end for t in eachcol(triangulation)
    ]
    s = sign.(a)

    return s.*a, (∂a) -> let
        ∂x = zeros(size(x))

        for (i,t) in enumerate(eachcol(triangulation))
            if t[1] > NB
                ∂x[1,t[1]-NB] += (q[2,t[2]]-q[2,t[3]])*∂a[i]*s[i]
                ∂x[2,t[1]-NB] -= (q[1,t[2]]-q[1,t[3]])*∂a[i]*s[i]
            end
            if t[2] > NB
                ∂x[1,t[2]-NB] += (q[2,t[3]]-q[2,t[1]])*∂a[i]*s[i]
                ∂x[2,t[2]-NB] -= (q[1,t[3]]-q[1,t[1]])*∂a[i]*s[i]
            end
            if t[3] > NB
                ∂x[1,t[3]-NB] += (q[2,t[1]]-q[2,t[2]])*∂a[i]*s[i]
                ∂x[2,t[3]-NB] -= (q[1,t[1]]-q[1,t[2]])*∂a[i]*s[i]
            end
        end

        (NoTangent(), 0.5*∂x)
    end

end

end