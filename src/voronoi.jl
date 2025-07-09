module Voronoi

using LinearAlgebra
using DelaunayTriangulation

import ChainRulesCore: rrule, NoTangent

export areas, voronoi_areas, sort_points_cw, corners_and_edges

# 2D cross product
∧(u, v) = u[1] * v[2] - u[2] * v[1]


const square = Float32.([-1.0 -1.0 1.0 1.0; -1.0 1.0 1.0 -1.0])

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


# For use with DelaunayTriangulate.jl
function PureDelaunayTri(X) 
    tri = triangulate(unique(eachcol(X)))
    # Sort the vertex indices within each triangle
    sorted_triangles = [sort(collect(f)) for f in tri.triangles if -1 ∉ f]
    # Sort the columns lexicographically to ensure consistent output
    triangulation = hcat(sorted_triangles...)
    triangulation = triangulation[:, sortperm(eachcol(triangulation))]
    return triangulation
end

function areas(x, boundary_corners=nothing)
    q = isnothing(boundary_corners) || isempty(boundary_corners) ? x : hcat(boundary_corners, x)
    triangulation = PureDelaunayTri(hcat(unique(eachcol(q))...))

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
    triangulation = PureDelaunayTri(hcat(unique(eachcol(q))...))

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

function corners_and_edges(widths::AbstractVector{<:Real}, n_points_per_edge::Int=0)
    # Generate all corners of the box
    dims = [[0.0, w] for w in widths]
    corners = [collect(corner) for corner in Iterators.product(dims...)]

    # If no additional points are requested, return only the corners
    if n_points_per_edge == 0
        return Float32.(hcat(corners...))
    end

    # Generate edges by connecting corners differing in one coordinate
    edges = []
    for corner in corners
        for i in axes(widths,1)
            if corner[i] == 0.0
                p2 = copy(corner)
                p2[i] = widths[i]
                push!(edges, (corner, p2))
            end
        end
    end

    # Generate points along each edge
    ts = range(0, 1, length=n_points_per_edge + 2)
    points = [ (1 - t) .* p1 .+ t .* p2 for (p1, p2) in edges, t in ts ]

    # Remove duplicate points
    unique_points = unique(points)

    # Convert to Float32 array
    return Float32.(hcat(unique_points...))
end

affine(simplex) = vcat(simplex, ones(Float32, 1, size(simplex,2)))
volume(simplex) = det(affine(simplex))


function volumes(x)
    d = size(x,1)
    q = hcat(boundary_corners(d), x)

    simplices = DelaunayTri(q)

    Ω = [ volume(hcat((q[:,i] for i in simplex)...)) for simplex in eachcol(simplices) ]
    s = sign.(Ω)

    return s.*Ω
end

function rrule(::typeof(volumes), x)
    d = size(x,1)
    b = boundary_corners(d)
    q = hcat(b, x)

    v₀ = size(b,2)
    simplices = DelaunayTri(q)

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

"""
New code to test real voronoi reularization
"""

function DelaunayTri(X)
    # Perform Delaunay triangulation
    tri = triangulate(unique(eachcol(X)))

    # Filter and lexicographically sort valid triangles
    sorted_triangles = sort(
        [sort(collect(f)) for f in tri.triangles],
        by = x -> (x[1], x[2], x[3])
    )

    return sorted_triangles::Vector{Vector{Int64}}
end

# Voronoi functions
function voronoi_verts(X::AbstractMatrix{Float32}, Delaunay::Vector{Vector{Int64}})
    c = mean(X; dims=2)  # c is size (size(X,1), 1)

    Delaunay_ignored = Zygote.ignore() do 
        Delaunay
    end

    vertices = [
        let tri = Delaunay_ignored[i]
            if -1 ∈ tri
                idx = filter(!=(-1), tri)  # remove any -1
                p, q = X[:, idx[1]], X[:, idx[2]]  # differentiable indexing into X
                pₘ = 0.5f0 * (p + q)
                δ  = q - p
                dᵢ = Float32[-δ[2], δ[1]]  # perpendicular vector
                d̂ᵢ = (dᵢ / norm(dᵢ)) * sign(dot(pₘ - c, dᵢ))  # outward direction
                pₘ + 10f0 * d̂ᵢ          # the final point
            else
                circumcenter(X[:, tri])   # indexing X[:, tri] is differentiable w.r.t. X
            end
        end
        for i in eachindex(Delaunay_ignored)
    ]

    return vertices
end

function voronoi_plan(X::AbstractMatrix{Float32}, Delaunay::Vector{Vector{Int64}})
    N = size(X, 2)
    voronoi_plan = Vector{Vector{Int}}(undef, N)  # Preallocate outer array
    for i in 1:N
        voronoi_plan[i] = Vector{Int}()  # Initialize each inner array
    end

    for (k, triangle) ∈ enumerate(Delaunay)
        for i in triangle
            if i != -1
                push!(voronoi_plan[i], k)
            end
        end
    end

    return voronoi_plan
end

function circumcenter(Tri::AbstractMatrix{T}) where T <: AbstractFloat
    x₁, y₁ = Tri[1, 1], Tri[2, 1]
    x₂, y₂ = Tri[1, 2], Tri[2, 2]
    x₃, y₃ = Tri[1, 3], Tri[2, 3]

    det = 2 * ((x₁ - x₂) * (y₂ - y₃) - (y₁ - y₂) * (x₂ - x₃))
    if iszero(det)
        return (NaN, NaN)
    end

    x₁²_y₁² = x₁^2 + y₁^2
    x₂²_y₂² = x₂^2 + y₂^2
    x₃²_y₃² = x₃^2 + y₃^2

    ux = (x₁²_y₁² * (y₂ - y₃) + x₂²_y₂² * (y₃ - y₁) + x₃²_y₃² * (y₁ - y₂)) / det
    uy = (x₁²_y₁² * (x₃ - x₂) + x₂²_y₂² * (x₁ - x₃) + x₃²_y₃² * (x₂ - x₁)) / det

    return [ux, uy]
end

function sort_points_ccw(T::Vector{Vector{Float32}})
    # Compute centroid
    cx = mean(p[1] for p in T)
    cy = mean(p[2] for p in T)

    # Compute angles relative to centroid
    angles = [atan(p[2] - cy, p[1] - cx) for p in T]

    # Get sorted indices
    sorted_indices = sortperm(angles)

    # Return points sorted in CCW order
    return T[sorted_indices]
end

function sort_points_cw(T::AbstractMatrix{<:Real})
    @assert size(T, 1) == 2 "Input must be a 2×N matrix."
    cx, cy = mean(T, dims=2)
    angles = atan.(T[2, :] .- cy, T[1, :] .- cx)
    sorted_indices = sortperm(angles, rev=true)
    return T[:, sorted_indices]
end


function polygon_area(points::Vector{<:AbstractVector{Float32}})
    N = length(points)
    if N < 3
        return error("Yikes!")  # A polygon must have at least 3 points
    end

    # Compute area directly using the shoelace formula without extra arrays
    area = 0.0f0
    for i in 1:N
        p₁ = points[i]
        p₂ = points[mod1(i + 1, N)]
        area += p₁[1] * p₂[2] - p₂[1] * p₁[2]
    end

    return 0.5f0 * area
end

function polygon_area(points::AbstractMatrix{Float32})
    x = points[1, :]
    y = points[2, :]
    N = length(x)

    area = 0.5f0 * sum(x .* y[[2:N; 1]] .- x[[2:N; 1]] .* y)
    return area
end

"""
 Clipping functions; laregly taken from https://github.com/mhdadk/sutherland-hodgman
"""


# Clipping functions; laregly translated from https://github.com/mhdadk/sutherland-hodgman
function is_inside(p1, p2, q)
    R = (p2[1] - p1[1]) * (q[2] - p1[2]) - (p2[2] - p1[2]) * (q[1] - p1[1])
    return R <= 0
end

# compute_intersection function
function compute_intersection(p₁, p₂, p₃, p₄)
    x₁, y₁ = p₁
    x₂, y₂ = p₂
    x₃, y₃ = p₃
    x₄, y₄ = p₄

    denom = (x₁ - x₂) * (y₃ - y₄) - (y₁ - y₂) * (x₃ - x₄)

    if denom == 0.0
        # Return midpoint when lines are parallel
        return ((x₁ + x₂) / 2.0, (y₁ + y₂) / 2.0)
    else
        # Precompute common terms
        A = x₁ * y₂ - y₁ * x₂
        B = x₃ * y₄ - y₃ * x₄
        C = x₁ - x₂
        D = y₁ - y₂
        E = x₃ - x₄
        F = y₃ - y₄

        x = (A * E - C * B) / denom
        y = (A * F - D * B) / denom
        return [x, y]
    end
end

# clip function
function clip(subject_polygon::Vector{<:AbstractVector}, clipping_polygon)
    final_polygon = hcat(subject_polygon...)  # Convert vector of vectors to a 2D array
    num_clip_vertices = size(clipping_polygon, 2)

    for i in 1:num_clip_vertices
        input_list = final_polygon
        final_polygon = Array{Float32}(undef, 2, 0)  # Initialize empty 2D array

        c_edge_start = clipping_polygon[:, i == 1 ? end : i - 1]
        c_edge_end = clipping_polygon[:, i]

        num_subject_vertices = size(input_list, 2)
        if num_subject_vertices == 0
            break
        end

        s = input_list[:, end]

        for j in 1:num_subject_vertices
            e = input_list[:, j]
            if is_inside(c_edge_start, c_edge_end, e)
                if !is_inside(c_edge_start, c_edge_end, s)
                    intersection = compute_intersection(s, e, c_edge_start, c_edge_end)
                  
                    final_polygon = hcat(final_polygon, intersection)
                end
                final_polygon = hcat(final_polygon, e)
            elseif is_inside(c_edge_start, c_edge_end, s)
                intersection = compute_intersection(s, e, c_edge_start, c_edge_end)
                final_polygon = hcat(final_polygon, intersection)
            end
            s = e
        end
    end

    # Return final polygon in 2D array form
    return final_polygon
end

function any_outside(testcell, boundary)
    # Find bounding box [min_x, max_x] × [min_y, max_y]
    min_x = minimum(boundary[1, :])
    max_x = maximum(boundary[1, :])
    min_y = minimum(boundary[2, :])
    max_y = maximum(boundary[2, :])
    
    # Check if any vertex is outside the box
    any(p -> p[1] < min_x || p[1] > max_x || p[2] < min_y || p[2] > max_y, testcell)
end

function voronoi_areas(X::AbstractMatrix{Float32}, boundary::AbstractMatrix{Float32})
    # Wrap any non-differentiable steps (including boundary) in Zygote.ignore
    DT, VTP, boundary_box = Zygote.ignore() do
        dt = DelaunayTri(X)
        vtp = voronoi_plan(X, dt)
        # boundary is also captured here as boundary_box
        return dt, vtp, boundary
    end

    # This remains differentiable w.r.t. X
    centers = voronoi_verts(X, DT)
    # The rest of the logic that depends on X remains AD-friendly
    areas = [
        let
            cell_indices = VTP[i]
            cell_verts   = centers[cell_indices]
            sorted_cell_verts = sort_points_ccw(cell_verts)

            do_clip = any_outside(sorted_cell_verts, boundary_box)
            clipped_cell = do_clip ? clip(sorted_cell_verts, boundary_box) : sorted_cell_verts

            polygon_area(clipped_cell)
        end
        for i in eachindex(VTP)
    ]

    return areas::Vector{Float32}
end