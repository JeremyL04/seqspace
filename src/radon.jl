module Radon

using LinearAlgebra
using Statistics
using Roots: find_zero, rrule

#=
TODO Lots of optimization to be done here. Add more arguments to the functions and have them return 
functions to be called in the loss function (i.e. only generate Hamiltonian once).
=#

function Square_CDFRadon(r,θ)
    α = mod(θ + π/4, π/2) + π/4
    C₁ = cot(α)
    C₂ = csc(α)
    A₁ = (-C₂*r+C₁+1)
    A₂ = (C₂*r+C₁+1)
    A₃ = (-C₂*r-C₁+1)
    A₄ = (C₂*r-C₁+1)
    D = 2*(abs(cos(α)-sin(α))+abs(cos(α)+sin(α))) # Missing a factor of 2 here
    return (-abs(A₁)*A₁+abs(A₂)*A₂+abs(A₃)*A₃-abs(A₄)*A₄)/(4*D*C₁*C₂) + 0.5
end

function Square_InvCDFRadon(y,θ)
    g(x) = Square_CDFRadon(x,θ) - y
    return find_zero(g,0)
end

function ϵ_Square(Z)
    N = size(Z,2)
    Nₛ = 6 # Number of slices
    Θ = (0+0.001:π/(Nₛ):π-0.01)
    proj = [[dot( point , [cos(θ),sin(θ)] ) for point ∈ eachcol(Z)] for θ ∈ Θ]
    Y = collect(0:1/(N-1):1)
    InvCDFs = [[Square_InvCDFRadon(y,θ) for y ∈ Y] for θ ∈ Θ]
    return mean([mean( (InvCDFs[i] .- sort(proj[i])).^2 ) for i ∈ 1:length(Θ)].^(1/2)) # Notice this is W₂
end

end