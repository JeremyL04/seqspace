module ML

using Random, Statistics
using LinearAlgebra, Flux, Zygote

import Base:
    length, reverse, iterate

export model, train!, validate, preprocess, update_dimension

# NOTE: all data is assumed to be dᵢ x N shaped here!
#       this is to allow for fast matrix multiplication through the network
#       it might be worthwhile to keep both memory layouts stored...

# ------------------------------------------------------------------------
# globals

σ₀(x)  = x + (sqrt(x^2 +1)-1)/2

# ------------------------------------------------------------------------
# types
# Multi Activation Function Layers

struct CustomActivationLayer
    linear::Dense
    activations::Vector{Function}
end

function (m::CustomActivationLayer)(x)
    z = m.linear(x)
    collect(hcat([m.activations[i].(z[i, :]) for i in axes(z,1)]...)')
end

# Iterator
"""
    struct LayerIterator
        width     :: Array{Int}
        dropout   :: Set{Int}
        normalize :: Set{Int}
        σᵢ        :: Function
        σₒ        :: Function
        σ         :: Function
    end

An iterator used to generate dense latent layers within a neural network.
`width` denotes the widths of each layer; the length of this array immediately determines the depth.
`dropout` denotes the layers, as given by `width` that are followed by a dropout layer.
`normalize` denotes the layers, as given by `width` that are followed by a batch normalization layer.
`σᵢ`, `σₒ`, `σ` is the activation energy on the first, last, and intermediate layers respectively.
"""
struct LayerIterator
    width     :: Array{Int}
    dropout   :: Set{Int}
    normalize :: Set{Int}
    σᵢ        :: Function # activation on input layers (first)
    σₒ        :: Union{Function, Vector{<:Function}} # activation on output layers (last / latent space)
    σ         :: Function # activation on interoir layers
end

length(it::LayerIterator)  = length(it.width) + length(it.dropout) + length(it.normalize)
reverse(it::LayerIterator) = LayerIterator(
                                reverse(it.width),
                                Set(length(it.width) - i - 1 for i in it.dropout),
                                Set(length(it.width) - i - 1 for i in it.normalize),
                                it.σ,
                                it.σᵢ, # intentional -> want to make output = input
                                it.σ,
                             )

function iterate(it::LayerIterator)
    w₁ = it.width[1]
    w₂ = it.width[2]
    isa(it.σᵢ, Vector{<:Function}) ? f = CustomActivationLayer(Dense(w₁, w₂),it.σᵢ) : f = Dense(w₁, w₂, it.σᵢ)

    return f, (
        index     = 2,
        dropout   = 1 ∈ it.dropout,
        normalize = 1 ∈ it.normalize,
    )
end

function iterate(it::LayerIterator, state)
    return if state.dropout
               Dropout(0.5), (
                   index     = state.index,
                   dropout   = false,
                   normalize = state.normalize,
               )
           elseif state.normalize
               BatchNorm(it.width[state.index]), (
                   index     = state.index,
                   dropout   = false,
                   normalize = false,
               )
           elseif state.index < length(it.width)
                w₁ = it.width[state.index]
                w₂ = it.width[state.index+1]

                i  = state.index+1
                    if i == length(it.width)
                        isa(it.σₒ, Vector{<:Function}) ? f = CustomActivationLayer(Dense(w₁, w₂),it.σₒ) : f = Dense(w₁, w₂, it.σₒ)
                    else
                        f  = Dense(w₁, w₂, it.σ)
                    end

                f, (
                     index     = i,
                     dropout   = (i-1) ∈ it.dropout,
                     normalize = (i-1) ∈ it.normalize,
                )
           else
               nothing
           end
end

# ------------------------------------------------------------------------
# functions

"""
    model(dᵢ, dₒ; Ws=Int[], normalizes=Int[], dropouts=Int[], σ=elu)

Initialize an autoencoding neural network with input dimension `dᵢ` and latent layers `dₒ`.
`Ws` specifies both the width and depth of the encoder layer - the width of each layer is given as an entry in the array while the length specifies the depth.
`normalizes` and `dropouts` denote which layers are followed by batch normalization and dropout specifically.
The decoder layer is given the mirror symmetric architecture.
"""
function model(dᵢ, dₒ; Ws=Int[], normalizes=Int[], dropouts=Int[], interior_activation=celu, exterior_activation=celu, initial_activation = relu)
    # check for obvious errors here
    length(dropouts)   > 0 && length(Ws) < maximum(dropouts)   ≤ 0 && error("invalid dropout layer position")
    length(normalizes) > 0 && length(Ws) < maximum(normalizes) ≤ 0 && error("invalid normalization layer position")
    isa(exterior_activation, Vector{<:Function}) && (length(exterior_activation) != dₒ) ? error("invalid number of exterior activation functions") : nothing

    layers = LayerIterator(
                [dᵢ; Ws; dₒ],
                Set(dropouts),
                Set(normalizes),
                initial_activation,
                exterior_activation,
                interior_activation,
             )

    F   = Chain(layers...)
    F¯¹ = Chain(reverse(layers)...)
    𝕀   = Chain(F, F¯¹)

    return (
        pullback=F,
        pushforward=F¯¹,
        identity=𝕀
    )
end

"""
    update_dimension(model, dₒ; ϵ = 1e-6)

Add a colection of new neurons in the encoding layer to encode in the encoding layer to increase dimensions to `dₒ`.
Model weights for the initial dimensions are kept the same.
"""
function update_dimension(model, dₒ; ϵ = 1e-6)
    F, F¯¹, 𝕀 = model

    lᵢ = F[end]
    lₒ = F¯¹[1]

    Wᵢ, bᵢ = Flux.params(lᵢ)
    Wₒ, bₒ = Flux.params(lₒ)

    size(Wᵢ,1) == dₒ && return nothing
    size(Wᵢ,1) >  dₒ && error("can not reduce dimensionality of model") 

    δ  = dₒ - size(Wᵢ, 1)

    W̄ᵢ = vcat(Wᵢ, ϵ*randn(δ, size(Wᵢ,2)))
    b̄ᵢ = vcat(bᵢ, ϵ*randn(δ))
    l̄ᵢ = Dense(W̄ᵢ, b̄ᵢ, lᵢ.σ)

    W̄ₒ = hcat(Wₒ, ϵ*randn(size(Wₒ,1), δ))
    b̄ₒ = bₒ
    l̄ₒ = Dense(W̄ₒ, b̄ₒ, lₒ.σ)

    F   = Chain( (i < length(F) ? f : l̄ᵢ for (i,f) ∈ enumerate(F))...)
    F¯¹ = Chain( (i > 1 ? f : l̄ₒ for (i,f) ∈ enumerate(F¯¹))...)
    𝕀   = Chain(F, F¯¹)

    return (
        pullback=F,
        pushforward=F¯¹,
        identity=𝕀
    )
end

# data batching
"""
    batch(data, n)

Randomly partition `data` into groups of size `n`.
"""
function batch(data, n)
    N = size(data,2)

    lo(i) = (i-1)*n + 1
    hi(i) = min((i)*n, N)

    ι = randperm(N)

    return (data[:,ι[lo(i):hi(i)]] for i in 1:ceil(Int, N/n)), 
           (ι[lo(i):hi(i)] for i in 1:ceil(Int, N/n))
end


"""
    validate(data, len)

Reserve `len` samples from `data` during training process to allow for model validation.
"""
function validate(data, len)
    ι = randperm(size(data,2))
    return (
        valid = data[:,ι[1:len]],
        train = data[:,ι[len+1:end]],
    ),(
        valid = ι[1:len],
        train = ι[len+1:end],
    )
end

function noop(epoch) end

# data training
"""
    train!(model, data, index, loss; B=64, η=1e-3, N=100, log=noop)

Trains autoencoder `model` on `data` by minimizing `loss`.
`index` stores the underlying indices of data used for training.
Will mutate the underlying parameters of `model`.
Optional parameters include:
  1. `B` denotes the batch size to be used.
  2. `N` denotes the number of epochs.
  3. `η` denotes the learning rate.
"""
function train!(model, data, index, loss; B=64, η=1e-3, N=100, log=noop)
    Θ   = Flux.params(model.identity)
    opt = NADAM(η)

    for n ∈ 1:N
        X, I = batch(data, B)
        for (i,x) ∈ zip(I,X)
            E, backpropagate = pullback(Θ) do
                loss(x, index[i], false)
            end

            isnan(E) && @goto done

            ∇Θ = backpropagate(1f0)
            Flux.Optimise.update!(opt, Θ, ∇Θ)
        end

        log(n)
    end
    @label done
end

# ------------------------------------------------------------------------
# tests

end
