module Mipmaps

include("MyImageResize.jl")
using Images
using ColorTypes
using StaticArrays
using Interpolations
using LinearAlgebra

struct Mipmap
    pyramid :: Vector{Image} where Image <: AbstractArray{C} where C <: Colorant
end

function interpolator(img; boundary_condition = eltype(img)(.9, .8, .1))
    inds = map(s -> range(0, 1, length=s), size(img))
    #return LinearInterpolation(inds, img; extrapolation_bc = oneunit(eltype(img)))
    #return LinearInterpolation(inds, img; extrapolation_bc = zero(eltype(img)))
    return LinearInterpolation(inds, img; extrapolation_bc = Flat())
    # return LinearInterpolation(inds, img; extrapolation_bc = boundary_condition)
    # return CubicSplineInterpolation(inds, img; extrapolation_bc = zero(eltype(img)))
end

# For 1D and 2D data, use a good antialiasing kernel
_restrict(img::Union{AbstractVector,AbstractMatrix}) = MyImageResize.resize(img, 0.5)
_restrict(img) = restrict(img)

function mipmap(img :: AbstractArray{C}; min_size = 10) where C <: Colorant
    img = float(img)
    p1 = interpolator(img)
    pyramid = Vector{typeof(p1)}()
    push!(pyramid, p1)

    while minimum(size(img)) > min_size
        img = _restrict(img)
        push!(pyramid, interpolator(img))
    end

    return Mipmap(pyramid)
end

"""
    (M::Mipmap)(i, j, level)

Return pixel `(i,j)` at (possibly fractional) pyramid `level` using linear
interpolation between pyramid levels. The interpolation type inside each pyramid
level is defined by the [`interpolator`](@ref) method.
"""
function (M::Mipmap)(i::Real, j::Real, level::Real)
    l0 = floor(Int, level)
    l1 = ceil(Int, level)

    inrange(ind, range) = max(min(ind, maximum(range)), minimum(range))

    img0 = M.pyramid[inrange(l0, 1:end)]
    img1 = M.pyramid[inrange(l1, 1:end)]

    p0 = img0(i,j)
    p1 = img1(i,j)

    α = modf(level)[1]

    return (1-α)*p0 + α*p1
end

function Base.size(M::Mipmap, level::Real)
    inrange(ind, range) = max(min(ind, maximum(range)), minimum(range))
    l0 = floor(Int, level)
    l1 = ceil(Int, level)
    img0 = M.pyramid[inrange(l0, 1:end)]
    img1 = M.pyramid[inrange(l1, 1:end)]
    α = modf(level)[1]
    return (1-α).*size(img0) .+ α.*size(img1)
end

function eigval_to_miplevel(x)
    # This function converts a jacobian value to the required mipmap level
    # that should be accessed to avoid aliasing. This conversion is a bit tricky.
    # See the details [[id:fb38281c-80d0-40b3-89f6-90e29e70d0f1][here]].
    scale_factor = min(1, abs(1 + x))
    reduction_factor = 1 / scale_factor
    miplevel = 1 + log2(reduction_factor)
end

struct UninvertibleException <: Exception
    Jacobian :: AbstractArray{T,2} where T
end

function Base.showerror(io::IO, e::UninvertibleException)
    print(io, "Deformation is too strong: vector field is not invertible. J = $(e.Jacobian).")
end

function jacobian_to_anisotropic_miplevels(J::AbstractArray{T,2}) where T
    E = eigen((J + J')/2)
    if minimum(E.values) < -1
        throw(UninvertibleException(J))
    end
    l = eigval_to_miplevel.(E.values)
    @assert l[1] >= l[2]
    return l, E.vectors[:,1]
end

"""
    (M::Mipmap)(i, j, J::Matrix)

Return pixel `(i,j)` using anisotropic mipmap sampling, taking into account the
Jacobian matrix `J`.
"""
function (M::Mipmap)(i::Real, j::Real, J::AbstractArray{T,2}) where T
    l, dir = jacobian_to_anisotropic_miplevels(J)
    δ = 2^(maximum(l) - 1) / 2^(minimum(l) - 1) / 2
    a = 15
    v = dir ./ size(M, minimum(l))
    c = M.pyramid[1] |> eltype |> zero
    for t in range(-δ, δ, length=a)
        px = [i,j] .+ t * v
        c += M(px..., minimum(l))
    end
    return c / a
end

function (M::Mipmap)(px::SVector{N}, l) where N
    return M(px..., l)
end

end
