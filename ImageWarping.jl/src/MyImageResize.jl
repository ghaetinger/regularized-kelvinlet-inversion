__precompile__();

module MyImageResize

using DSP
using Images
using Interpolations

export resize

_sampleloc(M,N) = clamp.(((0:M-1).+0.5)/M * N .+ 1 .- 0.5,1,N)

function _resample(itp::Interpolations.Flag, x::Array{T,N}, M::NTuple{N,Int}) where T where N
    y = interpolate(x, itp)
    y(_sampleloc.(M,size(x))...)
end

function _resample(itp::Interpolations.Flag, x::Array{T,N}, M::Int) where T where N
    _resample(itp, x, ntuple(λ->M, N))
end

resample_nearest(x, param) = _resample(OnCell() |> Constant |> BSpline, x, param)
resample_cubic(x, param) = _resample(OnCell() |> Flat |> Cubic |> BSpline, x, param)

function lowpass_lanczos(s::Vector, cutoff::U; lobes::Real = 3) where U <: Real
    if cutoff >= 1; return s end
    lobedist = 1/cutoff
    P = round(Int,lobes*lobedist)
    designmethod = FIRWindow(DSP.Windows.lanczos(2P+1))
    responsetype = Lowpass(cutoff)
    h = digitalfilter(responsetype, designmethod)
    t = similar(s)
    Images.imfilter!(t, s, centered(h), "reflect")
end

function lowpass_lanczos(s::Array{T,N}, cutoff::NTuple{N,U}; params...) where T where N where U <: Real
    u = deepcopy(s)
    for dim in 1:ndims(u)
        u .= mapslices(u; dims = dim) do slice
            lowpass_lanczos(slice, cutoff[dim]; params...)
        end
    end
    u
end

function lowpass_lanczos(s::Array{T,N}, cutoff::U; params...) where T where N where U <: Real
    lowpass_lanczos(s, ntuple(λ->cutoff, N); params...)
end

nop_prefilter(s, cutoff) = s

"""
    resize(s::Array{T,N}, siz::NTuple{N,Int}; prefilter = lowpass_lanczos, resampler = resample_cubic) where T where N

Resize (prefilter + resample) data in `s` to size `siz`.
"""
function resize(s::Array{T,N}, siz::NTuple{N,Int}; prefilter = lowpass_lanczos, resampler = resample_cubic) where T where N
    rates   = siz ./ size(s)
    cutoffs = min.(one(eltype(rates)), rates)
    g = prefilter(s, cutoffs)
    resampler(g, siz)
end

import Base.round
round(T::Type{Int64}, val::Int64, ::RoundingMode{:NearestTiesAway}) = val

"""
    resize(s::Array, rate::U; params...) where U <: Real

Resize (prefilter + resample) data in `s` to a percentage of `size(s)` given by rate.

# Examples

    u = resize(s, 0.5) # Reduce in half the number of samples
"""
function resize(s::Array, rate::U; params...) where U <: AbstractFloat
    siz = map( n->round(Int, rate*n, RoundNearestTiesAway), size(s) )
    resize(s, siz; params...)
end

function resize(s::Array, rates::NTuple{N,U}; params...) where U <: AbstractFloat where N
    siz = round.(Int, size(s) .* rates)
    resize(s, siz; params...)
end

end
