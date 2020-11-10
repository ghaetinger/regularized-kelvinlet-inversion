module ImageWarping


include("Fractals.jl")
include("Kelvinlets.jl")
include("Mipmaps.jl")
include("MyImageShowFullFidelity.jl")
include("MyVideoIO.jl")

using Colors
using Images
using ForwardDiff
using StaticArrays
using LinearAlgebra
using Interpolations

export Kelvinlets

abstract type AbstractImage end

# Images with Linear interpolation -- GL_LINEAR
struct LinearInterpolatedImage <: AbstractImage
    itp :: AbstractInterpolation
end

function LinearInterpolatedImage(img :: AbstractArray{C,N}; params...) where C where N
    return LinearInterpolatedImage(Mipmaps.interpolator(img; params...))
end

function (M::LinearInterpolatedImage)(x, y)
    return M.itp(x,y)
end

# Images with Linear interpolation -- GL_LINEAR
struct MipmappedImage <: AbstractImage
    itp :: Mipmaps.Mipmap
end

function MipmappedImage(img :: AbstractArray{C,N}) where C where N
	return MipmappedImage(Mipmaps.mipmap(img))
end

function (M::MipmappedImage)(x, y, J)
    return M.itp(x,y)
end

function (M::MipmappedImage)(xy, J)
    return M.itp(xy, J)
end

# Fractal Images
struct ContinuousImage <: AbstractImage
	query :: Function
	period :: Tuple{Float64, Float64}
	size :: Tuple{Int64, Int64}
end

function MandelbrotImage(;period :: Tuple{Float64, Float64} = (-1, 1),
						 size :: Tuple{Int64, Int64} = (500, 500),
						 samples :: Int64 = 100)
	@assert period[2] > period[1]
	return ContinuousImage((x, y) -> Fractals.mandelbrot(x, y, samples), period, size)
end

function JuliaImage(;period :: Tuple{Float64, Float64} = (-1, 1),
					size :: Tuple{Int64, Int64} = (500, 500),
					samples :: Int64 = 100)
	@assert period[2] > period[1]
	return ContinuousImage((x, y)  -> Fractals.julia(x, y, samples), period, size)
end

function (M::ContinuousImage)(y, x)
	periodMult = M.period[2] - M.period[1]
	x = x * periodMult + M.period[1]
	y = y * periodMult + M.period[1]
    return M.query(x,y)
end

function Base.size(continuous :: ContinuousImage)
	return continuous.size
end

function Base.zero(continuous :: ContinuousImage)
	return zeros(continuous.size)
end

function Base.zero(mip :: MipmappedImage)
    p1 = mip.itp.pyramid[1]
	return zeros(eltype(p1), size(p1))
end

function Base.size(mip :: MipmappedImage)
    p1 = mip.itp.pyramid[1]
	return size(p1)
end

function Base.zero(img :: LinearInterpolatedImage)
    p1 = img.itp
	return zeros(eltype(p1), size(p1))
end

function Base.size(img :: LinearInterpolatedImage)
    p1 = img.itp
	return size(p1)
end

function (M::LinearInterpolatedImage)(x, y, J)
    return M.itp(x,y)
end

function (M::LinearInterpolatedImage)(xy, J)
    return M.itp(xy...)
end

function warp_simple(
      vf :: Kelvinlets.VectorField{N,T}
    , img :: AbstractImage
    ; debug :: Val{DEBUG} = Val(false)
    , debug_dict :: Dict{Symbol,Any} = Dict{Symbol,Any}()
    , randomize_samples :: Val{RANDOMIZE_SAMPLES} = Val(false)
    , mipmap :: Val{MIPMAP} = Val(true)
) where {N, T, C <: Colorant, DEBUG, RANDOMIZE_SAMPLES, MIPMAP}
    img_warped = zero(img)
	
	# Commented due to abstractions
    # if MIPMAP
    #     img_prefiltered = Mipmaps.mipmap(img) # GL_LINEAR_MIPMAP_LINEAR
    # else
    #     img_prefiltered = Mipmaps.interpolator(img) # GL_LINEAR
    # end

    if DEBUG
        debug_dict[:iterations] = zeros(size(img));
        debug_dict[:jacobians] = fill(SMatrix{N,N,T}(zeros(T,N,N)), size(img));
        for datum in []
            debug_dict[datum] = Array{Any,N}(undef, size(img))
        end
    end

    optim_params = (
        max_iter   = 50,
        threshold  = 0.1,
        GN_damping = 0.5,
        LM_param   = 0.0,
        LM_update  = 0.5,
        LM_mat     = :identity
    )

    println("Inversion parameters: ", optim_params)
    flush(stdout)

    α = 1.0
    success = false
    while !success
        scaled_vf = Kelvinlets.ScaledVectorField(α, vf)
        threadlocal_α = ones(Threads.nthreads())
        threadlocal_success = trues(Threads.nthreads())
        Threads.@threads for Q in CartesianIndices(img_warped)
            try
                q = SVector(T.(Q.I))
                if RANDOMIZE_SAMPLES
                    q += rand(N) .- 0.5
                end
                p, iter =Kelvinlets.inverse_transform(scaled_vf, q ; optim_params... , initial_guess = q)
                J = Kelvinlets.jacobian(scaled_vf, p)
                ##################
                p_ndc = (p .- 1) ./ (size(img) .- 1)
                if MIPMAP
                    img_warped[Q] = img(p_ndc, J)
                else
                    img_warped[Q] = img(p_ndc...)
                end
                ##################
                if DEBUG
                    debug_dict[:iterations][Q] = iter
                    debug_dict[:jacobians][Q] = J
                end
            catch e
                if e isa Mipmaps.UninvertibleException
                    J = e.Jacobian
                    m = minimum(eigvals((J + J')/2))
                    α = -1/m
                    tid = Threads.threadid()
                    threadlocal_success[tid] = false
                    threadlocal_α[tid] = min(threadlocal_α[tid], α)
                end
            end
        end
        success = all(threadlocal_success)
        if !success
            α = 0.99 * minimum(threadlocal_α)
            println("Detected uninvertible deformation. Adapting force by α = $α.")
        end
    end

    return img_warped
end

function warp_simple_fast(
      vf :: Kelvinlets.VectorField{N,T}
    , img :: AbstractImage
    ; ignored_params...
) where {N, T}
    img_warped = zero(img)

    optim_params = (
        max_iter   = 50,
        threshold  = 0.1,
        GN_damping = 0.5,
    )

    Threads.@threads for Q in CartesianIndices(img_warped)
        q = SVector(T.(Q.I))
        p = Kelvinlets.inverse_transform_fast(vf, q ; optim_params... , initial_guess = q)
        J = Kelvinlets.jacobian(vf, p)
        p_ndc = (p .- 1) ./ (size(img) .- 1)
        img_warped[Q] = img(p_ndc, J)
    end

    return img_warped
end

function warp_simple_fast_multiscale(
      vf :: Kelvinlets.VectorField{N,T}
    , img :: AbstractImage
    ; ignored_params...
) where {N, T}
    img_warped = zero(img)

    optim_params = (
        max_iter   = 50,
        threshold  = 0.1,
        GN_damping = 0.5,
    )

    Δ = 10
    subinds = ntuple(n -> 1:Δ:size(img_warped,n), N)
    subinds_collected = Iterators.product(subinds...) |> collect
    subsols = Array{SVector{N,T}}(undef, size(subinds_collected)...)

    Threads.@threads for i in CartesianIndices(subsols)
        Q = CartesianIndices(img_warped)[subinds_collected[i]...]
        q = SVector(T.(Q.I))
        p = Kelvinlets.inverse_transform_fast(vf, q ; optim_params... , initial_guess = q)
        J = Kelvinlets.jacobian(vf, p)
        subsols[i] = p
    end

    subsols_itp = LinearInterpolation(subinds, subsols; extrapolation_bc = Flat())

    Threads.@threads for Q in CartesianIndices(img_warped)
        q = SVector(T.(Q.I))
        p = Kelvinlets.inverse_transform_fast(vf, q ; optim_params... , initial_guess = subsols_itp(q...))
        J = Kelvinlets.jacobian(vf, p)
        p_ndc = (p .- 1) ./ (size(img) .- 1)
        img_warped[Q] = img(p_ndc, J)
    end

    return img_warped
end

function warp_supersampling(
      vf :: Kelvinlets.VectorField
    , img :: AbstractImage
    ; num_samples :: Int = 10
    , params...
)
    params = Dict(k => v for (k,v) in params)
    params[:randomize_samples] = Val(true)
    img_warped = mapreduce(+, 1:num_samples) do itr
		warp_simple(vf, img; mipmap = Val(false), params...)
    end
    return img_warped ./ num_samples
end

end
