module Kelvinlets

using LinearAlgebra
using StaticArrays
using ForwardDiff

export Kelvinlet
export VectorField
export get_force
export transform
export inverse_transform

## Generic Vector Field
####################################

abstract type VectorField{N,T} end

function get_force(vf::V, point::AbstractVector{U}) where V <: VectorField where U <: Number
    error("Please specialize the method get_force() for type $V.")
end

transform(K::V, point) where V <: VectorField = point .+ get_force(K, point)

@inline function jacobian(vf::V, point::SVector{N,T}) where V <: VectorField{N,T} where N where T
    return ForwardDiff.jacobian(p -> get_force(vf, p), point)
end

Base.eltype(K::VectorField{N,T}) where N where T = T

function inverse_transform(
      vf :: VectorField{N,T}
    , q :: SVector{N,T}
    ; verbose :: Bool      = false
    , GN_damping :: Number = 0.5 # Gauss-Newton damping
    , max_iter :: Integer  = 50  # Gauss-Newton max iterations
    , threshold :: Number  = 0.1 # Residual-norm stop condition
    , LM_param :: Number   = 0.0 # Levenberg-Marquardt parameter
    , LM_update :: Number  = 0.5 # Levenberg-Marquardt update factor
    , LM_mat :: Symbol     = :identity # Levenberg-Marquardt matrix
    , initial_guess :: SVector{N,T} = q
) where N where T where DEBUG
    p = initial_guess # Initial guess
	i = 1
    while i < max_iter
        Jp = jacobian(vf, p)
        Kp = get_force(vf, p)
        residue = p + Kp - q
        b = (Jp' + I)*residue
        A = Jp'Jp + 2Jp' + I
        if LM_mat == :identity
            A = A + (LM_param * LM_update^i)*I
        elseif LM_mat == :diag
            A = A + (LM_param * LM_update^i)*diagm(abs.(diag(A))) # FIXME: Doesn't work!
        end
        δ = A \ b
        p = p - GN_damping*δ
        if verbose
            @show p, norm(residue)
        end
        if norm(residue) < threshold
            break
        end
		i += 1
    end
	return p, i
end

@inline function inverse_transform_fast(
      vf :: VectorField{N,T}
    , q :: SVector{N,T}
    ; GN_damping :: Number = 0.5 # Gauss-Newton damping
    , max_iter :: Integer  = 50  # Gauss-Newton max iterations
    , threshold :: Number  = 0.1 # Residual-norm stop condition
    , initial_guess :: SVector{N,T} = q
    , ignores_params...
) where {N, T}
    p = initial_guess # Initial guess
    for i in 1:max_iter
        Jp = jacobian(vf, p)
        Kp = get_force(vf, p)
        residue = p + Kp - q
        b = (Jp' + I)*residue
        A = Jp'Jp + 2Jp' + I
        δ = A \ b
        p = p - GN_damping*δ
        if norm(residue) < threshold
            break
        end
    end
	return p
end

function inverse_transform(K::VectorField{N,T}, point::AbstractVector{U}; params...) where N where T where U <: Number
    spoint = SVector{N,T}(point)
    return inverse_transform(K, spoint; params...)
end

struct ScaledVectorField{N,T} <: VectorField{N,T}
    α :: T
    v :: VectorField{N,T}
end

function get_force(svf::ScaledVectorField{N,T}, point::AbstractVector{U}) where N where T where U <: Number
    return svf.α * get_force(svf.v, point)
end

## Kelvinlet Vector Field
####################################

struct Kelvinlet{N,T,FixedBorder} <: VectorField{N,T}
    pivot :: SVector{N,T}
    force :: SVector{N,T}
    poiss :: T
    epsilon :: T
    elshear :: T
    a :: T
    b :: T
    c :: T
    domain_size :: SVector{N,T}
end

function Kelvinlet(
      pivot :: AbstractVector{<:Number}
    , force :: AbstractVector{<:Number}
    ; poiss :: Number = 0.4
    , epsilon :: Number = 1
    , elshear :: Number = 1.0
    , fixed_border :: Bool = false
    , domain_size :: Union{Nothing,NTuple{S,<:Number} where S} = nothing
)
    if fixed_border && isnothing(domain_size)
        error("Must inform Kelvinlet domain_size = size(img) when fixed_border = true.")
    end
    @assert length(pivot) == length(force) "Pivot and force must have the same dimensionality."
    T = promote_type(float(eltype(pivot)), float(eltype(force)))
    a = 1 / (4π * T(elshear)) |> T
    b = a / (4 - 4T(poiss)) |> T
    c = 2 / (3a - 2b) |> T
    N = length(pivot)
    ds = SVector{N}(isnothing(domain_size) ? zeros(N) : [domain_size...])
    spivot = SVector{N}(T.(pivot))
    sforce = SVector{N}(T.(force))
    return Kelvinlet{N,T,fixed_border}(spivot, sforce, T.(poiss), T.(epsilon), T.(elshear), a, b, c, ds)
end

function Base.show(io::IO, K::Kelvinlet{N,T,FixedBorder}) where {N, T, FixedBorder}
    print(io, "Kelvinlet{$T} in $(N)-D $(FixedBorder ? "with" : "without") fixed border and parameters:\n")
    for field in fieldnames(Kelvinlet)
        print(io, "  $field = ", getfield(K, field), "\n")
    end
end

@inline function distances_to_image_borders(siz::SVector{N}, px) where N
    dists = SVector{N}(ntuple(i -> min(px[i] - 1, siz[i] - px[i]), N)...)
end

@inline function border_transitions(siz::SVector{N}, px; radius = 50) where N
    dists = distances_to_image_borders(siz, px) :: SVector{N}
    βs = map(dists) do dist
        return sin(π*(min(dist,radius) / radius)/2)
    end
    return βs
end

# function border_transitions_jacobian(img, px; params...)
#     return ForwardDiff.jacobian(p -> border_transitions(img, p; params...), px)
# end

function get_force(K::Kelvinlet{N,T,FixedBorder}, point::AbstractVector{U}; overruleFixed = false) where N where T where U <: Number where FixedBorder
    rbold = point .- K.pivot
    re    = sqrt(sum(abs2, rbold) + K.epsilon^2)
    kelvinState = (((K.a - K.b) / re) * I +
                   (K.b / re^3) * (rbold * rbold') +
                   (K.a / T(2)) * (K.epsilon^2 / re^3) * I)
    u = K.c * K.epsilon * kelvinState * K.force
    if FixedBorder && !overruleFixed
        u = u .* border_transitions(K.domain_size, point)
    end
    return u
end

# TODO: generalize for any N
@inline function jacobian_explicit(K::Kelvinlet{N,T,FixedBorder}, point::SVector{N,T}) where N where T where FixedBorder
    @inline Power(a,b) = a^b
    @inline Sqrt(x) = sqrt(x)
	@inline Sin(x) = sin(x)
	@inline Cos(x) = cos(x)
	@inline Min(x, y, z) = min(x, y, z)

    a,b,c = K.a, K.b, K.c
    x, y = point
    ox, oy = K.pivot
    fx, fy = K.force
    z = K.epsilon
	J11 = c*z*((3*b*fy*(ox - x)*(-ox + x)*(-oy + y))/Power(Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5) + (b*fy*(-oy + y))/Power(Power(ox - x,2) + Power(oy - y,2) + Power(z,2),1.5) + fx*((3*b*(ox - x)*Power(-ox + x,2))/Power(Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5) + (3*a*(ox - x)*Power(z,2))/(2*Power(Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5)) + ((a - b)*(ox - x))/Power(Power(ox - x,2) + Power(oy - y,2) + Power(z,2),1.5) + (2*b*(-ox + x))/Power(Power(ox - x,2) + Power(oy - y,2) + Power(z,2),1.5)));
	J21 = c*z*((3*b*fx*(ox - x)*(-ox + x)*(-oy + y))/Power(Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5) + (b*fx*(-oy + y))/Power(Power(ox - x,2) + Power(oy - y,2) + Power(z,2),1.5) + fy*((3*b*(ox - x)*Power(-oy + y,2))/Power(Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5) + (3*a*(ox - x)*Power(z,2))/(2*Power(Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5)) + ((a - b)*(ox - x))/Power(Power(ox - x,2) + Power(oy - y,2) + Power(z,2),1.5)));
	J12 = c*z*((3*b*fy*(-ox + x)*(oy - y)*(-oy + y))/Power(Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5) + (b*fy*(-ox + x))/Power(Power(ox - x,2) + Power(oy - y,2) + Power(z,2),1.5) + fx*((3*b*Power(-ox + x,2)*(oy - y))/Power(Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5) + (3*a*(oy - y)*Power(z,2))/(2*Power(Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5)) + ((a - b)*(oy - y))/Power(Power(ox - x,2) + Power(oy - y,2) + Power(z,2),1.5)));
	J22 = c*z*((3*b*fx*(-ox + x)*(oy - y)*(-oy + y))/Power(Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5) + (b*fx*(-ox + x))/Power(Power(ox - x,2) + Power(oy - y,2) + Power(z,2),1.5) + fy*((3*b*(oy - y)*Power(-oy + y,2))/Power(Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5) + (3*a*(oy - y)*Power(z,2))/(2*Power(Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5)) + ((a - b)*(oy - y))/Power(Power(ox - x,2) + Power(oy - y,2) + Power(z,2),1.5) + (2*b*(-oy + y))/Power(Power(ox - x,2) + Power(oy - y,2) + Power(z,2),1.5)));

    J = [J11 J12; J21 J22]

	if FixedBorder
        z = 50
	    sx, sy = K.domain_size

		varx = 0
		vary = 0

		if sx - 2x <= -1 && sx - x - z <= 0
			varx = -1
		elseif sx - 2x > -1 && x - z <= 1
			varx = 1
		end

		if sy - 2y <= -1 && sy - y - z <= 0
			vary = -1
		elseif sy - 2y > -1 && y - z <= 1
			vary = 1
		end

		u = get_force(K, point; overruleFixed = true)
		b = border_transitions(K.domain_size, point)

	    Pi = π
		xborderDiff = (Pi*Cos((Pi*Min(sx - x,-1 + x,z))/(2z))*varx)/(2. *z)
		yborderDiff = (Pi*Cos((Pi*Min(sy - y,-1 + y,z))/(2z))*vary)/(2. *z)

        db = [xborderDiff, yborderDiff]

        J = diagm(u .* db) + diagm(b) * J
	end

    return J
end

end
