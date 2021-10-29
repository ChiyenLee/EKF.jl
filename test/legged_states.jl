###############################################################################
#
###############################################################################
struct LeggedState{T} <: EKF.State{28, T}
    rx::T; ry::T; rz::T
    qw::T; qx::T; qy::T; qz::T
    vx::T; vy::T; vz::T
    p1x::T; p1y::T; p1z::T
    p2x::T; p2y::T; p2z::T
    p3x::T; p3y::T; p3z::T
    p4x::T; p4y::T; p4z::T
    αx::T; αy::T; αz::T
    βx::T; βy::T; βz::T 
end


###############################################################################
#
###############################################################################
struct LeggedError{T} <: EKF.ErrorState{27, T}
    rx::T; ry::T; rz::T
    qx::T; qy::T; qz::T
    vx::T; vy::T; vz::T
    p1x::T; p1y::T; p1z::T
    p2x::T; p2y::T; p2z::T
    p3x::T; p3y::T; p3z::T
    p4x::T; p4y::T; p4z::T 
    αx::T; αy::T; αz::T
    βx::T; βy::T; βz::T 
end


###############################################################################
#
###############################################################################
struct ImuInput{T} <: EKF.Input{6, T}
    v̇𝑥::T; v̇𝑦::T; v̇𝑧::T
    ω𝑥::T; ω𝑦::T; ω𝑧::T
end

###############################################################################
#
###############################################################################
abstract type ContactMeasure{T} <: EKF.Measurement{3, T} end 

struct ContactMeasure1{T} <: ContactMeasure{T}
    px::T; py::T; pz::T
end

struct ContactMeasure2{T} <: ContactMeasure{T}
    px::T; py::T; pz::T
end

struct ContactMeasure3{T} <: ContactMeasure{T}
    px::T; py::T; pz::T
end

struct ContactMeasure4{T} <: ContactMeasure{T}
    px::T; py::T; pz::T
end

###############################################################################
#
###############################################################################
abstract type ErrorContactMeasure{T} <: EKF.ErrorMeasurement{3, T} end 

struct ErrorContactMeasure1{T} <: ErrorContactMeasure{T}
    p1x::T; p1y::T; p1z::T
end

struct ErrorContactMeasure2{T} <: ErrorContactMeasure{T}
    p1x::T; p1y::T; p1z::T
end

struct ErrorContactMeasure3{T} <: ErrorContactMeasure{T}
    p1x::T; p1y::T; p1z::T
end

struct ErrorContactMeasure4{T} <: ErrorContactMeasure{T}
    p1x::T; p1y::T; p1z::T
end

function getComponents(x::LeggedState)
    r = @SVector [x.rx, x.ry, x.rz]
    q = Rotations.UnitQuaternion(x.qw, x.qx, x.qy, x.qz)
    p1 = @SVector [x.p1x, x.p1y, x.p1z]
    p2 = @SVector [x.p2x, x.p2y, x.p2z]
    p3 = @SVector [x.p3x, x.p3y, x.p3z]
    p4 = @SVector [x.p4x, x.p4y, x.p4z]
    v = @SVector [x.vx, x.vx, x.vz]
    α = @SVector [x.αx, x.αy, x.αz]
    β = @SVector [x.βx, x.βy, x.βy]
    return r, q, v, p1, p2, p3 ,p4, α, β
end

function getComponents(u::ImuInput)
    v̇ = @SVector [u.v̇𝑥, u.v̇𝑦, u.v̇𝑧]
    ω = @SVector [u.ω𝑥, u.ω𝑦, u.ω𝑧]
    return v̇, ω
end

# # Add an error state to another state to create a new state
# function EKF.state_composition(x::ImuState, dx::ImuError)::ImuState
#     p = @SVector [x.p𝑥, x.p𝑦, x.p𝑧]
#     q = Rotations.UnitQuaternion(x.q𝑤, x.q𝑥, x.q𝑦, x.q𝑧)
#     v = @SVector [x.v𝑥, x.v𝑦, x.v𝑧]
#     α = @SVector [x.α𝑥, x.α𝑦, x.α𝑧]
#     β = @SVector [x.β𝑥, x.β𝑦, x.β𝑧]

#     𝕕p = @SVector [dx.𝕕p𝑥, dx.𝕕p𝑦, dx.𝕕p𝑧]
#     tmp = @SVector [dx.𝕕q𝑥, dx.𝕕q𝑦, dx.𝕕q𝑧]
#     𝕕q = Rotations.RotationError(tmp, Rotations.CayleyMap())
#     𝕕v = @SVector [dx.𝕕v𝑥, dx.𝕕v𝑦, dx.𝕕v𝑧]
#     𝕕α = @SVector [dx.𝕕α𝑥, dx.𝕕α𝑦, dx.𝕕α𝑧]
#     𝕕β = @SVector [dx.𝕕β𝑥, dx.𝕕β𝑦, dx.𝕕β𝑧]

#     pos = p + 𝕕p
#     ori = Rotations.add_error(q, 𝕕q)
#     vel = v + 𝕕v
#     acc_bias = α + 𝕕α
#     ori_bias = β + 𝕕β

#     x = ImuState(pos..., Rotations.params(ori)..., vel..., acc_bias..., ori_bias...)
#     return x
# end

# # # Compute the error state between two states
# function EKF.measurement_error(m2::ViconMeasure, m1::ViconMeasure)::ViconError
#     p₁ = @SVector [m1.p𝑥, m1.p𝑦, m1.p𝑧]
#     q₁ = Rotations.UnitQuaternion(m1.q𝑤, m1.q𝑥, m1.q𝑦, m1.q𝑧)

#     p₂ = @SVector [m2.p𝑥, m2.p𝑦, m2.p𝑧]
#     q₂ = Rotations.UnitQuaternion(m2.q𝑤, m2.q𝑥, m2.q𝑦, m2.q𝑧)

#     pos_er = p₂ - p₁
#     ori_er = Rotations.rotation_error(q₂, q₁, Rotations.CayleyMap())

#     dx = ViconError(pos_er..., ori_er...)
#     return dx
# end


# ###############################################################################
# #                               Dynamics
# ###############################################################################

function EKF.process(x::LeggedState, u::ImuInput, h::Float64)::LeggedState
    g = @SVector [0,0,9.81]
    f, ω = getComponents(u)
    r, q, v, p1, p2, p3 ,p4, α, β = getComponents(x)
    C = q; 

    # Integrate 
    rₖ₊₁ = r + h*v + 0.5*h^2*(C*(f-α)-g)
	vₖ₊₁ = v + h*(C*(f - α) - g)
    qₖ₊₁ = Rotations.params(q) + 0.5 * Rotations.∇differential(C) * (ω - β) * h 
    qₖ₊₁ = qₖ₊₁ / norm(qₖ₊₁)

    return LeggedState(rₖ₊₁...,qₖ₊₁..., vₖ₊₁..., p1..., p2..., p3..., p4..., α..., β...)
end

function EKF.error_process_jacobian(sₖ::LeggedState, uₖ::ImuInput, h::Float64)::SMatrix{length(LeggedError), length(LeggedError), Float64}
    F = ForwardDiff.jacobian(st->EKF.process(LeggedState(st), uₖ, h), SVector(sₖ))
    qₖ = Rotations.UnitQuaternion(sₖ.qw, sₖ.qx, sₖ.qy, sₖ.qz)
    sₖ₊₁ₗₖ = EKF.process(sₖ,uₖ,h) # not ideal to call it again here but oh well
	qₖ₊₁ₗₖ = Rotations.UnitQuaternion(sₖ₊₁ₗₖ.qw, sₖ₊₁ₗₖ.qx, sₖ₊₁ₗₖ.qy, sₖ₊₁ₗₖ.qz)

	Jₖ = @MMatrix zeros(length(LeggedState), length(LeggedError));
	Jₖ₊₁ₗₖ = @MMatrix zeros(length(LeggedState), length(LeggedError));
	Jₖ[4:7, 4:6] .= Rotations.∇differential(qₖ)
	Jₖ[diagind(Jₖ)[1:3]] .= 1.0;  Jₖ[diagind(Jₖ,-1)[7:length(LeggedError)]] .= 1;
	Jₖ₊₁ₗₖ[4:7, 4:6] .= Rotations.∇differential(qₖ₊₁ₗₖ)
    Jₖ₊₁ₗₖ[diagind(Jₖ)[1:3]] .= 1.0;  Jₖ₊₁ₗₖ[diagind(Jₖ,-1)[7:length(LeggedError)]] .= 1;

	Jₖ = SMatrix(Jₖ)
	Jₖ₊₁ₗₖ = SMatrix(Jₖ₊₁ₗₖ)
    return Jₖ₊₁ₗₖ' * F * Jₖ
end

function EKF.measure(x::LeggedState)::ContactMeasure1
    r, q, v, p1, p2, p3 ,p4, α, β = getComponents(x)
    p_body = q' * (p1  - r) 
    return ContactMeasure1(p_body...)
end

function EKF.measure(x::LeggedState)::ContactMeasure2
    return ContactMeasure2(x.p2x, x.p2y, x.p2z)
end

function EKF.measure(x::LeggedState)::ContactMeasure3
    return ContactMeasure3(x.p2x, x.p2y, x.p2z)
end

function EKF.measure(x::LeggedState)::ContactMeasure4
    return ContactMeasure4(x.p2x, x.p2y, x.p2z)
end

# function EKF.error_measure_jacobian(xₖ::ImuState)::SMatrix{length(ViconError), length(ImuError), Float64}
#     A = ForwardDiff.jacobian(st->EKF.measure(ImuState(st)), SVector(xₖ))

#     qₖ = Rotations.UnitQuaternion(xₖ.q𝑤, xₖ.q𝑥, xₖ.q𝑦, xₖ.q𝑧)

#     Jₖ = [(@SMatrix [i==j ? 1. : 0. for i = 1:3, j = 1:15]);
#           [(@SMatrix zeros(4, 3))  Rotations.∇differential(qₖ)  (@SMatrix zeros(4, 9))];
#           (@SMatrix [i+6==j ? 1. : 0. for i = 1:9, j = 1:15])]

#     ŷ = EKF.measure(xₖ)
#     q̂ = Rotations.UnitQuaternion(ŷ.q𝑤, ŷ.q𝑥, ŷ.q𝑦, ŷ.q𝑧)
#     Gₖ = [(@SMatrix [i==j ? 1. : 0. for i = 1:3, j = 1:6]);
#           [(@SMatrix zeros(4, 3))  Rotations.∇differential(q̂)]]

#     # ∂(dyₖ)/∂(yₖ) * ∂(yₖ)/∂(yₖ) * ∂(yₖ)/∂(dyₖ)
#     return Gₖ' * A * Jₖ
# end