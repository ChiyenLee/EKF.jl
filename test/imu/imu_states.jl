using EKF
using StaticArrays
using SparseArrays
using LinearAlgebra: I
using ForwardDiff: jacobian
using Rotations: UnitQuaternion, RotationError, CayleyMap, add_error
using Rotations: rotation_error, params, ∇differential, kinematics


###############################################################################
#
###############################################################################
mutable struct ImuState{T} <: State{16, T}
    p𝑥::T; p𝑦::T; p𝑧::T
    q𝑤::T; q𝑥::T; q𝑦::T; q𝑧::T
    v𝑥::T; v𝑦::T; v𝑧::T
    α𝑥::T; α𝑦::T; α𝑧::T
    β𝑥::T; β𝑦::T; β𝑧::T
end

function getComponents(state::ImuState)
    return (state[1:3], UnitQuaternion(state[4:7]..., false), state[8:10],
            state[11:13], state[14:16])
end

###############################################################################
#
###############################################################################
mutable struct ImuErrorState{T} <: ErrorState{15, T}
    𝕕p𝑥::T; 𝕕p𝑦::T; 𝕕p𝑧::T
    𝕕q𝑥::T; 𝕕q𝑦::T; 𝕕q𝑧::T
    𝕕v𝑥::T; 𝕕v𝑦::T; 𝕕v𝑧::T
    𝕕α𝑥::T; 𝕕α𝑦::T; 𝕕α𝑧::T
    𝕕β𝑥::T; 𝕕β𝑦::T; 𝕕β𝑧::T
end

function getComponents(err::ImuErrorState)
    return (err[1:3], RotationError(SVector{3}(err[4:6]), CayleyMap()), err[7:9],
            err[10:12], err[13:15])
end

###############################################################################
#
###############################################################################
mutable struct ImuInput{T} <: Input{6, T}
    v̇𝑥::T; v̇𝑦::T; v̇𝑧::T
    ω𝑥::T; ω𝑦::T; ω𝑧::T
end

function getComponents(in::ImuInput)
    return (in[1:3], in[4:6])
end

###############################################################################
#
###############################################################################
mutable struct ViconMeasurement{T} <: Measurement{7, T}
    p𝑥::T; p𝑦::T; p𝑧::T
    q𝑤::T; q𝑥::T; q𝑦::T; q𝑧::T
end

function getComponents(meas::ViconMeasurement)
    return (meas[1:3], UnitQuaternion(meas[4:7]..., false))
end

###############################################################################
#
###############################################################################
mutable struct ViconErrorMeasurement{T} <: ErrorMeasurement{6, T}
    𝕕p𝑥::T; 𝕕p𝑦::T; 𝕕p𝑧::T
    𝕕q𝑥::T; 𝕕q𝑦::T; 𝕕q𝑧::T
end

function getComponents(err::ViconErrorMeasurement)
    return (err[1:3], RotationError(SA[err[4:6]...], CayleyMap()))
end

# Add an error state to another state to create a new state
function EKF.state_composition(x::ImuState, dx::ImuErrorState)::ImuState
    p, q, v, α, β = getComponents(x)
    𝕕p, 𝕕q, 𝕕v, 𝕕α, 𝕕β = getComponents(dx)

    pos = p + 𝕕p
    ori = add_error(q, 𝕕q)
    vel = v + 𝕕v
    acc_bias = α + 𝕕α
    ori_bias = β + 𝕕β

    x = ImuState(pos..., params(ori)..., vel..., acc_bias..., ori_bias...)
    return x
end

# # Compute the error state between two states
function EKF.measurement_error(m2::ViconMeasurement, m1::ViconMeasurement)::ViconErrorMeasurement
    p₁, q₁ = getComponents(m1)
    p₂, q₂ = getComponents(m2)

    pos_er = p₂ - p₁
    ori_er = rotation_error(q₂, q₁, CayleyMap())

    dx = ViconErrorMeasurement(pos_er..., ori_er...)
    return dx
end


###############################################################################
#                               Dynamics
###############################################################################
function dynamics(state::ImuState, input::ImuInput)
	g = [0,0,9.81]

    p, q, v, α, β = getComponents(state)
    v̇ᵢ, ωᵢ = getComponents(input)
    # Body velocity writen in inertia cooridantes
    ṗ = q * v
    # Compute the rotational kinematics
    q̇ = kinematics(q, ωᵢ - β)
    # Translational acceleration
    v̇ = v̇ᵢ - α - q' * g
    # Rate of change in biases is 0
    α̇ = zeros(3); β̇ = zeros(3)
    return [ṗ; q̇; v̇; α̇; β̇]
end

function EKF.process(x::ImuState, u::ImuInput, dt::Float64)::ImuState
    k1 = dynamics(x, u)
    k2 = dynamics(x + 0.5 * dt * k1, u)
    k3 = dynamics(x + 0.5 * dt * k2, u)
    k4 = dynamics(x + dt * k3, u)
    xnext = ImuState(x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4))

    xnext.q𝑤, xnext.q𝑥, xnext.q𝑦, xnext.q𝑧 = params(UnitQuaternion(xnext.q𝑤, xnext.q𝑥, xnext.q𝑦, xnext.q𝑧))

    return xnext
end

function EKF.error_process_jacobian(state::ImuState, input::ImuInput, dt::Float64)
    A = jacobian(st->process(ImuState(st), input, dt), SVector(state))

    _, qₖ, _, _, _  = getComponents(state)
    Jₖ = cat(I(3), ∇differential(qₖ), I(9), dims=(1,2))

    _, qₖ₊₁, _, _, _  = getComponents(state)
    Jₖ₊₁ = cat(I(3), ∇differential(qₖ₊₁), I(9), dims=(1,2))

    # ∂(dxₖ)/∂xₖ * ∂f(xₖ,uₖ)/∂(xₖ₋₁) * ∂(xₖ₋₁)/∂(dxₖ₋₁)
    return Jₖ₊₁' * A * Jₖ
end

function EKF.measure(state::ImuState)::ViconMeasurement
    p, q, v, α, β = getComponents(state)
    return ViconMeasurement(p..., params(q)...)
end

function EKF.error_measure_jacobian(state::ImuState)
    A = jacobian(st->measure(ImuState(st)), state)

    _, qₖ₊₁, _, _, _  = getComponents(state)
    Jₖ₊₁ = cat(I(3), ∇differential(qₖ₊₁), I(9), dims=(1,2))

    _, q̂ = getComponents(measure(state))
    Gₖ₊₁ = cat(I(3), ∇differential(q̂), dims=(1,2))

    # ∂(dyₖ)/∂(yₖ) * ∂(yₖ)/∂(yₖ) * ∂(yₖ)/∂(dyₖ)
    return Gₖ₊₁' * A * Jₖ₊₁
end