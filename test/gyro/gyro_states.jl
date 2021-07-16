using EKF
using StaticArrays
using LinearAlgebra: I
using SparseArrays
using Rotations: UnitQuaternion, RotationError, CayleyMap, add_error
using Rotations: rotation_error, params, ∇differential, kinematics
using ForwardDiff: jacobian


###############################################################################
#
###############################################################################
mutable struct GyroState{T} <: State{7, T}
    q𝑤::T; q𝑥::T; q𝑦::T; q𝑧::T
    β𝑥::T; β𝑦::T; β𝑧::T

    # function GyroState(q𝑤::T, q𝑥::T, q𝑦::T, q𝑧::T, β𝑥::T, β𝑦::T, β𝑧::T) where T
    #     q𝑤, q𝑥, q𝑦, q𝑧 = params(UnitQuaternion(q𝑤, q𝑥, q𝑦, q𝑧))
    #     return new{T}(q𝑤, q𝑥, q𝑦, q𝑧, β𝑥, β𝑦, β𝑧)
    # end
end

function getComponents(state::GyroState)
    return (UnitQuaternion(state[1:4]..., false), state[5:7])
end

###############################################################################
#
###############################################################################
mutable struct GyroErrorState{T} <: ErrorState{6, T}
    𝕕q𝑥::T; 𝕕q𝑦::T; 𝕕q𝑧::T
    𝕕β𝑥::T; 𝕕β𝑦::T; 𝕕β𝑧::T
end

function getComponents(err::GyroErrorState)
    return (RotationError(SVector{3}(err[1:3]), CayleyMap()), err[4:6])
end

###############################################################################
#
###############################################################################
mutable struct GyroInput{T} <: Input{3, T}
    ω𝑥::T; ω𝑦::T; ω𝑧::T
end

function getComponents(in::GyroInput)
    return in[1:3]
end

###############################################################################
#
###############################################################################
mutable struct QuatMeasurement{T} <: Measurement{4, T}
    q𝑤::T; q𝑥::T; q𝑦::T; q𝑧::T

    # function QuatMeasurement(q𝑤::T, q𝑥::T, q𝑦::T, q𝑧::T) where T
    #     q𝑤, q𝑥, q𝑦, q𝑧 = params(UnitQuaternion(q𝑤, q𝑥, q𝑦, q𝑧))
    #     return new{T}(q𝑤, q𝑥, q𝑦, q𝑧)
    # end
end

function getComponents(meas::QuatMeasurement)
    return UnitQuaternion(meas[1:4]..., false)
end

###############################################################################
#
###############################################################################
mutable struct QuatErrorMeasurement{T} <: ErrorMeasurement{3, T}
    𝕕q𝑥::T; 𝕕q𝑦::T; 𝕕q𝑧::T
end

function getComponents(err::QuatErrorMeasurement)
    return RotationError(SA[err[1:3]...], CayleyMap())
end

# Add an error state to another state to create a new state
function EKF.state_composition(x::GyroState, dx::GyroErrorState)::GyroState
    q, β = getComponents(x)
    𝕕q, 𝕕β = getComponents(dx)

    ori = add_error(q, 𝕕q)
    ori_bias = β + 𝕕β

    x = GyroState(params(ori)..., ori_bias...)
    return x
end

# # Compute the error state between two states
function EKF.measurement_error(m2::QuatMeasurement, m1::QuatMeasurement)::QuatErrorMeasurement
    q₁ = getComponents(m1)
    q₂ = getComponents(m2)

    ori_er = rotation_error(q₂, q₁, CayleyMap())

    dx = QuatErrorMeasurement(ori_er...)
    return dx
end

###############################################################################
#                               Dynamics
###############################################################################
function dynamics(state::GyroState, input::GyroInput)
    q, β = getComponents(state)
    ωᵢ = getComponents(input)
    # Compute the rotational kinematics
    q̇ = kinematics(q, ωᵢ - β)
    # Rate of change in biases is 0
    β̇ = zeros(3)
    return [q̇; β̇]
end

function EKF.process(x::GyroState, u::GyroInput, dt::Float64)::GyroState
    k1 = dynamics(x, u)
    k2 = dynamics(x + 0.5 * dt * k1, u)
    k3 = dynamics(x + 0.5 * dt * k2, u)
    k4 = dynamics(x + dt * k3, u)
    xnext = GyroState(x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4))

    xnext.q𝑤, xnext.q𝑥, xnext.q𝑦, xnext.q𝑧 = params(UnitQuaternion(xnext.q𝑤, xnext.q𝑥, xnext.q𝑦, xnext.q𝑧))

    return xnext
end

function EKF.error_process_jacobian(state::GyroState, input::GyroInput, dt::Float64)::Matrix
    A = jacobian(st->process(GyroState(st), input, dt), SVector(state))

    qₖ, _ = getComponents(state)
    Jₖ = cat(∇differential(qₖ), I(3), dims=(1,2))
    qₖ₊₁, _ = getComponents(process(state, input, dt))
    Jₖ₊₁ = cat(∇differential(qₖ₊₁), I(3), dims=(1,2))

    # ∂(dxₖ)/∂xₖ * ∂f(xₖ,uₖ)/∂(xₖ₋₁) * ∂(xₖ₋₁)/∂(dxₖ₋₁)
    return Jₖ₊₁' * A * Jₖ
end

function EKF.measure(state::GyroState)::QuatMeasurement
    q, β = getComponents(state)
    return QuatMeasurement(params(q)...)
end

function EKF.error_measure_jacobian(state::GyroState)::Matrix
    A = jacobian(st->measure(GyroState(st)), state)

    qₖ₊₁, _ = getComponents(state)
    Jₖ₊₁ = cat(∇differential(qₖ₊₁), I(3), dims=(1,2))

    q̂ = getComponents(measure(state))
    Gₖ₊₁ = cat(∇differential(q̂), dims=(1,2))

    # ∂(dyₖ)/∂(yₖ) * ∂(yₖ)/∂(yₖ) * ∂(yₖ)/∂(dyₖ)
    return Gₖ₊₁' * A * Jₖ₊₁
end