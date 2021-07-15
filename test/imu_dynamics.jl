using StaticArrays
using LinearAlgebra: I
using SparseArrays
using Rotations: UnitQuaternion, RotationError, CayleyMap, add_error
using Rotations: rotation_error, params, ∇differential, kinematics
using ForwardDiff: jacobian

###############################################################################
#                               Dynamics
###############################################################################
function dynamics(state::ImuState, input::ImuInput)
    p, q, v, α, β = getComponents(state)
    v̇ᵢ, ωᵢ = getComponents(input)
    # Body velocity writen in inertia cooridantes
    ṗ = q * v
    # Compute the rotational kinematics
    q̇ = kinematics(q, ωᵢ - β)
    # Translational acceleration
    v̇ = v̇ᵢ - α
    # Rate of change in biases is 0
    α̇ = zeros(3); β̇ = zeros(3)
    return [ṗ; q̇; v̇; α̇; β̇]
end

function EKF.process(x::ImuState, u::ImuInput, dt::Float64)::ImuState
    k1 = dynamics(x, u)
    k2 = dynamics(x + 0.5 * dt * k1, u)
    k3 = dynamics(x + 0.5 * dt * k2, u)
    k4 = dynamics(x + dt * k3, u)
    xnext = x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return ImuState(xnext)
end

function EKF.error_process_jacobian(state::ImuState, input::ImuInput, dt::Float64)::Matrix
    A = jacobian(st->process(ImuState(st), input, dt), SVector(state))

    p, q, v, α, β = getComponents(state)
    Jₖ = Matrix(blockdiag(sparse(I(3)), sparse(∇differential(UnitQuaternion(q))), sparse(I(9))))
    
    p, q, v, α, β = getComponents(EKF.process(state, input, dt))
    Jₖ₊₁ = Matrix(blockdiag(sparse(I(3)), sparse(∇differential(UnitQuaternion(q))), sparse(I(9))))
    # ∂(dxₖ)/∂xₖ * ∂f(xₖ,uₖ)/∂(xₖ₋₁) * ∂(xₖ₋₁)/∂(dxₖ₋₁)
    return Jₖ₊₁' * A * Jₖ
end

function EKF.measure(state::ImuState)::ViconMeasurement
    p, q, v, α, β = getComponents(state)
    return ViconMeasurement(p..., params(q)...)
end

function EKF.error_measure_jacobian(state::ImuState, measurement::ViconMeasurement)::Matrix
    A = jacobian(st->measure(ImuState(st)), state)
    
    p, q, v, α, β = getComponents(state)
    Jₖ₊₁ = Matrix(blockdiag(sparse(I(3)), sparse(∇differential(UnitQuaternion(q))), sparse(I(9))))
    p, q = getComponents(measurement)
    Gₖ₊₁ = blockdiag(sparse(I(3)), sparse(∇differential(UnitQuaternion(q))))
    
    # ∂(dyₖ)/∂(yₖ) * ∂(yₖ)/∂(yₖ) * ∂(yₖ)/∂(dyₖ)
    return Gₖ₊₁' * A * Jₖ₊₁
end
