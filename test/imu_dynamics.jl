using StaticArrays
using LinearAlgebra: normalize, norm, ×, I
using Rotations: kinematics, RotMatrix, UnitQuaternion, params
using ForwardDiff: jacobian

"""
"""
function dynamics(state::State, input::Input)
    p, q, v, α, β = getComponents(state)
    v̇ᵢ, ωᵢ = getComponents(input)

    # Body velocity writen in inertia cooridantes
    ṗ = q * v
    # Compute the rotational kinematics
    q̇ = kinematics(q, ωᵢ - β)
    # Translational acceleration
    v̇ = v̇ᵢ - α

    # Rate of change in biases is 0
    α̇ = zeros(3)
    β̇ = zeros(3)

    return [ṗ; q̇; v̇; α̇; β̇]
end

"""
"""
function process(x::State, u::Input, dt::Float64)
    k1 = dynamics(x, u)
    k2 = dynamics(x + 0.5 * dt * k1, u)
    k3 = dynamics(x + 0.5 * dt * k2, u)
    k4 = dynamics(x + dt * k3, u)
    xnext = x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return State(xnext)
end

"""
"""
function process_jacobian(state::State, input::Input, dt::Float64)
    return jacobian(st->process(State(st), input, dt), SVector(state))
end

"""
"""
function measure(state::State)
    q, β = getComponents(state)
    return Measurement(params(q))
end

"""
"""
function measure_jacobian(state)
    return jacobian(st->measure(State(st)), state)
end
