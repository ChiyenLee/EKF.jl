using EKF
using StaticArrays
using LinearAlgebra: I
using SparseArrays
using Rotations: UnitQuaternion, RotationError, CayleyMap, add_error
using Rotations: rotation_error, params, ∇differential


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
