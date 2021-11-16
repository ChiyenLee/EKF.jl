"""
"""
struct ImuState{T} <: EKF.State{16, T}
    p𝑥::T; p𝑦::T; p𝑧::T
    q𝑤::T; q𝑥::T; q𝑦::T; q𝑧::T
    v𝑥::T; v𝑦::T; v𝑧::T
    α𝑥::T; α𝑦::T; α𝑧::T
    β𝑥::T; β𝑦::T; β𝑧::T
end

"""
"""
struct ImuError{T} <: EKF.ErrorState{15, T}
    𝕕p𝑥::T; 𝕕p𝑦::T; 𝕕p𝑧::T
    𝕕q𝑥::T; 𝕕q𝑦::T; 𝕕q𝑧::T
    𝕕v𝑥::T; 𝕕v𝑦::T; 𝕕v𝑧::T
    𝕕α𝑥::T; 𝕕α𝑦::T; 𝕕α𝑧::T
    𝕕β𝑥::T; 𝕕β𝑦::T; 𝕕β𝑧::T
end


"""
"""
struct ImuInput{T} <: EKF.Input{6, T}
    v̇𝑥::T; v̇𝑦::T; v̇𝑧::T
    ω𝑥::T; ω𝑦::T; ω𝑧::T
end


"""
"""
struct ViconMeasure{T} <: EKF.Measurement{7, T}
    p𝑥::T; p𝑦::T; p𝑧::T
    q𝑤::T; q𝑥::T; q𝑦::T; q𝑧::T
end


"""
"""
struct ViconError{T} <: EKF.ErrorMeasurement{6, T}
    𝕕p𝑥::T; 𝕕p𝑦::T; 𝕕p𝑧::T
    𝕕q𝑥::T; 𝕕q𝑦::T; 𝕕q𝑧::T
end

function getComponents(x::ImuState)
    p = SA[x.p𝑥, x.p𝑦, x.p𝑧]
    q = Rotations.UnitQuaternion(x.q𝑤, x.q𝑥, x.q𝑦, x.q𝑧)
    v = SA[x.v𝑥, x.v𝑦, x.v𝑧]
    α = SA[x.α𝑥, x.α𝑦, x.α𝑧]
    β = SA[x.β𝑥, x.β𝑦, x.β𝑧]
    return p, q, v, α, β
end


function getComponents(u::ImuInput)
    v̇ = SA[u.v̇𝑥, u.v̇𝑦, u.v̇𝑧]
    ω = SA[u.ω𝑥, u.ω𝑦, u.ω𝑧]
    return v̇, ω
end

# Add an error state to another state to create a new state
function EKF.state_composition(x::ImuState, dx::ImuError)::ImuState
    p = SA[x.p𝑥, x.p𝑦, x.p𝑧]
    q = Rotations.UnitQuaternion(x.q𝑤, x.q𝑥, x.q𝑦, x.q𝑧)
    v = SA[x.v𝑥, x.v𝑦, x.v𝑧]
    α = SA[x.α𝑥, x.α𝑦, x.α𝑧]
    β = SA[x.β𝑥, x.β𝑦, x.β𝑧]

    𝕕p = SA[dx.𝕕p𝑥, dx.𝕕p𝑦, dx.𝕕p𝑧]
    tmp = SA[dx.𝕕q𝑥, dx.𝕕q𝑦, dx.𝕕q𝑧]
    𝕕q = Rotations.RotationError(tmp, Rotations.CayleyMap())
    𝕕v = SA[dx.𝕕v𝑥, dx.𝕕v𝑦, dx.𝕕v𝑧]
    𝕕α = SA[dx.𝕕α𝑥, dx.𝕕α𝑦, dx.𝕕α𝑧]
    𝕕β = SA[dx.𝕕β𝑥, dx.𝕕β𝑦, dx.𝕕β𝑧]

    pos = p + 𝕕p
    ori = Rotations.add_error(q, 𝕕q)
    vel = v + 𝕕v
    acc_bias = α + 𝕕α
    ori_bias = β + 𝕕β

    x = ImuState(pos..., Rotations.params(ori)..., vel..., acc_bias..., ori_bias...)
    return x
end

# # Compute the error state between two states
function EKF.measurement_error(m2::ViconMeasure, m1::ViconMeasure)::ViconError
    p₁ = SA[m1.p𝑥, m1.p𝑦, m1.p𝑧]
    q₁ = Rotations.UnitQuaternion(m1.q𝑤, m1.q𝑥, m1.q𝑦, m1.q𝑧)

    p₂ = SA[m2.p𝑥, m2.p𝑦, m2.p𝑧]
    q₂ = Rotations.UnitQuaternion(m2.q𝑤, m2.q𝑥, m2.q𝑦, m2.q𝑧)

    pos_er = p₂ - p₁
    ori_er = Rotations.rotation_error(q₂, q₁, Rotations.CayleyMap())

    dx = ViconError(pos_er..., ori_er...)
    return dx
end


###############################################################################
#                               Dynamics
###############################################################################
function dynamics(x::ImuState, u::ImuInput)::SVector{16}
	g = SA[0, 0, 9.81]

    # Get various compoents
    p = SA[x.p𝑥, x.p𝑦, x.p𝑧]
    q = Rotations.UnitQuaternion(x.q𝑤, x.q𝑥, x.q𝑦, x.q𝑧)
    v = SA[x.v𝑥, x.v𝑦, x.v𝑧]
    α = SA[x.α𝑥, x.α𝑦, x.α𝑧]
    β = SA[x.β𝑥, x.β𝑦, x.β𝑧]

    v̇ᵢ = SA[u.v̇𝑥, u.v̇𝑦, u.v̇𝑧]
    ωᵢ = SA[u.ω𝑥, u.ω𝑦, u.ω𝑧]

    # Body velocity writen in inertia cooridantes
    ṗ = q * v
    # Compute the rotational kinematics
    q̇ = Rotations.kinematics(q, ωᵢ - β)
    # Translational acceleration
    v̇ = v̇ᵢ - α - q' * g
    # Rate of change in biases is 0
    α̇ = @SVector zeros(3); β̇ = @SVector zeros(3)

    ret = SA[ṗ[1], ṗ[2], ṗ[3],
             q̇[1], q̇[2], q̇[3], q̇[4],
             v̇[1], v̇[2], v̇[3],
             α̇[1], α̇[2], α̇[3],
             β̇[1], β̇[2], β̇[3]]
    return ret
end

function EKF.process(x::ImuState, u::ImuInput, dt::Float64)::ImuState
    k1 = dynamics(x, u)
    k2 = dynamics(x + 0.5 * dt * k1, u)
    k3 = dynamics(x + 0.5 * dt * k2, u)
    k4 = dynamics(x + dt * k3, u)

    tmp = ImuState(x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4))
    q𝑤, q𝑥, q𝑦, q𝑧 = Rotations.params(Rotations.UnitQuaternion(tmp.q𝑤, tmp.q𝑥, tmp.q𝑦, tmp.q𝑧))
    ret = ImuState(tmp.p𝑥, tmp.p𝑦, tmp.p𝑧,
                   q𝑤, q𝑥, q𝑦, q𝑧,
                   tmp.v𝑥, tmp.v𝑦, tmp.v𝑧,
                   tmp.α𝑥, tmp.α𝑦, tmp.α𝑧,
                   tmp.β𝑥, tmp.β𝑦, tmp.β𝑧)
    return ret
end

function EKF.error_process_jacobian(xₖ::ImuState, uₖ::ImuInput, dt::Float64)::SMatrix{length(ImuError), length(ImuError), Float64}
    A = ForwardDiff.jacobian(st->EKF.process(ImuState(st), uₖ, dt), SVector(xₖ))
    # Get various compoents
    qₖ = Rotations.UnitQuaternion(xₖ.q𝑤, xₖ.q𝑥, xₖ.q𝑦, xₖ.q𝑧)

    Jₖ = [(@SMatrix [i==j ? 1. : 0. for i = 1:3, j = 1:15]);
          [(@SMatrix zeros(4, 3))  Rotations.∇differential(qₖ)  (@SMatrix zeros(4, 9))];
          (@SMatrix [i+6==j ? 1. : 0. for i = 1:9, j = 1:15])]

    xₖ₊₁ = EKF.process(xₖ, uₖ, dt)
    qₖ₊₁ = Rotations.UnitQuaternion(xₖ₊₁.q𝑤, xₖ₊₁.q𝑥, xₖ₊₁.q𝑦, xₖ₊₁.q𝑧)
    Jₖ₊₁ = [(@SMatrix [i==j ? 1. : 0. for i = 1:3, j = 1:15]);
            [(@SMatrix zeros(4, 3))  Rotations.∇differential(qₖ₊₁)  (@SMatrix zeros(4, 9))];
            (@SMatrix [i+6==j ? 1. : 0. for i = 1:9, j = 1:15])]

    # ∂(dxₖ)/∂xₖ * ∂f(xₖ,uₖ)/∂(xₖ₋₁) * ∂(xₖ₋₁)/∂(dxₖ₋₁)
    return Jₖ₊₁' * A * Jₖ
end

function EKF.measure(::Type{<:ViconMeasure}, x::ImuState)::ViconMeasure
    return ViconMeasure(x.p𝑥, x.p𝑦, x.p𝑧, x.q𝑤, x.q𝑥, x.q𝑦, x.q𝑧)
end

function EKF.error_measure_jacobian(::Type{<:ViconMeasure}, xₖ::ImuState)::SMatrix{length(ViconError), length(ImuError), Float64}
    A = ForwardDiff.jacobian(st->EKF.measure(ViconMeasure, ImuState(st)), SVector(xₖ))

    qₖ = Rotations.UnitQuaternion(xₖ.q𝑤, xₖ.q𝑥, xₖ.q𝑦, xₖ.q𝑧)

    Jₖ = [(@SMatrix [i==j ? 1. : 0. for i = 1:3, j = 1:15]);
          [(@SMatrix zeros(4, 3))  Rotations.∇differential(qₖ)  (@SMatrix zeros(4, 9))];
          (@SMatrix [i+6==j ? 1. : 0. for i = 1:9, j = 1:15])]

    ŷ = EKF.measure(ViconMeasure, xₖ)
    q̂ = Rotations.UnitQuaternion(ŷ.q𝑤, ŷ.q𝑥, ŷ.q𝑦, ŷ.q𝑧)
    Gₖ = [(@SMatrix [i==j ? 1. : 0. for i = 1:3, j = 1:6]);
          [(@SMatrix zeros(4, 3))  Rotations.∇differential(q̂)]]

    # ∂(dyₖ)/∂(yₖ) * ∂(yₖ)/∂(yₖ) * ∂(yₖ)/∂(dyₖ)
    return Gₖ' * A * Jₖ
end