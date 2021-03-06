"""
"""
struct ImuState{T} <: EKF.State{16, T}
    pš„::T; pš¦::T; pš§::T
    qš¤::T; qš„::T; qš¦::T; qš§::T
    vš„::T; vš¦::T; vš§::T
    Ī±š„::T; Ī±š¦::T; Ī±š§::T
    Ī²š„::T; Ī²š¦::T; Ī²š§::T
end

"""
"""
struct ImuError{T} <: EKF.ErrorState{15, T}
    špš„::T; špš¦::T; špš§::T
    šqš„::T; šqš¦::T; šqš§::T
    švš„::T; švš¦::T; švš§::T
    šĪ±š„::T; šĪ±š¦::T; šĪ±š§::T
    šĪ²š„::T; šĪ²š¦::T; šĪ²š§::T
end

"""
"""
struct ImuInput{T} <: EKF.Input{6, T}
    vĢš„::T; vĢš¦::T; vĢš§::T
    Ļš„::T; Ļš¦::T; Ļš§::T
end

"""
"""
struct ViconMeasure{T} <: EKF.Measurement{7, T}
    pš„::T; pš¦::T; pš§::T
    qš¤::T; qš„::T; qš¦::T; qš§::T
end

"""
"""
struct ViconError{T} <: EKF.ErrorMeasurement{6, T}
    špš„::T; špš¦::T; špš§::T
    šqš„::T; šqš¦::T; šqš§::T
end

function getComponents(x::ImuState)
    p = SA[x.pš„, x.pš¦, x.pš§]
    q = Rotations.UnitQuaternion(x.qš¤, x.qš„, x.qš¦, x.qš§)
    v = SA[x.vš„, x.vš¦, x.vš§]
    Ī± = SA[x.Ī±š„, x.Ī±š¦, x.Ī±š§]
    Ī² = SA[x.Ī²š„, x.Ī²š¦, x.Ī²š§]
    return p, q, v, Ī±, Ī²
end

function getComponents(u::ImuInput)
    vĢ = SA[u.vĢš„, u.vĢš¦, u.vĢš§]
    Ļ = SA[u.Ļš„, u.Ļš¦, u.Ļš§]
    return vĢ, Ļ
end

# Add an error state to another state to create a new state
function EKF.state_composition(x::ImuState{T}, dx::ImuError{T})::ImuState{T} where T
    p = SA[x.pš„, x.pš¦, x.pš§]
    q = Rotations.UnitQuaternion(x.qš¤, x.qš„, x.qš¦, x.qš§)
    v = SA[x.vš„, x.vš¦, x.vš§]
    Ī± = SA[x.Ī±š„, x.Ī±š¦, x.Ī±š§]
    Ī² = SA[x.Ī²š„, x.Ī²š¦, x.Ī²š§]

    šp = SA[dx.špš„, dx.špš¦, dx.špš§]
    šq = Rotations.RotationError(SA[dx.šqš„, dx.šqš¦, dx.šqš§], Rotations.CayleyMap())
    šv = SA[dx.švš„, dx.švš¦, dx.švš§]
    šĪ± = SA[dx.šĪ±š„, dx.šĪ±š¦, dx.šĪ±š§]
    šĪ² = SA[dx.šĪ²š„, dx.šĪ²š¦, dx.šĪ²š§]

    pos = p + šp
    ori = Rotations.params(Rotations.add_error(q, šq))
    vel = v + šv
    acc_bias = Ī± + šĪ±
    ori_bias = Ī² + šĪ²

    x = ImuState{T}(pos..., ori..., vel..., acc_bias..., ori_bias...)
    return x
end

# # Compute the error state between two states
function EKF.measurement_error(m2::ViconMeasure{T}, m1::ViconMeasure{T})::ViconError{T} where T
    pā = SA[m1.pš„, m1.pš¦, m1.pš§]
    qā = Rotations.UnitQuaternion{T}(m1.qš¤, m1.qš„, m1.qš¦, m1.qš§)

    pā = SA[m2.pš„, m2.pš¦, m2.pš§]
    qā = Rotations.UnitQuaternion{T}(m2.qš¤, m2.qš„, m2.qš¦, m2.qš§)

    pos_er = pā - pā
    ori_er = Rotations.rotation_error(qā, qā, Rotations.CayleyMap())

    dx = ViconError{T}(pos_er..., ori_er...)
    return dx
end


###############################################################################
#                               Dynamics
###############################################################################
function dynamics(x::ImuState, u::ImuInput)::SVector{16}
	g = SA[0, 0, 9.81]

    # Get various compoents
    p = SA[x.pš„, x.pš¦, x.pš§]
    q = Rotations.UnitQuaternion(x.qš¤, x.qš„, x.qš¦, x.qš§)
    v = SA[x.vš„, x.vš¦, x.vš§]
    Ī± = SA[x.Ī±š„, x.Ī±š¦, x.Ī±š§]
    Ī² = SA[x.Ī²š„, x.Ī²š¦, x.Ī²š§]

    vĢįµ¢ = SA[u.vĢš„, u.vĢš¦, u.vĢš§]
    Ļįµ¢ = SA[u.Ļš„, u.Ļš¦, u.Ļš§]

    # Body velocity writen in inertia cooridantes
    pĢ = q * v
    # Compute the rotational kinematics
    qĢ = Rotations.kinematics(q, Ļįµ¢ - Ī²)
    # Translational acceleration
    vĢ = vĢįµ¢ - Ī± - q' * g
    # Rate of change in biases is 0
    Ī±Ģ = @SVector zeros(3); Ī²Ģ = @SVector zeros(3)

    ret = SA[pĢ[1], pĢ[2], pĢ[3],
             qĢ[1], qĢ[2], qĢ[3], qĢ[4],
             vĢ[1], vĢ[2], vĢ[3],
             Ī±Ģ[1], Ī±Ģ[2], Ī±Ģ[3],
             Ī²Ģ[1], Ī²Ģ[2], Ī²Ģ[3]]
    return ret
end

"""
4th Order Runga Kutta Method for integrating the dynamics function of the quadrotor.
"""
function EKF.process(x::ImuState, u::ImuInput, dt::T)::ImuState where T
    k1 = dynamics(x, u)
    k2 = dynamics(ImuState(x + 0.5 * dt * k1), u)
    k3 = dynamics(ImuState(x + 0.5 * dt * k2), u)
    k4 = dynamics(ImuState(x + dt * k3), u)

    tmp = ImuState(x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4))
    # Renormalize quaternion
    qš¤, qš„, qš¦, qš§ = Rotations.params(Rotations.UnitQuaternion(tmp.qš¤, tmp.qš„, tmp.qš¦, tmp.qš§))
    ret = ImuState(tmp.pš„, tmp.pš¦, tmp.pš§,
                   qš¤, qš„, qš¦, qš§,
                   tmp.vš„, tmp.vš¦, tmp.vš§,
                   tmp.Ī±š„, tmp.Ī±š¦, tmp.Ī±š§,
                   tmp.Ī²š„, tmp.Ī²š¦, tmp.Ī²š§)
    return ret
end

function EKF.error_process_jacobian(xā::ImuState, uā::ImuInput, dt::T)::SMatrix{15, 15} where T
    A = ForwardDiff.jacobian(st->EKF.process(ImuState(st), uā, dt), SVector(xā))
    # Get various compoents
    qā = Rotations.UnitQuaternion(xā.qš¤, xā.qš„, xā.qš¦, xā.qš§)

    Jā = [(@SMatrix [i==j ? 1. : 0. for i = 1:3, j = 1:15]);
          [(@SMatrix zeros(4, 3))  Rotations.ādifferential(qā)  (@SMatrix zeros(4, 9))];
          (@SMatrix [i+6==j ? 1. : 0. for i = 1:9, j = 1:15])]

    xāāā = EKF.process(xā, uā, dt)
    qāāā = Rotations.UnitQuaternion(xāāā.qš¤, xāāā.qš„, xāāā.qš¦, xāāā.qš§)
    Jāāā = [(@SMatrix [i==j ? 1. : 0. for i = 1:3, j = 1:15]);
            [(@SMatrix zeros(4, 3))  Rotations.ādifferential(qāāā)  (@SMatrix zeros(4, 9))];
            (@SMatrix [i+6==j ? 1. : 0. for i = 1:9, j = 1:15])]

    # ā(dxā)/āxā * āf(xā,uā)/ā(xāāā) * ā(xāāā)/ā(dxāāā)
    return Jāāā' * A * Jā
end

function EKF.measure(::Type{<:ViconMeasure}, x::ImuState)::ViconMeasure
    return ViconMeasure(x.pš„, x.pš¦, x.pš§, x.qš¤, x.qš„, x.qš¦, x.qš§)
end

function EKF.error_measure_jacobian(::Type{<:ViconMeasure}, xā::ImuState)::SMatrix{6, 15}
    A = ForwardDiff.jacobian(st->EKF.measure(ViconMeasure, ImuState(st)), SVector(xā))

    qā = Rotations.UnitQuaternion(xā.qš¤, xā.qš„, xā.qš¦, xā.qš§)

    Jā = [(@SMatrix [i==j ? 1. : 0. for i = 1:3, j = 1:15]);
          [(@SMatrix zeros(4, 3))  Rotations.ādifferential(qā)  (@SMatrix zeros(4, 9))];
          (@SMatrix [i+6==j ? 1. : 0. for i = 1:9, j = 1:15])]

    yĢ = EKF.measure(ViconMeasure, xā)
    qĢ = Rotations.UnitQuaternion(yĢ.qš¤, yĢ.qš„, yĢ.qš¦, yĢ.qš§)
    Gā = [(@SMatrix [i==j ? 1. : 0. for i = 1:3, j = 1:6]);
          [(@SMatrix zeros(4, 3))  Rotations.ādifferential(qĢ)]]

    # ā(dyā)/ā(yā) * ā(yā)/ā(yā) * ā(yā)/ā(dyā)
    return Gā' * A * Jā
end