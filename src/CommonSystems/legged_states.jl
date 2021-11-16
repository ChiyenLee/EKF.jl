###############################################################################
# State and State Error 
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
# Control Input 
###############################################################################
struct ImuInput{T} <: EKF.Input{6, T}
    v̇𝑥::T; v̇𝑦::T; v̇𝑧::T
    ω𝑥::T; ω𝑦::T; ω𝑧::T
end

###############################################################################
# Observation Model 
###############################################################################
struct ContactMeasure{T} <: EKF.Measurement{3, T}
    px::T; py::T; pz::T
end

struct ErrorContactMeasure{T} <: EKF.ErrorMeasurement{3, T}
    px::T; py::T; pz::T
end

mutable struct ContactObservation1{T} <: EKF.Observation{T}
    measurement::ContactMeasure{T}
    measure_cov::SMatrix{length(ContactMeasure), length(ContactMeasure), T, length(ContactMeasure) * length(ContactMeasure)}
end

mutable struct ContactObservation2{T} <: EKF.Observation{T}
    measurement::ContactMeasure{T}
    measure_cov::SMatrix{length(ContactMeasure), length(ContactMeasure), T, length(ContactMeasure) * length(ContactMeasure)}
end

mutable struct ContactObservation3{T} <: EKF.Observation{T}
    measurement::ContactMeasure{T}
    measure_cov::SMatrix{length(ContactMeasure), length(ContactMeasure), T, length(ContactMeasure) * length(ContactMeasure)}
end

mutable struct ContactObservation4{T} <: EKF.Observation{T}
    measurement::ContactMeasure{T}
    measure_cov::SMatrix{length(ContactMeasure), length(ContactMeasure), T, length(ContactMeasure) * length(ContactMeasure)}
end

# ###############################################################################
# #                               Process and Process Jacobian 
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


###############################################################################
#   Measurement and measure joacbians 
###############################################################################

function EKF.measure(::Type{ContactObservation1{T}}, x::LeggedState)::ContactMeasure where T 
    r, q, v, p1, p2, p3 ,p4, α, β = getComponents(x)
    p_body = q' * (p1  - r) 
    return ContactMeasure(p_body...)
end

function EKF.measure(::Type{ContactObservation2{T}}, x::LeggedState)::ContactMeasure where T 
    r, q, v, p1, p2, p3 ,p4, α, β = getComponents(x)
    p_body = q' * (p2  - r) 
    return ContactMeasure(p_body...)
end

function EKF.measure(::Type{ContactObservation3{T}}, x::LeggedState)::ContactMeasure where T 
    r, q, v, p1, p2, p3 ,p4, α, β = getComponents(x)
    p_body = q' * (p3  - r) 
    return ContactMeasure(p_body...)
end

function EKF.measure(::Type{ContactObservation4{T}}, x::LeggedState)::ContactMeasure where T 
    r, q, v, p1, p2, p3 ,p4, α, β = getComponents(x)
    p_body = q' * (p4  - r) 
    return ContactMeasure(p_body...)
end


function EKF.error_measure_jacobian(::Type{ContactObservation1{T}}, xₖ::LeggedState)::SMatrix{length(ErrorContactMeasure), length(LeggedError), Float64} where T 
    A = ForwardDiff.jacobian(st->EKF.measure(ContactObservation1{T}, LeggedState(st)), SVector(xₖ))

    qₖ = Rotations.UnitQuaternion(xₖ.qw, xₖ.qx, xₖ.qy, xₖ.qz)
	Jₖ = @MMatrix zeros(length(LeggedState), length(LeggedError));
	Jₖ[4:7, 4:6] .= Rotations.∇differential(qₖ)
 	Jₖ[diagind(Jₖ)[1:3]] .= 1.0;  Jₖ[diagind(Jₖ,-1)[7:length(LeggedError)]] .= 1;

    return A * Jₖ
end

function EKF.error_measure_jacobian(::Type{ContactObservation2{T}}, xₖ::LeggedState)::SMatrix{length(ErrorContactMeasure), length(LeggedError), Float64} where T
    A = ForwardDiff.jacobian(st->EKF.measure(ContactObservation2{T}, LeggedState(st)), SVector(xₖ))

    qₖ = Rotations.UnitQuaternion(xₖ.qw, xₖ.qx, xₖ.qy, xₖ.qz)
	Jₖ = @MMatrix zeros(length(LeggedState), length(LeggedError));
	Jₖ[4:7, 4:6] .= Rotations.∇differential(qₖ)
 	Jₖ[diagind(Jₖ)[1:3]] .= 1.0;  Jₖ[diagind(Jₖ,-1)[7:length(LeggedError)]] .= 1;

    return A * Jₖ
end

function EKF.error_measure_jacobian(::Type{ContactObservation3{T}}, xₖ::LeggedState)::SMatrix{length(ErrorContactMeasure), length(LeggedError), Float64} where T 
    A = ForwardDiff.jacobian(st->EKF.measure(ContactObservation3{T}, LeggedState(st)), SVector(xₖ))

    qₖ = Rotations.UnitQuaternion(xₖ.qw, xₖ.qx, xₖ.qy, xₖ.qz)
	Jₖ = @MMatrix zeros(length(LeggedState), length(LeggedError));
	Jₖ[4:7, 4:6] .= Rotations.∇differential(qₖ)
 	Jₖ[diagind(Jₖ)[1:3]] .= 1.0;  Jₖ[diagind(Jₖ,-1)[7:length(LeggedError)]] .= 1;

    return A * Jₖ
end

function EKF.error_measure_jacobian(::Type{ContactObservation4{T}}, xₖ::LeggedState)::SMatrix{length(ErrorContactMeasure), length(LeggedError), Float64} where T 
    A = ForwardDiff.jacobian(st->EKF.measure(ContactObservation4{T}, LeggedState(st)), SVector(xₖ))

    qₖ = Rotations.UnitQuaternion(xₖ.qw, xₖ.qx, xₖ.qy, xₖ.qz)
	Jₖ = @MMatrix zeros(length(LeggedState), length(LeggedError));
	Jₖ[4:7, 4:6] .= Rotations.∇differential(qₖ)
 	Jₖ[diagind(Jₖ)[1:3]] .= 1.0;  Jₖ[diagind(Jₖ,-1)[7:length(LeggedError)]] .= 1;

    return A * Jₖ
end


###############################################################################
#                          Compositions 
###############################################################################

function EKF.state_composition(s::LeggedState, ds::LeggedError)
    r, q, v, p1, p2, p3 ,p4, α, β = getComponents(s)
    dr, dϕ, dv, dp1, dp2, dp3 , dp4, dα, dβ = getComponents(ds)

	ang_error = Rotations.RotationError(SVector{3, Float64}(dϕ), Rotations.CayleyMap())
	qₖ₊₁ = Rotations.add_error(Rotations.UnitQuaternion(q), ang_error)
	qₖ₊₁ = @SVector [qₖ₊₁.w, qₖ₊₁.x, qₖ₊₁.y, qₖ₊₁.z]

	r = r + dr
	v = v + dv
	α = α + dα
	β = β + dβ
    p1 = p1 + dp1 
    p2 = p2 + dp2 
    p3 = p3 + dp3 
    p4 = p4 + dp4 

    return LeggedState(r..., qₖ₊₁..., v..., p1..., p2..., p3..., p4..., α..., β...)
end

function EKF.measurement_error(m2::ContactMeasure, m1::ContactMeasure)::ErrorContactMeasure
    p1 = @SVector [m1.px, m1.py, m1.pz]
    p2 = @SVector [m2.px, m2.py, m2.pz]

    pos_er = p2 - p1 

    dx = ErrorContactMeasure(pos_er)
    return dx
end

###############################################################################
#                          Utilities 
###############################################################################

function getComponents(x::LeggedState)
    r = @SVector [x.rx, x.ry, x.rz]
    q = Rotations.UnitQuaternion(x.qw, x.qx, x.qy, x.qz)
    p1 = @SVector [x.p1x, x.p1y, x.p1z]
    p2 = @SVector [x.p2x, x.p2y, x.p2z]
    p3 = @SVector [x.p3x, x.p3y, x.p3z]
    p4 = @SVector [x.p4x, x.p4y, x.p4z]
    v = @SVector [x.vx, x.vy, x.vz]
    α = @SVector [x.αx, x.αy, x.αz]
    β = @SVector [x.βx, x.βy, x.βz]
    return r, q, v, p1, p2, p3 ,p4, α, β
end

function getComponents(x::LeggedError)
    r = @SVector [x.rx, x.ry, x.rz]
    ϕ = @SVector [x.qx, x.qy, x.qz]
    p1 = @SVector [x.p1x, x.p1y, x.p1z]
    p2 = @SVector [x.p2x, x.p2y, x.p2z]
    p3 = @SVector [x.p3x, x.p3y, x.p3z]
    p4 = @SVector [x.p4x, x.p4y, x.p4z]
    v = @SVector [x.vx, x.vy, x.vz]
    α = @SVector [x.αx, x.αy, x.αz]
    β = @SVector [x.βx, x.βy, x.βz]
	return  r, ϕ, v, p1, p2, p3 ,p4, α, β
end

function getComponents(u::ImuInput)
    v̇ = @SVector [u.v̇𝑥, u.v̇𝑦, u.v̇𝑧]
    ω = @SVector [u.ω𝑥, u.ω𝑦, u.ω𝑧]
    return v̇, ω
end


