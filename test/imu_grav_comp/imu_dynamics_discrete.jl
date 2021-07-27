using Rotations
using ForwardDiff: jacobian
using Rotations: add_error, RotationError, ∇differential, CayleyMap, rotation_error
using SparseArrays
using StaticArrays
###############################################################################
#                        State Definitions 
###############################################################################
struct TrunkState{T} <: State{16,T}
	x::T; y::T; z::T 
	vx::T; vy::T; vz::T 
	qw::T; qx::T; qy::T; qz::T 
	αx::T; αy::T; αz::T
	βx::T; βy::T; βz::T 
end  

struct TrunkError{T} <: ErrorState{15,T} 
	𝕕x::T; 𝕕y::T; 𝕕z::T 
	𝕕vx::T; 𝕕vy::T; 𝕕vz::T 
	𝕕ϕx::T; 	𝕕ϕy::T; 	𝕕ϕz::T
	𝕕αx::T; 	𝕕αy::T ;	𝕕αz::T
	𝕕βx::T;	𝕕βy::T ;	𝕕βz::T
end 

struct ImuInput{T} <: Input{6,T}  
	fx::T;	fy::T; fz::T
	ωx::T; ωy::T; ωz::T
end 

struct Vicon{T} <: Measurement{7,T}  
	x::T; y::T; z::T
	qw::T; qx::T; qy::T; qz::T
end 

struct ViconError{T} <: ErrorMeasurement{6,T} 
	𝕕x::T;	𝕕y::T;	𝕕z::T
	𝕕ϕx::T;	𝕕ϕy::T;	𝕕ϕz::T
end 

###############################################################################
#                       Process / Process Jacobian
###############################################################################
function EKF.process(s::TrunkState, u::ImuInput, h::Float64)
	g = @SVector [0,0,9.81]
	ω = @SVector [u.ωx, u.ωy, u.ωz]
	f = @SVector [u.fx, u.fy, u.fz]

	r = @SVector [s.x, s.y, s.z]
	v = @SVector [s.vx, s.vy, s.vz]
	q = @SVector [s.qw, s.qx, s.qy, s.qz]
	α = @SVector [s.αx, s.αy, s.αz]
	β = @SVector [s.βx, s.βy, s.βz]
	# r, v, q, α, β = getComponents(s)
	
	C = UnitQuaternion(s.qw, s.qx, s.qy, s.qz) # from body to world

	rₖ₊₁ = r + h*v + 0.5*h^2*(C*(f-α)-g)  
	vₖ₊₁ = v + h*(C*(f - α) - g)
	qₖ₊₁ = q + 0.5 * ∇differential(C) * (ω - β) * h  #L(q) * ζ((ω-state.βω)*h)
	qₖ₊₁ = qₖ₊₁ / norm(qₖ₊₁)
	return TrunkState(rₖ₊₁...,vₖ₊₁...,qₖ₊₁...,α...,β...)
end 


function EKF.error_process_jacobian(s::TrunkState, u::ImuInput, h::Float64)
	sₖ₊₁ₗₖ = process(s,u,h) # not ideal to call it again here but oh well 
	qₖ = UnitQuaternion(s.qw, s.qx, s.qy, s.qz)
	qₖ₊₁ₗₖ = UnitQuaternion(sₖ₊₁ₗₖ.qw, sₖ₊₁ₗₖ.qx, sₖ₊₁ₗₖ.qy, sₖ₊₁ₗₖ.qz)

	R1 = @SMatrix [i==j ? 1. : 0. for i = 1:6, j = 1:15]
	M4 = @SMatrix zeros(4,6); M5 = ∇differential(qₖ); M6 =  @SMatrix zeros(4, 6)
    R2 = [M4  M5  M6]
    R3 = @SMatrix [i+6==j ? 1. : 0. for i = 1:6, j = 1:15]
    Jₖ = [R1; R2; R3]	

	# Jₖ = blockdiag(sparse(I(6)), sparse(∇differential(qₖ)), sparse(I(6))  )
	# Jₖ₊₁ₗₖ = blockdiag(sparse(I(6)), sparse(∇differential(qₖ₊₁ₗₖ)), sparse(I(6))  )
	Jₖ₊₁ₗₖ = AttitudeJacobian(blockdiag(sparse(I(6)), sparse(∇differential(qₖ₊₁ₗₖ)), sparse(I(6))  ))
    F = jacobian(st->process(TrunkState(st), u, h), SVector(s))
	return Jₖ₊₁ₗₖ' * F * Jₖ
end

###############################################################################
#                       Measure / Measurement Jacobian
###############################################################################
function EKF.measure(s::TrunkState)::Vicon
	return Vicon(s.x, s.y, s.z, s.qw, s.qx, s.qy, s.qz)
end

function EKF.error_measure_jacobian(s::TrunkState)
	H = zeros(length(ViconError),length(TrunkError))
	Jₓ = ∇differential(UnitQuaternion([s.qw, s.qx, s.qy, s.qz]))
	# Jy = ∇differential(UnitQuaternion([v.qw, v.qx, v.qy, v.qz]))

	H[4:6,7:9] = Jₓ' * I(4) * Jₓ
	H[1:3,1:3] = I(3)
	# H = SMatrix{length(ViconError),length(TrunkError)}(H)
	return H 
end 

###############################################################################
#                 State/Measurement & Composition/Difference
###############################################################################
# Add an error state to another state to create a new state
function EKF.state_composition(s::TrunkState, ds::TrunkError)
	dϕ = [ds.𝕕ϕx, ds.𝕕ϕy, ds.𝕕ϕz]
	r, v, q, α, β = getComponents(s)
	dr, dv, dϕ, dα, dβ = getComponents(ds)

	ang_error = RotationError(SVector{3, Float64}(dϕ), CayleyMap())
	qₖ₊₁ = add_error(UnitQuaternion(q), ang_error)
	qₖ₊₁ = @SVector [qₖ₊₁.w, qₖ₊₁.x, qₖ₊₁.y, qₖ₊₁.z]

	r = r + dr 
	v = v + dv 	
	α = α + dα
	β = β + dβ 

    return TrunkState([r; v; qₖ₊₁; α; β])
end

# Compute the error measurement between two measurement
function EKF.measurement_error(m2::Vicon, m1::Vicon)
	r1, q1 = getComponents(m1)
	r2, q2 = getComponents(m2)
	q1 = UnitQuaternion(q1)
	q2 = UnitQuaternion(q2)
	dr = r2 - r1 
	dϕ = rotation_error(q2, q1, CayleyMap())

    return ViconError([dr;dϕ])
end

###############################################################################
#                			Helper Functions 
###############################################################################
function getComponents(s::TrunkState)
	r = @SVector [s.x, s.y, s.z]
	v = @SVector [s.vx, s.vy, s.vz]
	q = @SVector [s.qw, s.qx, s.qy, s.qz]
	α = @SVector [s.αx, s.αy, s.αz]
	β = @SVector [s.βx, s.βy, s.βz]
	return (r, v, q, α, β)
end 

function getComponents(v::Vicon)
	r = @SVector [v.x, v.y, v.z]
	q = @SVector [v.qw, v.qx, v.qy, v.qz]
	return (r, q)
end 

function getComponents(e::TrunkError)
	dr = @SVector [e.𝕕x, e.𝕕y, e.𝕕z]
	dv = @SVector [e.𝕕vx, e.𝕕vy, e.𝕕vz]
	dϕ = @SVector [e.𝕕ϕx, e.𝕕ϕy, e.𝕕ϕz]
	dα = @SVector [e.𝕕αx, e.𝕕αy, e.𝕕αz]
	dβ = @SVector [e.𝕕βx, e.𝕕βy, e.𝕕βz]
	return (dr, dv, dϕ, dα, dβ)
end 

function getComponents(ve::ViconError)
	dr = @SVector [ve.𝕕x, ve.𝕕y, ve.𝕕z]
	dϕ = @SVector [ve.𝕕ϕx, ve.𝕕ϕy, ve.𝕕ϕz]
	return (dr, dϕ)
end 