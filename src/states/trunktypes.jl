###############################################################################
#                        State Definitions 
###############################################################################
# abstract type State{N, T} <: FieldVector{N, T} end 
# abstract type ErrorState{Nₑ, T} <: FieldVector{Nₑ, T} end 

# abstract type Input{M, T} <: FieldVector{M, T} end  

# abstract type Measurement{S, T} <: FieldVector{S, T} end  
# abstract type ErrorMeasurement{Sₑ, T} <: FieldVector{Sₑ, T} end

struct TrunkState{T} <: State{16,T} where T 
	x::T # position
	y::T 
	z::T 

	vx::T # velocity
	vy::T 
	vz::T 

	qw::T # quaternion
	qx::T 
	qy::T 
	qz::T 

	αx::T # acceleration bias 
	αy::T 
	αz::T

	βx::T # angular velocity bias 
	βy::T 
	βz::T 
end  

struct TrunkError{T} <: ErrorState{15,T} where T 
	𝕕x::T 
	𝕕y::T 
	𝕕z::T 
	
	𝕕vx::T 
	𝕕vy::T 
	𝕕vz::T 

	𝕕ϕx::T 
	𝕕ϕy::T 
	𝕕ϕz::T

	𝕕αx::T 
	𝕕αy::T 
	𝕕αz::T

	𝕕βx::T
	𝕕βy::T
	𝕕βz::T
end 

struct ImuInput{T} <: Input{6,T} where T 
	fx::T 
	fy::T
	fz::T

	βx::T
	βy::T
	βz::T
end 

struct Vicon{T} <: Measurement{7,T} where T 
	x::T
	y::T
	z::T
	qw::T
	qx::T
	qy::T
	qz::T
end 

struct ViconError{T} <: ErrorMeasurement{6,T} where T
	𝕕x::T
	𝕕y::T
	𝕕z::T
	𝕕ϕx::T
	𝕕ϕy::T
	𝕕ϕz::T
end 

###############################################################################
#                       Process / Process Jacobian
###############################################################################
function process(s::TrunkState, u::ImuInput, h::Float64)
	g = [0,0,9.81]
	ω = [u.ωx, u.ωy, u.ωz]
	f = [u.fx, u.fy, u.fz]
	r = [s.x, s.y, s.z]
	v = [s.vx, s.vy, s.vz]
	α = [s.αx, s.αy, s.αz]
	β = [s.βx, s.βy, s.βz]
	q = [s.qw, s.qx, s.qy, s.qz]
	C = UnitQuaternion(q)' # from body to world

	rₖ₊₁ = r + h*v + 0.5*h^2*(C'*(f-α)-g) 
	vₖ₊₁ = v + h*(C'*(f - α) - g)
	qₖ₊₁ = 0.5 * ∇differential(C) * (ω - β)*h  #L(q) * ζ((ω-state.βω)*h)

	return TrunkState([rₖ₊₁;vₖ₊₁;qₖ₊₁;α;β])
end 


function error_process_jacobian(s::TrunkState, u::ImuInput, h::Float64)
	sₖ₊₁ₗₖ = process(s,u,h) # not ideal to call it again here but oh well 
	qₖ = UnitQuaternion([s.qw, s.qx, s.qy, s.qz]) 
	qₖ₊₁ₗₖ = UnitQuaternion([sₖ₊₁ₗₖ.qw, sₖ₊₁ₗₖ.qx, sₖ₊₁ₗₖ.qy, sₖ₊₁ₗₖ.qz]) 

	Jₖ = blockdiag(sparse(I(6), sparse(∇differential(qₖ)), sparse(I(6))  ))
	Jₖ₊₁ₗₖ = blockdiag(sparse(I(6), sparse(∇differential(qₖ₊₁ₗₖ)), sparse(I(6))  ))

    F = jacobian(st->process(TrunkState(st), u, dt), SVector(s))
	return Jₖ₊₁ₗₖ' * F * Jₖ
end

###############################################################################
#                       Measure / Measurement Jacobian
###############################################################################
function measure(s::TrunkState)::Vicon
	return Vicon([s.x, s.y, s.z, s.qx, s.qy, s.qz])
end

function error_measurement_jacobian(v::Vicon, s::TrunkState)
	H = zeros(length(Vicon),length(TrunkError))
	Jₓ = ∇differential(UnitQuaternion([s.qw, s.qx, s.qy, s.qz]))
	Jy = ∇differential(UnitQuaternion([v.qw, v.qx, v.qy, v.qz]))

	H[4:6,4:6] = Jy' * I(4) * Jx 
	H[1:3,1:3] = I(3)
	return H 
end 

###############################################################################
#                 State/Measurement & Composition/Difference
###############################################################################
# Add an error state to another state to create a new state
function ⊕ₛ(s::State, ds::ErrorState)

	dϕ = [ds.𝕕ϕx, ds.𝕕ϕy, ds.𝕕ϕz]

	q = [s.qw, s.qx, s.qy, s.qz]

	# Method 2 this works :)
	ang_error = RotationError(SVector{3, Float64}(dϕ), CayleyMap())
	qₖ₊₁ = add_error(UnitQuaternion(q), ang_error)
	qₖ₊₁ .= [qₖ₊₁.w, qₖ₊₁.x, qₖ₊₁.y, qₖ₊₁.z]

	state.r .= state.r + state.dr
	state.v .= state.v + state.dv
	state.βf .= state.βf + state.dβf
	state.βω .= state.βω + state.dβω		


    return State(x + dx)
end

# Compute the error measurement between two measurement
function ⊖ₘ(m2::Measurement, m1::Measurement)
    return m2 - m1
end

###############################################################################
#                			Helper Functions 
###############################################################################
