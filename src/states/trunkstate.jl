###############################################################################
#                        State Definitions 
###############################################################################
abstract type State{N, T} <: FieldVector{N, T} end 
abstract type ErrorState{Nₑ, T} <: FieldVector{Nₑ, T} end 

abstract type Input{M, T} <: FieldVector{M, T} end  

abstract type Measurement{S, T} <: FieldVector{S, T} end  
abstract type ErrorMeasurement{Sₑ, T} <: FieldVector{Sₑ, T} end

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
	𝕕
end 

###############################################################################
#                       Error Mapping Jacobians
###############################################################################
function error_state_jacobian(state::State)
    return I
end

function error_measurement_jacobian(measurement::Measurement)
    return I
end

###############################################################################
#                 State/Measurement & Composition/Difference
###############################################################################
# Add an error state to another state to create a new state
function ⊕ₛ(x::State, dx::ErrorState)
    return State(x + dx)
end

# Compute the error measurement between two measurement
function ⊖ₘ(m2::Measurement, m1::Measurement)
    return m2 - m1
end
