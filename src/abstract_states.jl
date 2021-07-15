
###############################################################################
#                        Abstract Filter Types
###############################################################################
abstract type State{N, T} <: FieldVector{N, T} end 
abstract type ErrorState{Nₑ, T} <: FieldVector{Nₑ, T} end 

abstract type Input{M, T} <: FieldVector{M, T} end  

abstract type Measurement{S, T} <: FieldVector{S, T} end  
abstract type ErrorMeasurement{Sₑ, T} <: FieldVector{Sₑ, T} end  


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
