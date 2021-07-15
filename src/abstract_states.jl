
###############################################################################
#                        Abstract Filter Types
###############################################################################
abstract type State{N, T} <: FieldVector{N, T} end 
abstract type ErrorState{Nₑ, T} <: FieldVector{Nₑ, T} end 

abstract type Input{M, T} <: FieldVector{M, T} end  

abstract type Measurement{S, T} <: FieldVector{S, T} end  
abstract type ErrorMeasurement{Sₑ, T} <: FieldVector{Sₑ, T} end  

###############################################################################
#                      Process / Process Jacobian
###############################################################################
function process(state::State, input::Input, dt::Float64)::State
    return state
end

function error_process_jacobian(state::State, input::Input, dt::Float64)
    return I
end


###############################################################################
#                       Measure / Measurement Jacobian
###############################################################################
function measure(state::State)::Measurement
    return state
end

function error_measure_jacobian(state::State, measurement::Measurement)
    return I
end

###############################################################################
#                 State/Measurement & Composition/Difference
###############################################################################
# Add an error state to another state to create a new state
function state_composition(x::State, dx::ErrorState)
    return State(x + dx)
end

# Compute the error measurement between two measurement
function measurement_error(m2::Measurement, m1::Measurement)
    return m2 - m1
end
