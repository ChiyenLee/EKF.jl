struct EkfError <: Exception
     msg::String
 end

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
function process(state::State, input::Input, dt::Float64)
    EkfError("User must define the `process` function: `process(state::$(typeof(state)), input::$(typeof(input)), dt::Float64)`")
end

function error_process_jacobian(state::State, input::Input, dt::Float64)
    EkfError("User must define the `error_process_jacobian` function: `error_process_jacobian(state::$(typeof(state)), input::$(typeof(input)), dt::Float64)`")
end

###############################################################################
#                       Measure / Measurement Jacobian
###############################################################################
function measure(state::State)
    EkfError("User must define the `measure` function: `measure(state::$(typeof(state)))`")
end

function error_measure_jacobian(state::State)
    EkfError("User must define the `error_measure_jacobian` function: `error_measure_jacobian(state::$(typeof(state)))`")
end

###############################################################################
#                 State/Measurement & Composition/Difference
###############################################################################
# Add an error state to another state to create a new state
function state_composition(state::State, error_state::ErrorState)::State
    EkfError("User must define the `state_composition` function: `state_composition(state::$(typeof(state)), error_state::$(typeof(error_state)))::$(typeof(state))`")
end

# Compute the error measurement between two measurement
function measurement_error(measure_1::Measurement, measure_2::Measurement)::ErrorMeasurement
    EkfError("User must define the `measurement_error` function: `measurement_error(measure_1::$(typeof(measure_1)), measure_2::$(typeof(measure_2)))::ErrorMeasurement`")
end