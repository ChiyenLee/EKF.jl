struct EkfError <: Exception
     msg::String
 end

###############################################################################
#                        Abstract Filter Types
###############################################################################
abstract type State{Nₛ, T} <: FieldVector{Nₛ, T} end
abstract type ErrorState{Nₑₛ, T} <: FieldVector{Nₑₛ, T} end

abstract type Input{Nᵢ, T} <: FieldVector{Nᵢ, T} end

"""
    Measurement{Nₘ, T}

Abstract type describing what is being measured.
"""
abstract type Measurement{Nₘ, T} <: FieldVector{Nₘ, T} end
abstract type ErrorMeasurement{Nₑₘ, T} <: FieldVector{Nₑₘ, T} end

"""
    Observation(meas::Measurement, cov::SMatrix)

A observation consists of the acutal measurement and its associated
covariance. This is particularly useful for when you have multiple
independent measurments in a EKF.

# Example
Consider measuring the position of a robot:
    ...
    struct Position{T} <: EKF.Measurement{3, T}
        px::T; py::T; pz::T
    end
    ...
We express a particular obervation of this measurement as:
    ...
    positionObservation = Observation(Position(0.,0.,0.),
                                      SA[1. 0 0; 0 1 0; 0 0 1;])
    ...
"""
struct Observation{M<:Measurement, Nₘ, Nₑₘ, T, Lₑₘ}
    meas::SVector{Nₘ, T}
    cov::SMatrix{Nₑₘ, Nₑₘ, T, Lₑₘ}

    function Observation(meas::Measurement{Nₘ, T}, cov::SMatrix{Nₑₘ, Nₑₘ, T, Lₑₘ}) where {Nₘ, Nₑₘ, T, Lₑₘ}
        M = typeof(meas)
        vec = SVector{Nₘ, T}(meas)
        new{M, Nₘ, Nₑₘ, T, Lₑₘ}(vec, cov)
    end
end

"""
    getMeasurement(obs::Observation)

Useful helper function.
"""
@inline getMeasurement(obs::Observation{M, Nₘ, Nₑₘ, T}) where {M<:Measurement, Nₘ, Nₑₘ, T} = M(obs.meas)

"""
    getCovariance(obs::Observation)

Useful helper function.
"""
@inline getCovariance(obs::Observation{M, Nₘ, Nₑₘ, T}) where {M<:Measurement, Nₘ, Nₑₘ, T} = obs.cov

###############################################################################
#                      Process / Process Jacobian
###############################################################################
function process(state::State, input::Input, dt::Float64)
    throw(
        EkfError("User must define the `process` function: `process(state::$(typeof(state)), input::$(typeof(input)), dt::Float64)`")
    )
end

function error_process_jacobian(state::State, input::Input, dt::Float64)
    throw(
        EkfError("User must define the `error_process_jacobian` function: `error_process_jacobian(state::$(typeof(state)), input::$(typeof(input)), dt::Float64)`")
    )
end

###############################################################################
#                       Measure / Measurement Jacobian
###############################################################################
function measure(meas_typ::Type{<:Measurement}, state::State) where {M}
    throw(
        EkfError("User must define the `measure` function: `measure(::Type{$(meas_typ)}, state::$(typeof(state)))`")
    )
end

function error_measure_jacobian(meas_typ::Type{<:Measurement}, state::State)
    throw(
        EkfError("User must define the `error_measure_jacobian` function: `error_measure_jacobian(::Type{$(meas_typ)}, state::$(typeof(state)))`")
    )
end

###############################################################################
#                 State/Measurement & Composition/Difference
###############################################################################
# Add an error state to another state to create a new state
function state_composition(state::State, error_state::ErrorState)::State
    throw(
        EkfError("User must define the `state_composition` function: `state_composition(state::$(typeof(state)), error_state::$(typeof(error_state)))::$(typeof(state))`")
    )
end

# Compute the error measurement between two measurement
function measurement_error(measure_1::Measurement, measure_2::Measurement)::ErrorMeasurement
    throw(
        EkfError("User must define the `measurement_error` function: `measurement_error(measure_1::$(typeof(measure_1)), measure_2::$(typeof(measure_2)))::ErrorMeasurement`")
    )
end