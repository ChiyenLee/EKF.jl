module EKF
    using StaticArrays
    # export State, ErrorState, Input, Measurement, ErrorMeasurement
    # export ErrorStateFilter, estimateState!, prediction, innovation, update!
    # export measure, process
    # export error_measure_jacobian, error_process_jacobian
    # export state_composition, measurement_error
    # export prediction!, update!

    include("abstract_states.jl")
    include("filter.jl")

    include("CommonSystems/CommonSystems.jl")
    using .CommonSystems
end # module ErrorStateEKF
