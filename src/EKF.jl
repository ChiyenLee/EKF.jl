module EKF
    using StaticArrays

    include("abstract_states.jl")
    include("filter.jl")

    include("CommonSystems/CommonSystems.jl")
    using .CommonSystems
end # module ErrorStateEKF
