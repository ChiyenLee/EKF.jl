module EKF
    using StaticArrays
    using LinearAlgebra: I

    include("abstract_states.jl")
    include("filter.jl")

    include("CommonSystems/CommonSystems.jl")
    using .CommonSystems
end # module ErrorStateEKF
