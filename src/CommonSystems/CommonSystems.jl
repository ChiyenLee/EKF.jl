module CommonSystems
    import EKF
    import ForwardDiff
    import Rotations
    using StaticArrays

    # include("imu_vicon.jl")
    include("imu_dynamics_discrete.jl")
end