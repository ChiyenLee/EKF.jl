using EKF

include("test_states.jl")

println(rand(ImuInput))

# rand(Random)

MethodError

# ErrorStateFilter{ImuState, 
#                 ImuErrorState, 
#                 ImuInput, 
#                 ViconMeasurement, 
#                 ViconErrorMeasurement}(1, 2, 3, 4)