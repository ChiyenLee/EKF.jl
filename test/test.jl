using EKF
using StaticArrays: length
using LinearAlgebra: I 

include("imu/imu_states.jl")
include("imu/imu_dynamics.jl")


est_state = rand(ImuState)
est_cov = Matrix(.3 * I(length(ImuErrorState)))

process_cov = Matrix(.3 * I(length(ImuErrorState)))
measure_cov = Matrix(.3 * I(length(ViconErrorMeasurement)))

ekf = ErrorStateFilter{ImuState, ImuErrorState, ImuInput, ViconMeasurement, 
                       ViconErrorMeasurement}(est_state, est_cov, process_cov, measure_cov)

input = rand(ImuInput)
measurement = rand(ViconMeasurement)

# println(prediction(ekf, est_state, est_cov, input; dt=0.1))


estimateState!(ekf, input, measurement, 0.1);

