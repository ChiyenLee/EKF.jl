using Pkg 
using StaticArrays
using Revise
using EKF
using LinearAlgebra

include("imu/imu_states.jl")
include("imu/imu_dynamics.jl")

# state = ImuState{Float64}(zeros(16)); state.q𝑤 = 1
# state_err = ImuErrorState{Float64}(zeros(15)); 
# input = ImuInput{Float64}(zeros(6))
# vicon = ViconMeasurement{Float64}(zeros(7)); vicon.q𝑤 = 1
# vicon_err = ViconErrorMeasurement{Float64}(zeros(6))

state = rand(ImuState)
vicon = rand(ViconMeasurement)
input = rand(ImuInput)

P = Matrix(0.3* I(length(ImuErrorState))) 
W = Matrix(0.3* I(length(ImuErrorState))) 
R = Matrix(0.3 * I(length(ViconErrorMeasurement))) 

ekf = ErrorStateFilter{ImuState, ImuErrorState, ImuInput,
          ViconMeasurement, ViconErrorMeasurement}(state, P, W, R)

# input.v̇𝑥 = 10

# x_next, P_next = prediction(ekf, state, P, input, dt=0.01)
# innovation(ekf, x_next, P, vicon)
estimateState!(ekf, input, vicon, 0.01)
