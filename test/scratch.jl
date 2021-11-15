using Revise
import EKF
import EKF.CommonSystems as ComSys
using StaticArrays
using BenchmarkTools
using Test

dt = .1
# Test Performance of IMU Vicon Filter
state = ComSys.ImuState{Float32}(zeros(3)..., [1.,0,0,0]..., zeros(9)...)
stateErr = ComSys.ImuError{Float32}(zeros(15)...)
input = ComSys.ImuInput{Float32}(zeros(6))
meas = ComSys.ViconMeasure{Float32}(zeros(3)..., [1.,0,0,0]...)
measErr = ComSys.ViconError{Float32}(zeros(6))

EKF.state_composition(state, stateErr)

# %%
state = ComSys.ImuState(zeros(3)..., [1.,0,0,0]..., zeros(9)...)
input = ComSys.ImuInput(zeros(6))
meas = ComSys.ViconMeasure(zeros(3)..., [1.,0,0,0]...)
est_cov = @SMatrix [i==j ? 1.5 : 0. for i = 1:15, j = 1:15]
process_cov = @SMatrix [i==j ? .3 : 0. for i = 1:15, j = 1:15]

ekf = EKF.ErrorStateFilter{ComSys.ImuState, ComSys.ImuError, ComSys.ImuInput}(state, est_cov, process_cov)

meas_cov = @SMatrix [i==j ? 1. : 0. for i = 1:6, j = 1:6]
oriObs = EKF.Observation(meas, meas_cov)

b = @benchmark begin
    EKF.prediction!($ekf, $input, $dt)
end
@test maximum(b.gctimes) == 0  # no garbage collection
@test b.memory == 0            # no dynamic memory allocations
display(b)

# %%
@btime ComSys.ImuState($ekf.est_state)