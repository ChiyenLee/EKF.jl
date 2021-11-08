using Revise
import EKF
import EKF.CommonSystems as ComSys
using StaticArrays
using BenchmarkTools
using Test

dt = .1
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
    EKF.update!($ekf, $oriObs)
end
@test maximum(b.gctimes) == 0  # no garbage collection
@test b.memory == 0            # no dynamic memory allocations
display(b)