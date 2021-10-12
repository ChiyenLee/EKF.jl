import EKF
import EKF.CommonSystems as comSys

import Rotations
import LinearAlgebra: I
using BenchmarkTools
import ForwardDiff

# %%
dt = 0.1
est_state = comSys.ImuState(rand(3)..., Rotations.params(rand(Rotations.UnitQuaternion))..., rand(9)...)
input = comSys.ImuInput(rand(6)...)
measurement = comSys.ViconMeasure(rand(3)..., Rotations.params(rand(Rotations.UnitQuaternion))...)

est_cov = Matrix(2.2 * I(length(comSys.ImuError)))
process_cov = Matrix(0.5 * I(length(comSys.ImuError)))
measure_cov = Matrix(0.005 * I(length(comSys.ViconError)))

ekf = EKF.ErrorStateFilter{comSys.ImuState, comSys.ImuError, comSys.ImuInput,
                           comSys.ViconMeasure, comSys.ViconError}(est_state,
                                                                   est_cov,
                                                                   process_cov,
                                                                   measure_cov)

# %%
b = @benchmark EKF.process($est_state, $input, $dt)
display(b)

# %%
b = @benchmark EKF.error_process_jacobian($est_state, $input, $dt)
display(b)

# %%
b = @benchmark EKF.prediction!($ekf, $input, $dt)
display(b)

# %%
b = @benchmark EKF.update!($ekf, $measurement)
display(b)
