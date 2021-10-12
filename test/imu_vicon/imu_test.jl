using Pkg
if Base.active_project() != "$(dirname(dirname(@__FILE__)))/Project.toml"
    Pkg.activate("$(dirname(dirname(@__FILE__)))")
end

# %%
# %%
include("../../src/EKF.jl")
import .EKF

# %%
using LinearAlgebra: I
using BenchmarkTools
using DataFrames: DataFrame, sort!, outerjoin, nrow
using CSV

using StaticArrays
using ForwardDiff: jacobian
using Rotations: UnitQuaternion, RotationError, CayleyMap, add_error
using Rotations: rotation_error, params, ∇differential, kinematics

include("$(@__DIR__)/imu_states.jl")

# %%
function main()
    imu_df = DataFrame(CSV.File("$(@__DIR__)/../data/imu_first_success.csv"))
    vicon_df = DataFrame(CSV.File("$(@__DIR__)/../data/vicon_first_success.csv"))

    filter_data = Vector{ImuState}()
    vicon_data = Vector{Vicon}()

    # %%
    est_state = ImuState(rand(16)...)
    est_cov = Matrix(2.2 * I(length(ImuError)))
    process_cov = Matrix(2.2 * I(length(ImuError)))
    measure_cov = Matrix(2.2 * I(length(ViconError)))

    ekf = ErrorStateFilter{ImuState, ImuError, ImuInput, Vicon,
                           ViconError}(est_state, est_cov, process_cov, measure_cov)
    input = zeros(ImuInput)
    measurement = Vicon(rand(3)..., params(ones(UnitQuaternion))...)
    lastTime = imu_df[1, :time]

    for imu_row in eachrow(imu_df[2:end, :])
        t = imu_row[:time]
        dt = t - lastTime

        vicon_row = vicon_df[argmin(abs.(vicon_df[!, :time] .- t)), :]

        input = ImuInput(imu_row[:acc_x], imu_row[:acc_y], imu_row[:acc_z],
                         imu_row[:gyr_x], imu_row[:gyr_y], imu_row[:gyr_z])
        measurement = Vicon(vicon_row[:pos_x], vicon_row[:pos_y], vicon_row[:pos_z],
                            vicon_row[:quat_w], vicon_row[:quat_x], vicon_row[:quat_y], vicon_row[:quat_z])

        estimateState!(ekf, input, measurement, dt)
        push!(filter_data, ImuState(ekf.est_state...))
        push!(vicon_data, measurement)

        lastTime = t
    end

    return filter_data, vicon_data
end

filter_data, vicon_data = main()

using Plots

plot([filter_data[i].p𝑥 for i in 1:length(filter_data)])
plot!([vicon_data[i].p𝑥 for i in 1:length(vicon_data)])