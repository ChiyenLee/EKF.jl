#
using Pkg
Pkg.activate("$(@__DIR__)/..")

using Revise
using EKF
using StaticArrays: length
using LinearAlgebra: I
using BenchmarkTools


include("$(@__DIR__)/imu_states.jl")

using DataFrames: DataFrame, sort!, outerjoin, nrow
using CSV

function main()
    imu_df = DataFrame(CSV.File("$(@__DIR__)/../data/imu_first_success.csv"))
    vicon_df = DataFrame(CSV.File("$(@__DIR__)/../data/vicon_first_success.csv"))

    # %%
    est_state = ImuState(rand(16)...)
    est_cov = SMatrix{length(ImuErrorState), length(ImuErrorState)}(2.2 * I(length(ImuErrorState)))
    process_cov = SMatrix{length(ImuErrorState), length(ImuErrorState)}(2.2 * I(length(ImuErrorState)))
    measure_cov = SMatrix{length(ViconErrorMeasurement), length(ViconErrorMeasurement)}(2.2 * I(length(ViconErrorMeasurement)))

    ekf = ErrorStateFilter{ImuState, ImuErrorState, ImuInput, ViconMeasurement,
                           ViconErrorMeasurement}(est_state, est_cov, process_cov, measure_cov)
    input = zeros(ImuInput)
    measurement = ViconMeasurement(rand(3)..., params(ones(UnitQuaternion))...)
    lastTime = imu_df[1, :time]

    for imu_row in eachrow(imu_df[2:end, :])
        t = imu_row[:time]
        dt = t - lastTime

        vicon_row = vicon_df[argmin(abs.(vicon_df[!, :time] .- t)), :]

        input = ImuInput(imu_row[:acc_x], imu_row[:acc_y], imu_row[:acc_z],
                         imu_row[:gyr_x], imu_row[:gyr_y], imu_row[:gyr_z])
        measurement = ViconMeasurement(vicon_row[:pos_x], vicon_row[:pos_y], vicon_row[:pos_z],
                                       vicon_row[:quat_w], vicon_row[:quat_x], vicon_row[:quat_y], vicon_row[:quat_z])

        @btime prediction!($ekf, $input, $dt);
        # @btime estimateState!($ekf, $input, $measurement, $dt);

        lastTime = t
        break
    end
end

main()