using EKF
using StaticArrays: length
using LinearAlgebra: I 

include("$(@__DIR__)/gyro/gyro_states.jl")

using DataFrames: DataFrame, sort!, outerjoin
using CSV

# %%
imu_df = DataFrame(CSV.File("$(@__DIR__)/data/imu_first_success.csv"))[!, [:time, :gyr_x, :gyr_y, :gyr_z]]
vicon_df = DataFrame(CSV.File("$(@__DIR__)/data/vicon_first_success.csv"))[!, [:time, :quat_w, :quat_x, :quat_y, :quat_z]]

# %%
est_state = GyroState(rand(7)...)
est_cov = Matrix(.3 * I(length(GyroErrorState)))
process_cov = Matrix(.3 * I(length(GyroErrorState)))
measure_cov = Matrix(.3 * I(length(QuatErrorMeasurement)))

ekf = ErrorStateFilter{GyroState, GyroErrorState, GyroInput, QuatMeasurement, 
                       QuatErrorMeasurement}(est_state, est_cov, process_cov, measure_cov)
input = zeros(GyroInput)
measurement = QuatMeasurement(params(ones(UnitQuaternion))...)

println("Inital State: ", est_state)

# %%
lastTime = imu_df[1, :time]

for imu_row in eachrow(imu_df[2:end, :])
    time = imu_row[:time]
    dt = time - lastTime

    vicon_row = vicon_df[argmin(vicon_df[!, :time] .- time), :]
    # input = GyroInput(imu_row[[:gyr_x, :gyr_y, :gyr_z]]...)
    # measurement = QuatMeasurement(vicon_row[[:quat_w, :quat_x, :quat_y, :quat_z]]...)

    estimateState!(ekf, input, measurement, .1);

    lastTime = time
end

println("Final State: ", ekf.est_state)