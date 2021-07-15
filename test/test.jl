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
est_state = rand(GyroState)
est_cov = Matrix(.3 * I(length(GyroErrorState)))
process_cov = Matrix(.3 * I(length(GyroErrorState)))
measure_cov = Matrix(.3 * I(length(QuatErrorMeasurement)))

ekf = ErrorStateFilter{GyroState, GyroErrorState, GyroInput, QuatMeasurement, 
                       QuatErrorMeasurement}(est_state, est_cov, process_cov, measure_cov)
input = zeros(GyroInput)
measurement = QuatMeasurement(params(ones(UnitQuaternion))...)

# %%
lastTime = imu_df[1, :time]
for imu_row in eachrow(imu_df[2:end, :])
    time = imu_row[:time]
    
    vicon_row = vicon_df[argmin(vicon_df[!, :time] .- time), :]
    input = GyroInput(imu_row[[:gyr_x, :gyr_y, :gyr_z]]...)
    measurement = QuatMeasurement(vicon_row[[:quat_w, :quat_x, :quat_y, :quat_z]]...)

    
    println(time)
end



# println(prediction(ekf, est_state, est_cov, input; dt=0.1))
# println(estimateState!(ekf, input, measurement, 0.1))