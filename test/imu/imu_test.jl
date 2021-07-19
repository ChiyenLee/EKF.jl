using EKF
using StaticArrays: length
using LinearAlgebra: I 

include("$(@__DIR__)/imu_states.jl")

using DataFrames: DataFrame, sort!, outerjoin
using CSV

# %%
imu_df = DataFrame(CSV.File("$(@__DIR__)/../data/imu_first_success.csv"))
vicon_df = DataFrame(CSV.File("$(@__DIR__)/../data/vicon_first_success.csv"))

# %%
est_state = ImuState(rand(16)...)
est_cov = Matrix(2.2 * I(length(ImuErrorState)))
process_cov = Matrix(2.2 * I(length(ImuErrorState)))
measure_cov = Matrix(2.2 * I(length(ViconErrorMeasurement)))

# %%
ekf = ErrorStateFilter{ImuState, ImuErrorState, ImuInput, ViconMeasurement, 
                       ViconErrorMeasurement}(est_state, est_cov, process_cov, measure_cov)
input = zeros(ImuInput)
measurement = ViconMeasurement(rand(3)..., params(ones(UnitQuaternion))...)

println("Inital State: ", est_state)

# %%
filtered_data = Vector{ImuState}()
vicon_data = Vector{ViconMeasurement}()
lastTime = imu_df[1, :time]

for imu_row in eachrow(imu_df[2:end, :])
    t = imu_row[:time]
    dt = t - lastTime

    vicon_row = vicon_df[argmin(abs.(vicon_df[!, :time] .- t)), :]
    input = ImuInput(imu_row[[:acc_x, :acc_y, :acc_z, :gyr_x, :gyr_y, :gyr_z]]...)
    measurement = ViconMeasurement(vicon_row[[:pos_x, :pos_y, :pos_z, :quat_w, :quat_x, :quat_y, :quat_z]]...)
    push!(vicon_data, copy(measurement))

    state, _ = estimateState!(ekf, input, measurement, dt);
    push!(filtered_data, copy(state))

    lastTime = t
end

println("Final State: ", ekf.est_state)

# %%
using Plots
plotly()

p1 = plot([vicon_data[i].p𝑥 for i in 1:length(filtered_data)], linewidth = 2)
plot!(p1, [filtered_data[i].p𝑥 for i in 1:length(filtered_data)])