using Revise
import EKF
import EKF.CommonSystems as ComSys
using StaticArrays
using Test
using BenchmarkTools

@testset "Common Systems Performance Evaluations" begin

    # Note this function only checks the performance of the Common System functions,
    # not the physical accuracy of the process etc.
    function performance_test(
        name::String,
        state::EKF.State,
        stateErr::EKF.ErrorState,
        input::EKF.Input,
        meas::M,
        measErr::EKF.ErrorMeasurement
    ) where {M <: EKF.Measurement}
        @testset "$name" begin
            dt = 0.1

            b = @benchmark EKF.state_composition($state, $stateErr)
            @test maximum(b.gctimes) == 0  # no garbage collection
            @test b.memory == 0            # no dynamic memory allocations

            b = @benchmark EKF.measurement_error($meas, $meas)
            @test maximum(b.gctimes) == 0  # no garbage collection
            @test b.memory == 0            # no dynamic memory allocations

            b = @benchmark EKF.process($state, $input, $dt)
            @test maximum(b.gctimes) == 0  # no garbage collection
            @test b.memory == 0            # no dynamic memory allocations

            b = @benchmark EKF.error_process_jacobian($state, $input, $dt)
            @test maximum(b.gctimes) == 0  # no garbage collection
            @test b.memory == 0            # no dynamic memory allocations

            b = @benchmark EKF.measure($M, $state)
            @test maximum(b.gctimes) == 0  # no garbage collection
            @test b.memory == 0            # no dynamic memory allocations

            b = @benchmark EKF.error_measure_jacobian($M, $state)
            @test maximum(b.gctimes) == 0  # no garbage collection
            @test b.memory == 0            # no dynamic memory allocations
        end
    end

    # Test Performance of IMU Vicon Filter
    state = ComSys.ImuState{Float32}(zeros(3)..., [1.,0,0,0]..., zeros(9)...)
    stateErr = ComSys.ImuError{Float32}(zeros(15)...)
    input = ComSys.ImuInput{Float32}(zeros(6))
    meas = ComSys.ViconMeasure{Float32}(zeros(3)..., [1.,0,0,0]...)
    measErr = ComSys.ViconError{Float32}(zeros(6))

    performance_test("IMU/Vicon Filter: Float32", state, stateErr, input, meas, measErr)

    state = ComSys.ImuState{Float64}(zeros(3)..., [1.,0,0,0]..., zeros(9)...)
    stateErr = ComSys.ImuError{Float64}(zeros(15)...)
    input = ComSys.ImuInput{Float64}(zeros(6))
    meas = ComSys.ViconMeasure{Float64}(zeros(3)..., [1.,0,0,0]...)
    measErr = ComSys.ViconError{Float64}(zeros(6))

    performance_test("IMU/Vicon Filter: Float64", state, stateErr, input, meas, measErr)

end
