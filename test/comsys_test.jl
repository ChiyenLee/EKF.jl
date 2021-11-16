using Revise
import EKF
import EKF.CommonSystems as ComSys
using StaticArrays

@testset "Common Systems Performance Evaluations" begin

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

            # Performance
            b = @benchmark EKF.state_composition($state, $stateErr)
            @test maximum(b.gctimes) == 0  # no garbage collection
            @test b.memory == 0            # no dynamic memory allocations

            # Performance
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
    state = ComSys.ImuState(zeros(3)..., [1.,0,0,0]..., zeros(9)...)
    stateErr = ComSys.ImuError(zeros(15)...)
    input = ComSys.ImuInput(zeros(6))
    meas = ComSys.ViconMeasure(zeros(3)..., [1.,0,0,0]...)
    measErr = ComSys.ViconError(zeros(6))

    performance_test("IMU/Vicon Filter", state, stateErr, input, meas, measErr)
end
