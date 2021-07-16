###############################################################################
#                        Abstract Filter Types
###############################################################################
abstract type State{N, T} <: FieldVector{N, T} end 
abstract type ErrorState{Nₑ, T} <: FieldVector{Nₑ, T} end 

abstract type Input{M, T} <: FieldVector{M, T} end  

abstract type Measurement{S, T} <: FieldVector{S, T} end  
abstract type ErrorMeasurement{Sₑ, T} <: FieldVector{Sₑ, T} end  

###############################################################################
#                      Process / Process Jacobian
###############################################################################
function process end

function error_process_jacobian end


###############################################################################
#                       Measure / Measurement Jacobian
###############################################################################
function measure end

function error_measure_jacobian end

###############################################################################
#                 State/Measurement & Composition/Difference
###############################################################################
# Add an error state to another state to create a new state
function state_composition end

# Compute the error measurement between two measurement
function measurement_error end
