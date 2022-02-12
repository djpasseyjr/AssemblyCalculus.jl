using Test
using AssemblyCalculus

@testset "Types" begin
    include("types.jl")
end

@testset "Simulation Functions" begin
    include("simulation_functions.jl")
end

@testset "PartialNeuralArea" begin
    include("partial_neural_area.jl")
end
