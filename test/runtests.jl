using AssemblyCalculus
using AssemblyCalculus: Neuron, NeuralArea, empty_synapses, num_synapses
using Test

findparam(n::Neuron{T, U}) where {T, U} = T

@testset "Neuron" begin
    @testset "Constructors" begin
        neuronf64 = Neuron()
        neuronf32 = Neuron(Float32)
        @test findparam(neuronf64) == Float64
        @test findparam(neuronf32) == Float32
        neuronf64 = Neuron(NeuralArea(), 0, empty_synapses(Float64))
        T = Float32
        neuronf32 = Neuron(NeuralArea(T), 0, empty_synapses(T))
        @test findparam(neuronf64) == Float64
        @test findparam(neuronf32) == Float32
    end
    @testset "Utilities" begin
        T = Float32
        area1 = NeuralArea(T)
        neuron1 = Neuron(area1, 1, empty_synapses(T))
        push!(area1.neurons, neuron1)
        area2 = NeuralArea(T)
        neuron2 = Neuron(area2, 1, empty_synapses(T))
        push!(area2.neurons, neuron2)
        neuron1[area2, neuron2] = 1.5f8
        neuron2[area1, neuron1] = 1.7f8
        @test neuron1[area2][neuron2] == 1.5f8
        @test neuron2[area1][neuron1] == 1.7f8
        @test get(neuron1, area1, true)
        @test length(get!(neuron1, area2, [])) == 1
        @test length(get!(neuron1, area1, Dict{Neuron{T, NeuralArea{T}}, T}())) == 0
        @test length(collect(keys(neuron1))) == 2
        @test num_synapses(neuron1, area1) == 0
        @test num_synapses(neuron1, area2) == 1
        @test num_synapses(neuron2, area1) == 1
        @test num_synapses(neuron2, area2) == 0
        @test num_synapses(neuron1) == 1
        @test num_synapses(Neuron()) == 0
    end
end

@testset "NeuralArea" begin

    @testset "Constructors" begin

    end
    @testset "Type Utilities" begin
    end
end

@testset "IonCurrent Type" begin
    @testset "Constructors" begin
    end
    @testset "Type Utilities" begin
    end
end

@testset "Assembly" begin
    @testset "Constructors" begin
    end
    @testset "Type Utilities" begin
    end
end

@testset "BrainAreas" begin
    @testset "Constructors" begin
    end
    @testset "Type Utilities" begin
    end
end
