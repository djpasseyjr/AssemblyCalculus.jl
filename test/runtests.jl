using AssemblyCalculus
using AssemblyCalculus: Neuron, empty_synapses, num_synapses
using AssemblyCalculus: NeuralArea, num_neurons, random_firing!, winners!, change_in_assembly
using AssemblyCalculus: IonCurrent, random_current, zero_current
using AssemblyCalculus: getneuron, globalindex, select, neuron_attrib_graph
using Graphs
using StatsBase
using Test


@testset "Neuron" begin
    findparam(n::Neuron{T, U}) where {T, U} = T
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
    findparam(na::NeuralArea{T}) where T = T
    @testset "Constructors" begin
        na = NeuralArea()
        @test findparam(na) == Float64
        @test findparam(NeuralArea(Float32)) == Float32
        @test findparam(NeuralArea(100, 10, 1.0)) == Float64
        @test findparam(NeuralArea(100, 10, 1.0f8)) == Float32
        na = NeuralArea()
        @test num_neurons(na) == 0
        push!(na.neurons, Neuron())
        @test num_neurons(na) == 1
        @test_throws UndefVarError push!(na.neurons, Neurons(Float32))
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
        @test num_synapses(area1) == 1
        @test num_synapses(area2) == 1
        for n in area1
            @test num_synapses(n) == 1
        end
        @test length(area1.firing) == 0
        area1.assembly_size = 1
        random_firing!(area1)
        @test length(area1.firing) == 1
        area1.assembly_size = 2
        @test_throws ErrorException random_firing!(area1)
        # k-winners
        area1.assembly_size = 3
        @test_throws ErrorException winners!(area1, rand(Float32, 3))
        for i in 2:9
            push!(area1.neurons, Neuron(area1, i, empty_synapses(Float32)))
        end
        currents = zeros(Float32, 10)
        @test_throws ErrorException winners!(area1, currents)
        push!(area1.neurons, Neuron(area1, 10, empty_synapses(Float32)))
        winners!(area1, currents)
        @test length(area1.firing) == 0
        currents[2:4] .= 1.0f8
        winners!(area1, currents)
        @test all(area1.firing .== [2, 3, 4])
        currents = rand(Float32, 10)
        perm = sortperm(currents, rev=true)
        winners!(area1, currents)
        @test all(area1.firing .== perm[1:3])
        # Prev assembly comparison
        area1.firing = [1, 2, 3]
        area1.firing_prev = [1, 2, 3]
        @test change_in_assembly(area1) == 0
        area1.firing = [1, 2, 4]
        @test change_in_assembly(area1) == 1
    end
end

@testset "IonCurrent Type" begin
    getparam(ic::IonCurrent{T}) where T = T
    @testset "Constructors" begin
        @test getparam(IonCurrent()) == Float64
        @test getparam(IonCurrent(Float32)) == Float32
        area1 = NeuralArea(4000, 20, 0.01)
        # One edge per neuron
        for neuron in area1
            neuron[area1, neuron] = 1.0
        end
        ic = random_current(area1)
        @test ic.area == area1
        @test length(ic.currents) == length(area1)
        @test 0.001 < mean(ic.currents) < 0.01
        ic = random_current(area1, p=1. / 400)
        @test ic.area == area1
        @test length(ic.currents) == length(area1)
        @test 0.01 < mean(ic.currents) < 0.1
        @test sum(zero_current(area1).currents) == 0
        @test_throws ErrorException random_current(area1, p=1.0)
    end
end

@testset "Assembly" begin
    getparam(a::Assembly{T}) where T = T
    @testset "Constructors" begin
        @test getparam(Assembly()) == Float64
        @test getparam(Assembly(Float32)) == Float32
        area1 = NeuralArea(100, 10, 0.01)
        ic = zero_current(area1)
        @test_throws ErrorException Assembly(area1, [ic], Assembly{Float64}[])
        random_firing!(area1)
        a = Assembly(area1, [ic], Assembly{Float64}[])
        @test a.area == area1
        @test length(a.parent_currents) == 1
        @test length(a.parent_assemblies) == 0
        @test length(a.neurons) == area1.assembly_size
    end
end

@testset "BrainAreas" begin
    g = erdos_renyi(10, 1.0, is_directed=true)
    sizes = [3, 3, 4]
    assemblies = [1, 1, 2]
    plasticities = [.01f8, .01f8, .01f8]
    ba = BrainAreas(g, sizes, assemblies, plasticities)
    @testset "Constructors" begin
        @test length(ba) == 3
        @test typeof(ba[2]) == NeuralArea{Float32}
        @test num_synapses(ba) == 90
        @test num_neurons(ba) == 10
        adj = Float32.(rand(10, 10) .< 0.2)
        ba = BrainAreas(adj, sizes, assemblies, plasticities)
        @test all(adj .== adjacency_matrix(ba))
    end
    @testset "Type Utilities" begin
        g_idx = 5
        @test globalindex(ba, getneuron(ba, g_idx)) == g_idx
        @test select(ba, 1) == ba.areas[1]
        @test_throws ErrorException select(ba, 4)
    end
end

@testset "AtributionGraph" begin
    @testset "Constructors" begin
        T = Float32
        n1, n2, n3 = Neuron(T), Neuron(T), Neuron(T)
        ag = neuron_attrib_graph(T)
        contributors = get!(ag, n1)
        push!(contributors, n2)
        contributors = get!(ag, n1)
        push!(contributors, n3)
        @test all(ag[n1] .== [n2, n3])

    end
end
