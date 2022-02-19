using AssemblyCalculus: Neuron, empty_synapses, num_synapses
using AssemblyCalculus: NeuralArea, num_neurons, random_firing!, winners!, change_in_assembly
using AssemblyCalculus: Stimulus, rand_stim, zero_stim
using AssemblyCalculus: getneuron, globalindex, select, neuron_attrib_graph, freeze_synapses!
using AssemblyCalculus: MaxIters, Converge
using Graphs
using StatsBase


@testset "Neuron" begin
    findparam(n::Neuron{T, U}) where {T, U} = T
    @testset "Constructors" begin
        neuronf64 = Neuron(Float64)
        neuronf32 = Neuron(Float32)
        @test findparam(neuronf64) == Float64
        @test findparam(neuronf32) == Float32
        neuronf64 = Neuron(NeuralArea(Float64), 0, empty_synapses(Float64))
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
        neuron1[area2, neuron2] = 1.5f0
        neuron2[area1, neuron1] = 1.7f0
        @test neuron1[area2][neuron2] == 1.5f0
        @test neuron2[area1][neuron1] == 1.7f0
        @test get(neuron1, area1, true)
        @test length(get!(neuron1, area2, [])) == 1
        @test length(get!(neuron1, area1, Dict{Neuron{T, NeuralArea{T}}, T}())) == 0
        @test length(collect(keys(neuron1))) == 2
        @test num_synapses(neuron1, area1) == 0
        @test num_synapses(neuron1, area2) == 1
        @test num_synapses(neuron2, area1) == 1
        @test num_synapses(neuron2, area2) == 0
        @test num_synapses(neuron1) == 1
        @test num_synapses(Neuron(Float64)) == 0
    end
end

@testset "NeuralArea" begin
    findparam(na::NeuralArea{T}) where T = T
    @testset "Constructors" begin
        na = NeuralArea(Float64)
        @test findparam(na) == Float64
        @test findparam(NeuralArea(Float32)) == Float32
        @test findparam(NeuralArea(100, 10, 1.0)) == Float64
        @test findparam(NeuralArea(100, 10, 1.0f0)) == Float32
        na = NeuralArea(Float64)
        @test num_neurons(na) == 0
        push!(na.neurons, Neuron(Float64))
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
        neuron1[area2, neuron2] = 1.5f0
        neuron2[area1, neuron1] = 1.7f0
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
        currents[2:4] .= 1.0f0
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

@testset "Stimulus Type" begin
    getparam(ic::Stimulus{T}) where T = T
    @testset "Constructors" begin
        @test getparam(Stimulus()) == Float64
        @test getparam(Stimulus(Float32)) == Float32
        area1 = NeuralArea(4000, 20, 0.01)
        # One edge per neuron
        for neuron in area1
            neuron[area1, neuron] = 1.0
        end
        ic = rand_stim(area1)
        @test ic.area == area1
        @test length(ic.currents) == length(area1)
        @test 0.001 < mean(ic.currents) < 0.01
        ic = rand_stim(area1, p=1. / 400)
        @test ic.area == area1
        @test length(ic.currents) == length(area1)
        @test 0.01 < mean(ic.currents) < 0.1
        @test sum(zero_stim(area1).currents) == 0
        @test_throws ErrorException rand_stim(area1, p=1.0)
    end
end

@testset "Assembly" begin
    getparam(a::Assembly{T}) where T = T
    @testset "Constructors" begin
        @test getparam(Assembly()) == Float64
        @test getparam(Assembly(Float32)) == Float32
        area1 = NeuralArea(100, 10, 0.01)
        ic = zero_stim(area1)
        @test_throws ErrorException Assembly(area1, [ic], Assembly{Float64}[])
        random_firing!(area1)
        a = Assembly(area1, [ic], Assembly{Float64}[])
        @test a.area == area1
        @test length(a.parent_stims) == 1
        @test length(a.parent_assemblies) == 0
        @test length(a.neurons) == area1.assembly_size
    end
end

@testset "Brain" begin
    g = erdos_renyi(10, 1.0, is_directed=true)
    sizes = [3, 3, 4]
    assemblies = [1, 1, 2]
    plasticities = [.01f0, .01f0, .01f0]
    ba = Brain(g, sizes, assemblies, plasticities)
    @testset "Constructors" begin
        @test length(ba) == 3
        @test typeof(ba[2]) == NeuralArea{Float32}
        @test num_synapses(ba) == 90
        @test num_neurons(ba) == 10
        adj = Float32.(rand(10, 10) .< 0.2)
        ba = Brain(adj, sizes, assemblies, plasticities)
        @test all(adj .== adjacency_matrix(ba))
        freeze_synapses!(ba)
        @test all([a.plasticity for a in ba.areas] .== 0)
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
        ag = neuron_attrib_graph(T, NeuralArea{T})
        contributors = get!(ag, n1)
        push!(contributors, n2)
        contributors = get!(ag, n1)
        push!(contributors, n3)
        @test all(ag[n1] .== [n2, n3])
        @test all(get(ag, n2, [n1]) .== [n1])

    end
end

@testset "StopCriteria" begin
    areas = [NeuralArea(100, 10, 0.1) for i in 1:3]
    currents = [zeros(100) for i in 1:3]
    random_firing!.(areas)

    @testset "MaxIters" begin
        iters = 9
        mi = MaxIters(max_iters=iters, record=true)
        i = 0
        while ! mi(areas, currents)
            i += 1
        end
        @test i == iters
        @test length(mi.spikes) == 3
        @test all([all(size(s) .== (10, iters + 1)) for s in mi.spikes])
        # Without recording
        mi = MaxIters(max_iters=iters, record=false)
        i = 0
        while ! mi(areas, currents)
            i += 1
        end
        @test i == iters
        @test length(mi.spikes) == 1
        @test all(size(mi.spikes[1]) .== (1, 1))
    end

    @testset "Converge" begin
        iters = 9
        cv = Converge(max_iters=iters, record=true)
        i = 0
        while ! cv(areas, currents)
            i += 1
        end
        @test i == iters
        @test length(cv.spikes) == 3
        @test all([all(size(s) .== (10, iters + 1)) for s in cv.spikes])
        @test all(size(cv.distances) .== (3, iters + 1))
        @test all(cv.distances .== 10)

        # Without recording
        cv = Converge(max_iters=iters, record=false)
        i = 0
        while ! cv(areas, currents)
            i += 1
        end
        @test i == iters
        @test length(cv.spikes) == 1
        @test all(size(cv.spikes[1]) .== (1, 1))

        iters = 20
        tol = 0
        fire_past_convergence = 0
        random_firing!.(areas)
        firing = [a.firing for a in areas]
        cv = Converge(
            tol=tol,
            fire_past_convergence=fire_past_convergence,
            max_iters=iters, record=true)
        i = 0
        while ! cv(areas, currents)
            k = i < 10 ? i + 1 : 10
            for (j, a) in enumerate(areas)
                a.firing_prev = firing[j][1:k]
            end
            i += 1
        end
        @test i == 10 - tol + fire_past_convergence

        iters = 20
        tol = 3
        fire_past_convergence = 0
        random_firing!.(areas)
        firing = [a.firing for a in areas]
        cv = Converge(
            tol=tol,
            fire_past_convergence=fire_past_convergence,
            max_iters=iters, record=true)
        i = 0
        while ! cv(areas, currents)
            k = i < 10 ? i + 1 : 10
            for (j, a) in enumerate(areas)
                a.firing_prev = firing[j][1:k]
            end
            i += 1
        end
        @test i == 10 - tol + fire_past_convergence

        iters = 20
        tol = 3
        fire_past_convergence = 5
        random_firing!.(areas)
        firing = [a.firing for a in areas]
        cv = Converge(
            tol=tol,
            fire_past_convergence=fire_past_convergence,
            max_iters=iters, record=true)
        i = 0
        while ! cv(areas, currents)
            k = i < 10 ? i + 1 : 10
            for (j, a) in enumerate(areas)
                a.firing_prev = firing[j][1:k]
            end
            i += 1
        end
        @test i == 10 - tol + fire_past_convergence
    end
end
