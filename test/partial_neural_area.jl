using AssemblyCalculus: PartialNeuralArea, generate_synapses

@testset "Constructors" begin
    pna = PartialNeuralArea(Float32)
    @test length(pna) == 0
    @test num_neurons(pna) == 0
    pna = PartialNeuralArea(10, 3, 0.01f0, 0.2f0)
    @test length(pna) == 10
    @test num_neurons(pna) == 10
end

@testset "Utilities" begin
    T = Float32
    area1 = PartialNeuralArea(T)
    neuron1 = Neuron(area1, 1)
    push!(area1.neurons, neuron1)
    area2 = PartialNeuralArea(T)
    neuron2 = Neuron(area2, 1)
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
        push!(area1.neurons, Neuron(area1, i))
    end
    currents = zeros(Float32, 10)
    @test_throws ErrorException winners!(area1, currents)
    push!(area1.neurons, Neuron(area1, 10))
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

@testset "Hebbian Updates" begin
    # Make an area
    num_neurons = 10
    T = Float32
    area1 = PartialNeuralArea(num_neurons, 3, 1.f0, 0f0)
    area2 = PartialNeuralArea(num_neurons, 3, 1.f0, 0f0)
    current = ones(T, 10)
    ag = neuron_attrib_graph(T, PartialNeuralArea{T})
    # Make synapses
    for i in 1:num_neurons
        source = area1[i]
        target = area2[i]
        source[area2, target] = 1.f0
        if i > (num_neurons / 2)
            contrib = get!(ag, target)
            push!(contrib, source)
        end
    end

    # Ion current array update
    hebb_update!(area1, current)
    @test all(current .== 1.f0)
    random_firing!(area1)
    hebb_update!(area1, current)
    @test length(findall(current .> 1f0)) == 3
    area2.firing =[1, 9, 10]
    # Synapse Update
    hebb_update!(area2, ag)
    syn = map(x -> collect(values(x[area2]))[1], area1)
    @test all(syn[1:8] .== 1f0)
    @test all(syn[9:10] .== 2f0)

    init_currents = zeros(T, 10)
    init_currents[1:5] .= rand(T, 5)
    currents = zeros(T, 10)
    pows = [1., 2, 4, 1, 8]
    currents[1:5] .= init_currents[1:5] .* (2f0 .^ pows)
    ag = neuron_attrib_graph(T, PartialNeuralArea{T})
    ic_currents = zeros(T, 10)
    ic_currents[1:5] .= init_currents[1:5]
    ic = IonCurrent(area2, ic_currents)
    hebb_update!(ag, ic, currents, init_currents)
    @test all(ic.currents .== currents)

    ic_currents = zeros(T, 10)
    ic_currents[1:3] .= init_currents[1:3]
    ic = IonCurrent(area2, ic_currents)
    for i in 1:3
        contrib = get!(ag, area2[i])
        push!(contrib, area1[i])
    end
    hebb_update!(ag, ic, currents, init_currents)

    @test all(ic.currents[1:3] .== currents[1:3])
    @test all(ic.currents[4:5] .!= currents[4:5])
    syn = map(x -> collect(values(x[area2]))[1], area1)
    @test all(syn[1:3] .== (2f0 .^ pows[1:3]))
    @test all(syn[4:5] .== 1f0)
end

@testset "Firing" begin
    T = Float32
    num_neurons = 10
    areas = [PartialNeuralArea(num_neurons, 3, 1.f0, 0f0) for i in 1:3]
    a1, a2, a3 = areas
    # Make synapses
    for i in 1:num_neurons
        source = a1[i]
        target = a2[i]
        target2 = a3[i]
        source[a2, target] = 1.1f0
        source[a3, target2] = 1.2f0
        target[a3, target2] = 1.3f0
    end

    # Assembly firing
    currents = zeros(Float32, num_neurons)
    random_firing!.(areas)
    assem = Assembly(a2, IonCurrent{T}[], Assembly{T}[])
    ag = neuron_attrib_graph(Float32, typeof(a2))
    fire!(assem, a1, currents, ag)
    @test all(currents .== 0f0)
    @test length(ag) == 0
    fire!(assem, a3, currents, ag)
    @test all(currents[assem.neurons] .== 1.3f0)
    currents[assem.neurons] .= 0f0
    @test all(currents .== 0f0)
    @test length(ag) .== assem.area.assembly_size
    @test all([length(v) for v in values(ag)] .== 1)

    # Active Area firing
    active_areas = [a1, a3]
    area_currents  = [zeros(T, num_neurons) for i in 1:2]
    ag = fire!(active_areas, area_currents)
    @test all(area_currents[1] .== 0f0)
    @test all(area_currents[2][a1.firing] .== 1.2f0)
    @test length(ag) == a1.assembly_size
    @test all([length(v) for v in values(ag)] .== 1)
    @test length(intersect(collect(keys(ag)), a3.neurons)) == a1.assembly_size
end

@testset "Simulation" begin
    k = 100
    n = k ^ 2
    β = 1f0
    mean_degree = 100f0
    p = mean_degree / n

    ba = BrainAreas(1, n, k, β, p)
    random_firing!(ba[1])
    init_firing = ba[1].firing
    stims = [random_current(ba[1])]
    init_currents = stims[1].currents
    assems = Assembly{Float32}[]
    new_assems, spikes, dists = simulate!(
        stims,
        assems,
        tol=0,
        fire_past_convergence=10,
        random_initial=false
    )
    @test length(new_assems) == 1
    @test all([length(a.neurons) for a in assems] .== n)


    adj = adjacency_matrix(ba)
    adj[adj .!= 0f0] .= 1.f0
    ba2 = BrainAreas(adj, [n], [k], [β])
    ba2[1].firing = init_firing
    stims2 = [IonCurrent(ba2[1], init_currents)]
    new_assems2, spikes2, dists2 = simulate!(
        stims2,
        assems,
        tol=0,
        fire_past_convergence=10,
        random_initial=false
    )
    @test all(dists2 .== dists)

end
