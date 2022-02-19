using AssemblyCalculus: NeuralArea, neuron_attrib_graph, hebb_update!
using AssemblyCalculus: fire!
using Graphs: adjacency_matrix

@testset "Hebbian Updates" begin
    # Make an area
    num_neurons = 10
    T = Float32
    area1 = NeuralArea(num_neurons, 3, 1.f0)
    area2 = NeuralArea(num_neurons, 3, 1.f0)
    current = ones(T, 10)
    ag = neuron_attrib_graph(T, NeuralArea{T})
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
    ag = neuron_attrib_graph(T, NeuralArea{T})
    ic_currents = zeros(T, 10)
    ic_currents[1:5] .= init_currents[1:5]
    ic = Stimulus(area2, ic_currents)
    hebb_update!(ag, ic, currents, init_currents)
    @test all(ic.currents .== currents)

    ic_currents = zeros(T, 10)
    ic_currents[1:3] .= init_currents[1:3]
    ic = Stimulus(area2, ic_currents)
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
    areas = [NeuralArea(num_neurons, 3, 1.f0) for i in 1:3]
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
    assem = Assembly(a2, Stimulus{T}[], Assembly{T}[])
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

@testset "Full Simulation" begin
# From scratch simulation
T = Float32
adj = T.([
    0 1 0 0 1 0;
    1 0 0 0 0 1;
    0 0 0 1 0 0;
    0 0 1 0 0 0;
    0 0 0 0 0 1;
    0 0 0 0 1 0;
])

plasticities = [1f0, 1f0, 2f0]
assem_current = [1f0, 1f0, 0f0, 0f0]
ion_current = [1f0, 1f0, 0f0, 0f0]
max_iters = 5
firing = [1, 0, 1, 0]
true_spikes = [ones(1, 1), ones(1, 1)]
init_currents = ion_current + assem_current
currents = deepcopy(init_currents)
true_adj = deepcopy(adj)
sub = true_adj[1:4, 1:4]
for i in 1:max_iters
    total_current = sub * firing + currents
    firing_next = zeros(4)
    spike1 = argmax(total_current[1:2])
    spike2 = argmax(total_current[3:4])
    firing_next[spike1] = 1
    firing_next[spike2 + 2] = 1
    sub[firing_next * transpose(firing) .== 1] .*= (1 + plasticities[1])
    currents[firing_next .== 1] .*= (1 + plasticities[1])
    firing = firing_next
    true_spikes = map(hcat, true_spikes, [[spike1], [spike2]])
end
# Adj update
true_adj[1:4, 1:4] .= sub
# Ion current update
nz = ion_current .!= 0
times_grown = log.(currents[nz] ./ init_currents[nz]) ./ log(2f0)
true_ion_current = zeros(4)
true_ion_current[nz] = (1 + plasticities[1]) .^ times_grown
# Assembly edge update
true_adj[1, 5] *= (1f0 + plasticities[3]) ^ times_grown[1]
true_adj[2, 6] *= (1f0 + plasticities[3]) ^ times_grown[2]

true_distances = vcat(zeros(T, 1, 6), ones(T, 1, 6))
true_distances[1, 1:2] .= 1

# Run sim
area_sizes = [2, 2, 2]
assembly_sizes =[1, 1, 2]
plasticities = [1f0, 1f0, 2f0]
ba = Brain(adj, area_sizes, assembly_sizes, plasticities)
# Initial firing
a1, a2, a3 = ba.areas
a1.firing = [1]
a2.firing = [1]
a3.firing = [1, 2]
# IonCurrents and Assemblies
inputs = [Stimulus(a1, ones(T, 2)), zero_stim(a2)]
assemblies = [Assembly(a3)]
# Simulate
new_assems, spikes, distances = simulate!(
    inputs,
    assemblies,
    tol=0,
    max_iters=5,
    random_initial=false
    )
# Test
@test all(true_adj .== adjacency_matrix(ba))
@test all([all(true_spikes[i] .== spikes[i]) for i in 1:2])
@test all(true_distances .== distances)
@test all([new_assems[i].neurons[1] == true_spikes[i][end] for i in 1:2])

# Rerun with MaxIters stop criteria
timesteps = size(distances)[2] - 1
ba = Brain(adj, area_sizes, assembly_sizes, plasticities)
# Initial firing
a1, a2, a3 = ba.areas
a1.firing = [1]
a2.firing = [1]
a3.firing = [1, 2]
# IonCurrents and Assemblies
inputs = [Stimulus(a1, ones(T, 2)), zero_stim(a2)]
assemblies = [Assembly(a3)]
new_assems, spikes = simulate!(inputs, assemblies, timesteps, random_initial=false)
@test all(true_adj .== adjacency_matrix(ba))
@test all([all(true_spikes[i] .== spikes[i]) for i in 1:2])
@test all([new_assems[i].neurons[1] == true_spikes[i][end] for i in 1:2])

# Simulate w/o recording
ba = Brain(adj, area_sizes, assembly_sizes, plasticities)
# Initial firing
a1, a2, a3 = ba.areas
a1.firing = [1]
a2.firing = [1]
a3.firing = [1, 2]
# IonCurrents and Assemblies
inputs = [Stimulus(a1, ones(T, 2)), zero_stim(a2)]
assemblies = [Assembly(a3)]
new_assems = simulate!(
    inputs,
    assemblies,
    tol=0,
    max_iters=5,
    random_initial=false,
    record=false
    )
@test all(true_adj .== adjacency_matrix(ba))
@test all([new_assems[i].neurons[1] == true_spikes[i][end] for i in 1:2])

end
