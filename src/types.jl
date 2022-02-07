"""Types for fast Assembly Calculus simulations.

Author: DJ Passey
Email: djpasseyjr@gmail.com
"""

mutable struct Neuron{T, U}
    area::U
    idx::Int
    synapses::Dict{U, Dict{Neuron{T, U}, T}}
end

"""Returns all synapses in the grouping corresponding to the key `k`."""
Base.getindex(n::Neuron{T, U}, k::U) where {T, U} = n.synapses[k]

"""Returns all synapses in the grouping corresponding to the key `k`. If `k`
is not a valid key, assigns it to map to `default` and return `default`.
"""
Base.get!(n::Neuron{T}, k::U, default) where {T, U} = get!(n.synapses, k, default)

"""Returns all synapses in the grouping corresponding to the key `k`. If `k`
is not a valid key, return `default`.
"""
Base.get(n::Neuron{T}, k::U, default) where {T, U} = get(n.synapses, k, default)

"""Sets the synapse weight of the connection from `source_neuron` to
`target_neuron_in_group` where `target_neuron_in_group` is a member of the
grouping indexed by the key `group_key`.

Defines the assignment operation:

    source_neuron[group_key, target_neuron_in_group] = synapse_weight

If `target_neuron_in_group` is not a member of the group, adds it to the group.
If `group_key` does not map to any groups, creates a new, empty group.
"""
function Base.setindex!(
    source_neuron::Neuron{T, U},
    synapse_weight::T,
    group_key::U,
    target_neuron_in_group::Neuron{T, U}
) where {T, U}
    group = get!(source_neuron, group_key, Dict{Neuron{T, U}, T}())
    group[target_neuron_in_group] = synapse_weight
end

"""Returns the keys for all synapse groupings."""
Base.keys(n::Neuron) = keys(n.synapses)

"""Returns the number of synapses in the synapse group corresponding to key `k`.
"""
function num_synapses(n::Neuron{T, U}, k::U) where {T, U}
    num = get(n, k, Dict{Neuron{T, U}, T}())
    return length(num)
end
"""Returns the total number of synapses eminating from the `Neuron` `n`."""
function num_synapses(n::Neuron{T, U}) where {T, U}
    if length(n.synapses) > 0
        return mapreduce(x -> num_synapses(n, x), +, keys(n))
    else
        return 0
    end
end

"""Constructs an empty synapse dictionary."""
empty_synapses(U::Type, T::Type) = Dict{U, Dict{Neuron{T, U}, T}}()

"""Displays a `Neuron` object."""
function Base.show(io::IO, n::Neuron{T, U}) where {T, U}
    print(io, "Neuron{$T}")
end

## NEURALAREA

mutable struct NeuralArea{T}
    neurons::Array{Neuron{T, NeuralArea{T}}, 1}
    assembly_size::Int
    firing::Array{Int, 1}
    firing_prev::Array{Int, 1}
    plasticity ::T
end

"""Constructs an empty `NeuralArea`."""
NeuralArea(T::Type=Float64) = NeuralArea(Neuron{T, NeuralArea{T}}[], 0, Int[], Int[], T(0))

"""Constructs an empty `NeuralArea` with assembly
size `k` and plasticity `β`."""
NeuralArea(k::Int, β::T) where T = NeuralArea(Neuron{T, NeuralArea{T}}[], k, Int[], Int[], β)

"""Number of neurons in a  `NeuralArea`."""
Base.length(na::NeuralArea) = length(na.neurons)

"""Iterate through neurons in a NeuralArea."""
Base.iterate(na::NeuralArea) = iterate(na.neurons)
Base.iterate(na::NeuralArea, i::Int64) = iterate(na.neurons, i)

"""Retrive neuron at index `i` in the neuron list."""
Base.getindex(na::NeuralArea, i::Int) = na.neurons[i]

num_neurons(na::NeuralArea) = length(na)
num_synapses(na::NeuralArea) = mapreduce(num_synapses, +, na.neurons)

"""Print format for `NeuralArea`."""
function Base.show(io::IO, na::NeuralArea{T}) where T
    descr = "NeuralArea{$T}: $(num_neurons(na)) neurons $(num_synapses(na)) synapses"
    print(io, descr)
end

"""Randomly assigns neurons in the area to fire."""
function random_firing!(area::NeuralArea{T}) where T
    area.firing = sample(1:length(area), area.assembly_size, replace=false)
end

"""Assigns the indexes of the k largest currents in `currents` to `na.firing`."""
function winners!(na::NeuralArea{T}, currents::Array{T}) where T
    na.firing_prev = na.firing
    partial_sort = sortperm(
        currents,
        rev=true,
        alg=PartialQuickSort(na.assembly_size)
    )
    na.firing = partial_sort[1:na.assembly_size]
end

function change_in_assembly(area::NeuralArea{T}) where T
    difference = area.assembly_size - intersect(area.firing, area.firing_prev)
end

"""Constructs empty synapses for a `Neuron` type with `NeuralArea` keys."""
empty_synapses(T::Type) = empty_synapses(NeuralArea{T}, T)

"""Constructs and empty `Neuron` with `NeuralArea` synapse keys."""
function Neuron(T::Type=Float64)
    return Neuron(NeuralArea(T), 0, empty_synapses(T))
end

## IONCURRENT

mutable struct IonCurrent{T}
    area::NeuralArea{T}
    currents::Array{T, 1}
end

IonCurrent(type::Type=Float64) = IonCurrent(NeuralArea(type), type[])

function random_input_current(na::NeuralArea{T}) where T
    num_neurons = length(na)
    p = mapreduce(x -> num_synapses(x, na), +, na.neurons) / (num_neurons ^ 2)
    dist = Binomial(na.assembly_size, p)
    currents = T.(rand(dist, num_neurons))
    return IonCurrent(na, currents)
end

function Base.show(io::IO, ic::IonCurrent{T}) where T
    n = length(ic.currents)
    print(io, "IonCurrent{$T} into $n neurons")
end

## ASSEMBLY

mutable struct Assembly{T}
    area::NeuralArea{T}
    neurons::Array{Int, 1}
    parent_currents::Array{IonCurrent{T}, 1}
    parent_assemblies::Array{Assembly{T}, 1}
end

Assembly(type::Type=Float64) = Assembly(NeuralArea(type), Int[], IonCurrent{type}[], Assembly{type}[])

function Assembly(
    na::NeuralArea{T},
    parent_currents::Array{IonCurrent{T}, 1},
    parent_assemblies::Array{Assembly{T}, 1},
) where T
    return Assembly(na, na.firing, parent_currents, parent_assemblies)
end

function Base.show(io::IO, assem::Assembly{T}) where T
    k = length(assem.neurons)
    print(io, "Assembly{$T} of $k neurons")
end

## BRAINAREAS
mutable struct BrainAreas{K, T}
    areas::Array{NeuralArea{T}, 1}
    names::Array{K, 1}
end

"""Returns the number of neural areas in a `BrainAreas` type."""
Base.length(ba::BrainAreas) = length(ba.areas)
Base.getindex(ba::BrainAreas, i::Int) = ba.areas[i]
function select(ba::BrainAreas{K, T}, k::K) where {K, T}
    i = find(ba.names == k)
    i == nothing && error("No brain area with name $k.")
    return ba.areas[i]
end

num_synapses(ba::BrainAreas{K, T}) where {K, T} = mapreduce(num_synapses, +, ba.areas)
num_neurons(ba::BrainAreas{K, T}) where {K, T} = mapreduce(length, +, ba.areas)

function Base.show(io::IO, ba::BrainAreas{K, T}) where {K, T}
    descr = "BrainAreas{$K, $T}: $(num_neurons(ba)) neurons $(num_synapses(ba)) synapses"
    print(io, descr)
    for (name, area) in zip(ba.names, ba.areas)
        print(io, "\n\t")
        print(io, "$name => ")
        print(io, area)
    end
end

function global_index(ba::BrainAreas, neuron::Neuron)
    return global_index(ba, neuron.area, neuron.idx)
end

function global_index(ba::BrainAreas, na::NeuralArea, idx::Int)
    i = findfirst([a == na for a in ba.areas])
    i == nothing && error("Given NeuralArea not found in BrainAreas")
    return global_index(ba, i, idx)
end

function global_index(ba::BrainAreas, area_idx::Int, idx::Int)
    global_idx = sum([length(a) for a in ba.areas[1:(area_idx-1)]]) + idx
    return global_idx
end

function local_index(ba::BrainAreas, global_idx::Int)
    area_sizes = [length(na) for na in ba.areas]
    area_adjusted_idxs = cumsum(area_sizes) .- global_idx
    area = findfirst(area_adjusted_idxs .>= 0)
    prior_areas = [area_sizes[j] for j in 1:(area - 1)]
    idx = length(prior_areas) == 0 ? global_idx : global_idx - sum(prior_areas)
    return area, idx
end

"""Access the `i`th neuron in the `BrainAreas`.

`Neurons` are assumed to be indexed consecutively along the `NeuralArea`s.
That is, if `a1` is a neural area with 200 `Neurons` and `a2` is a `NeuralArea`
with 100 `Neurons`, then if

    ba = BrainAreas(Dict(1 => a1, 2 => a2))

`getneuron(ba, 201)` will return the first neuron in `a2` (`a2[1]`) and
`getneuron(ba, 1)` will return the first neuron in `a1`.

"""
function getneuron(ba::BrainAreas, global_idx::Int)
    area, idx = local_index(ba, global_idx)
    return ba[area][idx]
end

"""Builds a network of brain areas from a `DiGraph`."""
function BrainAreas(
    g::DiGraph,
    area_sizes::Array{Int, 1},
    assembly_sizes::Array{Int, 1},
    plasticities::Array{T, 1},
) where {T, F}
    areas = [NeuralArea(k, β) for (k, β) in zip(assembly_sizes, plasticities)]
    for (a, sz) in zip(areas, area_sizes)
        a.neurons = [Neuron(a, i, empty_synapses(T)) for i in 1:sz]
    end
    brain_areas = BrainAreas(areas, collect(1:length(areas)))
    for e in edges(g)
        source = getneuron(brain_areas, e.src)
        target = getneuron(brain_areas, e.dst)
        source[target.area, target] = T(1)
    end
    return brain_areas
end

function Graphs.adjacency_matrix(ba::BrainAreas{K, T}) where {K, T}
    n = num_neurons(ba)
    adj = zeros(T, n, n)
    for a in ba.areas
        for neuron in a.neurons
            source_idx = global_index(ba, neuron)
            for area in keys(neuron.synapses)
                for (target_neuron, weight) in neuron[area]
                    target_idx = global_index(ba, target_neuron)
                    adj[target_idx, source_idx] = weight
                end
            end
        end
    end
    return adj
end

## ATTRIBUTIONGRAPH

mutable struct AttributionGraph{U}
    contributors::Dict{U, Array{U, 1}}
end

function neuron_attrib_graph(T::Type)
    AttributionGraph(
        Dict{Neuron{T, NeuralArea{T}}, Array{Neuron{T,  NeuralArea{T}}, 1}}()
    )
end

"""Get the list of neurons that fired onto a particular neuron."""
function Base.getindex(ag::AttributionGraph{N}, n::N) where N
    return ag.contributors[n]
end

"""Returns the list at key `n` or makes an empty list if no value is found."""
function Base.get!(ag::AttributionGraph{N}, n::N) where N
    return get!(ag.contributors, n, N[])
end

function Base.get(ag::AttributionGraph{N}, n::N, default) where N
    return get(ag.contributors, n, default)
end

## STOPCRITERIA

abstract type StopCriteria end

function (sc::StopCriteria)(
    areas::Array{NeuralArea{T}},
    currents::Array{Array{T, 1}}
) where T
    store!(sc, areas)
    return assess!(sc, areas, currents)
end

function store!(sc::StopCriteria, areas::Array{NeuralArea{T}, 1}) where T
    if sc.record
        firing = [reshape(a.firing, :, 1)  for a in areas]
        if sc.curr_iter == 0
            sc.spikes = firing
        else
            sc.spikes = map(hcat, sc.spikes, firing)
        end
    end
    sc.curr_iter += 1
end

mutable struct Converge <: StopCriteria
    curr_iter::Int
    max_iters::Int
    tol::Int
    fire_past_convergence::Int
    iters_convergent::Array{Int, 1}
    distances::Array{Array{Int, 2}}
    spikes::Array{Array{Int, 2}}
    record::Bool
end

function Converge(
    tol::Int = 2,
    max_iters::Int = 50,
    fire_past_convergence::Int = 0,
    record::Bool = true
)
    Converge(0, max_iters, tol, fire_past_convergence, zeros(1), [zeros(1, 1)], [zeros(1, 1)], record)
end

function assess!(
    cv::Converge,
    areas::Array{NeuralArea{T}},
    currents::Array{Array{T, 1}}
) where T
    if cv.curr_iter == 1
        cv.iters_convergent = zeros(length(areas))
        cv.distances = reshape([a.assembly_size for a in areas], :, 1)
    else
        distances = reshape([change_in_assembly(a) for a in areas], :, 1)
        hstack(cv.distances, distances)
    end
    i = cv.curr_iter
    # Locate areas where less that `tol` neurons changed from the previous step
    small_change = cv.distances[:, i] .< cv.tol
    # Increase the convergent iteration count for the convergent areas
    cv.iters_convergent[small_change] .+= 1
    # Set all non convergent areas iteration count to zero
    cv.iters_convergent[.~small_change] .= 0
    # Stop if all areas have been convergent for `fire_past_convergence` iters
    stop = all(iters_convergent >= (1 + cv.fire_past_convergence))
    # Check if max iters has been reached
    stop = stop || (sc.curr_iter >= sc.max_iters)
    return stop
end

mutable struct MaxIters <: StopCriteria
    curr_iter::Int
    max_iters::Int
    spikes::Array{Array{Int, 2}}
    record::Bool
end

function MaxIters(; max_iters::Int = 50, record::Bool = true)
    return MaxIters(0, max_iters, [zeros(1, 1)], record)
end

function assess!(
    mi::MaxIters,
    areas::Array{NeuralArea{T}, 1},
    currents::Array{Array{T, 1}}
) where T
    stop = mi.curr_iter >= mi.max_iters
    return stop
end
