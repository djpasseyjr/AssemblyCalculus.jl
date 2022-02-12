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
Base.get!(n::Neuron{T, U}, k::U, default) where {T, U} = get!(n.synapses, k, default)

"""Returns all synapses in the grouping corresponding to the key `k`. If `k`
is not a valid key, return `default`.
"""
Base.get(n::Neuron{T, U}, k::U, default) where {T, U} = get(n.synapses, k, default)

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
Base.keys(n::Neuron{T, U}) where {T, U} = keys(n.synapses)

"""Check if a synapse grouping exists"""
Base.haskey(n::Neuron{T, U}, k::U) where {T, U} = haskey(n.synapses, k)

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
abstract type BrainRegion{T} end

mutable struct NeuralArea{T} <: BrainRegion{T}
    neurons::Array{Neuron{T, NeuralArea{T}}, 1}
    assembly_size::Int
    firing::Array{Int, 1}
    firing_prev::Array{Int, 1}
    plasticity ::T
end

# TODO I don't think this is needed
"""Constructs an empty `NeuralArea`."""
NeuralArea(T::Type) = NeuralArea(Neuron{T, NeuralArea{T}}[], 0, Int[], Int[], T(0))

"""Constructs an empty `NeuralArea` with `num_neurons` neurons, assembly
size `assem_size` and plasticity `plasticity`."""
function NeuralArea(num_neurons::Int, assem_size::Int, plasticity::T) where T
    area = NeuralArea(Neuron{T, NeuralArea{T}}[], assem_size, Int[], Int[], plasticity)
    area.neurons = [Neuron(area, i) for i in 1:num_neurons]
    return area
end

# TODO Reconsider this closure
"""Constructs empty synapses for a `Neuron` type with `NeuralArea` keys."""
empty_synapses(T::Type) = empty_synapses(NeuralArea{T}, T)

# TODO Reconsider this closure
"""Constructs and empty `Neuron` with `NeuralArea` synapse keys."""
function Neuron(T::Type=Float64)
    return Neuron(NeuralArea(T), 0, empty_synapses(T))
end

"""Initialize a neuron belonging to a particular area."""
function Neuron(area::BrainRegion{T}, i::Int) where T
    Neuron(area, i, empty_synapses(typeof(area), T))
end

"""Print format for `NeuralArea`."""
function Base.show(io::IO, na::NeuralArea{T}) where T
    descr = "NeuralArea{$T}: $(num_neurons(na)) neurons $(num_synapses(na)) synapses"
    print(io, descr)
end

"""Number of neurons in a  `NeuralArea`."""
Base.length(na::BrainRegion{T}) where T = length(na.neurons)

"""Iterate through neurons in a NeuralArea."""
Base.iterate(na::BrainRegion{T}) where T = iterate(na.neurons)
Base.iterate(na::BrainRegion{T}, i::Int64) where T = iterate(na.neurons, i)

"""Retrive neuron at index `i` in the neuron list."""
Base.getindex(na::BrainRegion{T}, i::Int) where T = na.neurons[i]

num_neurons(na::BrainRegion{T}) where T = length(na)
num_synapses(na::BrainRegion{T}) where T = length(na) == 0 ? 0 : mapreduce(num_synapses, +, na.neurons)

"""Randomly assigns neurons in the area to fire."""
function random_firing!(area::BrainRegion{T}) where T
    length(area) < area.assembly_size && error(string(
        "NeuralArea assembly size ($(area.assembly_size)) is larger than the ",
        "number of neurons in the area ($(length(area))."))
    area.firing = sample(1:length(area), area.assembly_size, replace=false)
end

"""Assigns the indexes of the k largest currents in `currents` to `na.firing`."""
function winners!(area::BrainRegion{T}, currents::Array{T}) where T
    # Validate
    length(area) != length(currents) && error(string(
        "Number of neurons in area ($(length(area))) must equal the number of ",
        "input currents ($(length(currents)))"))
    length(area) < area.assembly_size && error(string(
        "NeuralArea assembly size ($(area.assembly_size)) is larger than the ",
        "number of neurons in the area ($(length(area)))."))
    # Reassign currently firing neurons
    area.firing_prev = area.firing
    nonzero = findall(currents .> T(0))
    nnz = length(nonzero)
    # Check for positive nonzero currents
    if nnz == nothing
        # If no currents are positive, no neurons fire.
        area.firing = Int[]
    else
        # If the number of positive currents is smaller than the
        # assembly size, all neurons getting a positive current fire.
        if nnz < area.assembly_size
            area.firing = nonzero
        else
            # If the number of neurons getting positive current is
            # greater than the assembly size, only the neurons
            # receiving the largest current fire.
            partial_sort = sortperm(
                currents[nonzero],
                rev=true,
                alg=PartialQuickSort(area.assembly_size)
            )
            area.firing = nonzero[partial_sort[1:area.assembly_size]]
        end
    end
end

function change_in_assembly(area::BrainRegion{T}) where T
    difference = area.assembly_size - length(intersect(area.firing, area.firing_prev))
end


## IONCURRENT

# TODO rename to Stimulus

mutable struct IonCurrent{T}
    area::BrainRegion{T}
    currents::Array{T, 1}
end

IonCurrent(T::Type=Float64) = IonCurrent(NeuralArea(T), T[])

"""Constructs a random pattern of ionic currents corresponding to the
passed `NeuralArea`. Returns a `IonCurrent` type.
"""
function random_current(area::BrainRegion{T}; p::Union{Nothing, T} = nothing) where T
    n = num_neurons(area)
    if p == nothing
        p = mapreduce(x -> num_synapses(x, area), +, area.neurons) / (n ^ 2)
    elseif ! (0 < p < 1)
        error("Argument `p` must be `nothing`, or be in the interval (0, 1).")
    end
    currents = rand(Binomial(area.assembly_size, p), n)
    return IonCurrent(area, T.(currents))
end

""" Constructs an `IonCurrent` that sends zero current to all the neurons
in the passed `NeuralArea`.
"""
function zero_current(area::BrainRegion{T}) where T
    n = num_neurons(area)
    return IonCurrent(area, zeros(T, n))
end

function Base.show(io::IO, ic::IonCurrent{T}) where T
    n = length(ic.currents)
    print(io, "IonCurrent{$T} into $n neurons")
end

## ASSEMBLY

mutable struct Assembly{T}
    area::BrainRegion{T}
    neurons::Array{Int, 1}
    parent_currents::Array{IonCurrent{T}, 1}
    parent_assemblies::Array{Assembly{T}, 1}
end

Assembly(T::Type=Float64) = Assembly(NeuralArea(T), Int[], IonCurrent{T}[], Assembly{T}[])

function Assembly(
    na::BrainRegion{T},
    parent_currents::Array{IonCurrent{T}, 1},
    parent_assemblies::Array{Assembly{T}, 1},
) where T
    if length(na.firing) == 0
        error("No neurons firing in `NeuralArea`")
    end
    return Assembly(na, na.firing, parent_currents, parent_assemblies)
end

Assembly(area::BrainRegion{T}) where T = Assembly(area, IonCurrent{T}[], Assembly{T}[])

function Base.show(io::IO, assem::Assembly{T}) where T
    k = length(assem.neurons)
    print(io, "Assembly{$T} of $k neurons")
end


## BRAINAREAS
mutable struct BrainAreas{K, T}
    areas::Array{<:BrainRegion{T}, 1}
    names::Array{K, 1}
end

"""Returns the number of neural areas in a `BrainAreas` type."""
Base.length(ba::BrainAreas{K, T}) where {K, T} = length(ba.areas)

"""Returns the `i`th `NeuralArea` stored in `BrainAreas`.
"""
Base.getindex(ba::BrainAreas{K, T}, i::Int) where {K, T} = ba.areas[i]

"""Selects the `NeuralArea` corresponding to the given name."""
function select(ba::BrainAreas{K, T}, k::K) where {K, T}
    i = findfirst(ba.names .== k)
    i == nothing && error("No brain area with name $k.")
    return ba.areas[i]
end

"""Total number of synapses in `BrainAreas`."""
num_synapses(ba::BrainAreas{K, T}) where {K, T} = mapreduce(num_synapses, +, ba.areas)
"""Total number of neurons in `BrainAreas`."""
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

"""Returns the index of `neuron` with respect to all neurons in the `BrainAreas`.
"""
function globalindex(ba::BrainAreas{K, T}, neuron::Neuron{T}) where {K, T}
    return globalindex(ba, neuron.area, neuron.idx)
end

"""Returns the index of the neuron in `na` at index `idx` with respect to
all neurons in the `BrainAreas`.
"""
function globalindex(ba::BrainAreas{K, T}, na::BrainRegion{T}, idx::Int) where {K, T}
    i = findfirst([a == na for a in ba.areas])
    i == nothing && error("Given BrainRegion not found in BrainAreas")
    return globalindex(ba, i, idx)
end

"""Given the index, `area_idx`, for a `NeuralArea` within `ba`, and and index
`idx` of a neuron in the `NeuralArea` corresponding to `area_idx`, returns the
`index` of the neuron relative to all neurons in `ba`.
"""
function globalindex(ba::BrainAreas{K, T}, area_idx::Int, idx::Int) where {K, T}
    global_idx = sum([length(a) for a in ba.areas[1:(area_idx-1)]]) + idx
    return global_idx
end

"""For the given `BrainAreas`, and the index of a `Neuron` in the `BrainAreas`,
return the local index, `(area_idx, idx)`. That is, the index of the neruon's
parent area, `area_idx` (within `ba`) and the neuron's index within its
parent `NeuralArea`, `idx`."""
function localindex(ba::BrainAreas{K, T}, global_idx::Int) where {K, T}
    area_sizes = [length(na) for na in ba.areas]
    area_adjusted_idxs = cumsum(area_sizes) .- global_idx
    area_idx = findfirst(area_adjusted_idxs .>= 0)
    prior_areas = [area_sizes[j] for j in 1:(area_idx - 1)]
    idx = length(prior_areas) == 0 ? global_idx : global_idx - sum(prior_areas)
    return area_idx, idx
end

"""Access the `i`th neuron in the `BrainAreas`.

`Neurons` are assumed to be indexed consecutively along the `NeuralArea`s.
That is, if `a1` is a neural area with 200 `Neurons` and `a2` is a `NeuralArea`
with 100 `Neurons`, then if

    ba = BrainAreas([a1, a2], [1, 2])

`getneuron(ba, 201)` will return the first neuron in `a2` (`a2[1]`) and
`getneuron(ba, 1)` will return the first neuron in `a1`.

"""
function getneuron(ba::BrainAreas{K, T}, global_idx::Int) where {K, T}
    area, idx = localindex(ba, global_idx)
    return ba[area][idx]
end

"""Builds a network of brain areas from a `DiGraph`."""
function BrainAreas(
    g::DiGraph,
    area_sizes::Array{Int, 1},
    assembly_sizes::Array{Int, 1},
    plasticities::Array{T, 1},
) where T
    # Make neural areas
    areas = [NeuralArea(n, k, β) for (n, k, β) in zip(area_sizes, assembly_sizes, plasticities)]
    brain_areas = BrainAreas(areas, collect(1:length(areas)))
    for e in edges(g)
        source = getneuron(brain_areas, e.src)
        target = getneuron(brain_areas, e.dst)
        source[target.area, target] = T(1)
    end
    return brain_areas
end

"""Initializes `BrainAreas` from an adjacency matrix."""
function BrainAreas(
    adj::Array{T, 2},
    area_sizes::Array{Int, 1},
    assembly_sizes::Array{Int, 1},
    plasticities::Array{T, 1},
) where T
    return BrainAreas(DiGraph(transpose(adj)), area_sizes, assembly_sizes, plasticities)
end

"""Returns the adjacency matrix of a given `BrainAreas` type. This may be
incomplete if the `BrainAreas.areas` field contains `PartialNeuralAreas` because
some synapses may not be generated yet.
"""
function Graphs.adjacency_matrix(ba::BrainAreas{K, T}) where {K, T}
    n = num_neurons(ba)
    adj = zeros(T, n, n)
    for a in ba.areas
        for neuron in a.neurons
            source_idx = globalindex(ba, neuron)
            for area in keys(neuron.synapses)
                for (target_neuron, weight) in neuron[area]
                    target_idx = globalindex(ba, target_neuron)
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

function neuron_attrib_graph(T::Type, U::Type)
    AttributionGraph(
        Dict{Neuron{T, U}, Array{Neuron{T,  U}, 1}}()
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

"""Returns the value corresponding to key `n` or returns `default` if the key
is not found.
"""
function Base.get(ag::AttributionGraph{N}, n::N, default) where N
    return get(ag.contributors, n, default)
end

Base.length(ag::AttributionGraph{N}) where N = length(ag.contributors)
Base.values(ag::AttributionGraph{N}) where N = values(ag.contributors)
Base.keys(ag::AttributionGraph{N}) where N = keys(ag.contributors)

## STOPCRITERIA

"""StopCriteria types are used to determine when a simulation should end.

A StopCriteria subtype must have the follow fields

1. `spikes::Array{Array{Int, 2}}`

2. `record::Bool`

3. `curr_iter::Int`

It may have other fields.

Developers must implement a function

    assess!(
        sc::ConcreteStopCriteria,
        areas::Array{<:BrainRegion{T}},
        currents::Array{Array{T, 1}}
    ) where T

that accepts the `StopCriteria`, NeuralAreas involved in the simulation
and the `currents` going into the `NeuralArea` `Neurons` at the given point
in the simulation, and returns `true` if the simulation should stop.

At beginning of each step of in a simulation, the current state of the
active `NeuralAreas` and the ionic current are passed to a `StopCriteria` type.
The `assess!` function is used to determine if the simulation should stop or
continue for another timestep.
"""
abstract type StopCriteria end

"""Returns `true` if the stop criteria is met (implying that the assembly
calculus simulation should end). Optionally stores the currently firing neurons.
"""
function (sc::StopCriteria)(
    areas::Array{<:BrainRegion{T}},
    currents::Array{Array{T, 1}}
) where T
    store!(sc, areas)
    return assess!(sc, areas, currents)
end

"""Stores the currently firing neurons from each passed area inside the
`StopCriteria.spikes` field.
"""
function store!(sc::StopCriteria, areas::Array{<:BrainRegion{T}, 1}) where T
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

mutable struct MaxIters <: StopCriteria
    curr_iter::Int
    max_iters::Int
    spikes::Array{Array{Int, 2}}
    record::Bool
end

"""
    MaxIters(; max_iters::Int = 50, record::Bool = true)

Creates a MaxIters stop critera for assembly calculus simulations.

Passing a `MaxIters` object an `Array{<:BrainRegion{T}, 1}`,
and a `Array{Array{T, 1}}` will return `false` exactly `max_iters` times in a row
and will return `true` afterwards.

This is used to run a simulation for exactly `max_iters` timesteps. If
`record` is set to `true`, the `MaxIters` object will collect the neurons
firing in each `NeuralArea`, each time it is passed the arguments outlined above.

These neurons can be accesed via the `MaxIters.spikes` field.
"""
function MaxIters(; max_iters::Int = 50, record::Bool = true)
    return MaxIters(0, max_iters, [zeros(1, 1)], record)
end

function assess!(
    mi::MaxIters,
    areas::Array{<:BrainRegion{T}, 1},
    currents::Array{Array{T, 1}}
) where T
    stop = mi.curr_iter >= mi.max_iters + 1
    return stop
end

mutable struct Converge <: StopCriteria
    curr_iter::Int
    max_iters::Int
    tol::Int
    fire_past_convergence::Int
    iters_convergent::Array{Int, 1}
    distances::Array{Int, 2}
    spikes::Array{Array{Int, 2}}
    record::Bool
end

"""
    Converge(;
        tol::Int = 2,
        max_iters::Int = 50,
        fire_past_convergence::Int = 0,
        record::Bool = true
    )

Creates a `Converge` stop criteria for assembly calculus simulations.

Retains all functionality of the `MaxIters` stop criteria, but will end a
simulation early if "convergence" criteria is met. The convergence criteria
is determined by `tol` and `fire_past_convergence`.

Each time a `Converge` type is passed an `Array{<:BrainRegion{T}, 1}`
and an `Array{Array{T, 1}}`, it assesses the difference between each
`NeuralArea.firing` and `NeuralArea.firing_prev`. If the number of neurons that
appear in `NeuralArea.firing` but not in `NeuralArea.firing_prev` is less than
or equal to `tol`, the area is considered convergent. For example, if `tol=5`
and `NeuralArea.firing`only contains 3 neurons that are not in
`NeuralArea.firing_prev`, then that area is labeled as convergent.

The `fire_past_convergence` parameter will cause the simulation to continue
firing after all areas are convergent for the supplied number of timesteps.

If at any point, a previously convergent `NeuralArea` becomes dynamic again, the
`fire_past_convergence` count is reset to zero for that area.
"""
function Converge(;
    tol::Int = 2,
    max_iters::Int = 50,
    fire_past_convergence::Int = 0,
    record::Bool = true
)
    Converge(0, max_iters, tol, fire_past_convergence, zeros(1), zeros(1, 1), [zeros(1, 1)], record)
end

function assess!(
    cv::Converge,
    areas::Array{<:BrainRegion{T}},
    currents::Array{Array{T, 1}}
) where T
    if cv.curr_iter == 1
        cv.iters_convergent = zeros(length(areas))
        cv.distances = reshape([a.assembly_size for a in areas], :, 1)
    else
        distances = reshape([change_in_assembly(a) for a in areas], :, 1)
        cv.distances = hcat(cv.distances, distances)
    end
    i = cv.curr_iter
    # Locate areas where less that `tol` neurons changed from the previous step
    small_change = cv.distances[:, i] .<= cv.tol
    # Increase the convergent iteration count for the convergent areas
    cv.iters_convergent[small_change] .+= 1
    # Set all non convergent areas iteration count to zero
    cv.iters_convergent[.~small_change] .= 0
    # Stop if all areas have been convergent for `fire_past_convergence` iters
    stop = all(cv.iters_convergent .>= (1 + cv.fire_past_convergence))
    # Check if max iters has been reached
    stop = stop || (cv.curr_iter >= cv.max_iters + 1)
    return stop
end
