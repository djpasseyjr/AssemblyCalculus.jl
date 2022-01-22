"""Functions and types for fast Assembly Calculus simulations.

DJ Passey
Jan 2022
djpasseyjr@gmail.com
"""

mutable struct Neuron{T}
    id::Int
    area::NeuralArea{T}
    area_idx::Int
    synapses::Dict{NeuralArea{T}, Dict{Neuron{T}, T}}
end

Neuron(T::Type=Float64) = Neuron(0, NeuralArea(T), 0, T(0), Dict{NeuralArea{T}, Dict{Neuron{T}, T}}())
Base.getindex(n::Neuron{T}, na::NeuralArea{T}) where T = n.synapses[na]
edges(n::Neuron{T}, na::NeuralArea{T}) where T = length(n[na])

function Base.setindex!(
    n1::Neuron{T},
    w::T,
    na::NeuralArea,
    n2::Neuron{T}
) where T
    setindex!(n1.synapses[na], w, n2)
end

## NEURALAREA

mutable struct NeuralArea{T}
    neurons::Array{Neuron{T}, 1}
    assembly_size::Int
    firing::Array{Int, 1}
    firing_prev::Array{Int, 1}
    plasticity ::T
end

"""
    NeuralArea(type::Type=Float64) -> neural_area

Constructs an empty `NeuralArea`.
"""
NeuralArea(type::Type=Float64) = NeuralArea(Neuron{type}[], 0, Int[], Int[], type(0))

"""
    Base.length(na::NeuralArea) -> num_neurons

Number of neurons in a  `NeuralArea`.
"""
Base.length(na::NeuralArea) = length(na.neurons)

"""Iterate through neurons in a NeuralArea."""
Base.iterate(na::NeuralArea) = iterate(na.neurons)
Base.iterate(na.NeuralArea, i::Int64) = iterate(na.neurons, i)

"""Randomly assigns neurons in the area to fire."""
function random_firing!(area::NeuralArea{T}) where T
    area.firing = sample(1:length(area), area.assembly_size, replace=false)
end

function winners!(na::NeuralArea{T}, currents::Array{T}) where T
    na.firing_prev = na.firing
    na.firing = sortperm(
        currents,
        rev=true,
        alg=PartialQuickSort(na.assembly_size)
    )
end

function change_in_assembly(area::NeuralArea{T}) where T
    difference = area.assembly_size - intersect(area.firing, area.firing_prev)
end

## IONCURRENT

mutable struct IonCurrent{T} # Input current is better i think
    area::NeuralArea
    currents::Array{T, 1}
end

IonCurrent(type::Type=Float64) = IonCurrent(NeuralArea(type), type[])

function random_input_current(na::NeuralArea{T}) where T
    num_neurons = length(na)
    p = mapreduce(x -> edges(x, na), +, na.neurons) / (num_neurons ^ 2)
    dist = Binomial(na.assembly_size, p)
    currents = rand(dist, num_neurons)
    return IonCurrent(na, currents)
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
    parent_currents::Array{IonCurrent{T}, 1}
    parent_assemblies::Array{Assembly{T}, 1}
) where T
    reurn Assembly(na, na.firing, parent_currents, parent_assemblies)
end

## BRAINAREAS
mutable struct BrainAreas{T}
    areas::Dict{Any, NeuralArea{T}}
end

"""
    Base.length(ac::AssemblyComplex) -> num_areas

Returns the number of neural areas in an `AssemblyComplex`.
"""
Base.length(ac::BrainAreas) = length(ac.areas)

function BrainAreas(
    g::Graph,
    area_sizes::Array{Int, 1},
    assembly_sizes::Array{Int, 1}
    plasticities::Array{T, 1}
) where T
    nothing
end

## ATTRIBUTIONGRAPH

mutable struct AttributionGraph{T}
    contributors::Dict{Neuron{T}, Array{Neuron{T}, 1}}
end

AttributionGraph(T::Type=Float64) = AttributionGraph(Dict{Neuron{T}, Array{Neuron{T}, 1}}())

"""Get the list of neurons that fired onto a particular neuron."""
function Base.getindex(ag::AttributionGraph{T}, n::Neuron{T}) where T
    return ag.contributors[n]
end

function Base.get!(ag::AttributionGraph{T}, n::Neuron{T})
    return get!(ag.contributors, n, Neuron{T}[])
end

function Base.get(ag::AttributionGraph{T}, n::Neuron{T}, default)
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
    if sc.store
        firing = [reshape(a.firing, :, 1)  for a in areas]
        if sc.curr_iter == 0
            sc.spikes = firing
        else
            map(hcat, zip(sc.spikes, firing))
    end
    sc.curr_iter += 1
end

mutable struct Converge <: StopCritera
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
    max_iters::Int = 50
    fire_past_convergence::Int = 0,
    record::Bool = true
)
    Converge(0, max_iters, tol, fire_past_convergence, zeros(1), [zeros(1, 1)], [zeros(1, 1)], record)
end

function assess!(
    cv::Converge,
    areas::NeuralArea{T},
    currents::Array{Array{T, 1}}
) where T
    if cv.curr_iter == 1
        cv.iters_convergent = zeros(length(areas))
        cv.distances = reshape([a.assembly_size for a in areas], :, 1)
    else
        distances = reshape([change_in_assembly(a) for a in areas], : 1)
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

function MaxIters(max_iters::Int = 50, record::Bool = true)
    return MaxIters(0, max_iters, [zeros(1, 1)], record)
end

function assess!(
    cv::MaxIters,
    areas::NeuralArea{T},
    currents::Array{Array{T, 1}}
) where T
    stop = sc.curr_iter >= sc.max_iters
    return stop
end


## SIMULATION

function simulate!(
    inputs::Array{IonCurrent{T}, 1},
    assemblies::Array{Assembly{T}, 1}
    stop_criteria::StopCriteria
) where T
    active_areas = [inp.area for inp in inputs]
    # Consolidate input current with input current from assemblies
    currents, assem_attrib = simulation_currents(inputs, assemblies)
    # Save initial input currents
    init_currents = deepcopy(currents)
    # Assign random neurons to fire initially
    map(random_firing!, active_areas)
    # Allocate ion current arrays for each area
    next_currents = deepcopy(currents)
    # stop_criteria checks for convergence
    while not stop_criteria(active_areas, currents)
        # All neurons fire
        attrib = fire!(active_areas, next_currents)
        # Compute k winners
        map(winners!, zip(active_areas, next_currents))
        # Hebbian update for synapses
        map(area -> hebb_update!(area, attrib), active_areas)
        # Hebbian update for currents
        map(hebb_update!, zip(active_areas, currents))
        # Reset currents
        next_currents = deepcopy(currents)
    # Hebbian update to assembly synapses
    map(hebb_update!, zip(inputs, assem_attrib, currents, init_currents))
    final_assemblies = map(x -> Assembly(x, inputs, assemblies), active_areas)
    return inputs, final_assemblies
end

function simulate!(
    inputs::Array{IonCurrent{T}, 1},
    assemblies::Array{Assembly{T}, 1},
    timesteps::Int
) where T
    stop_crit = MaxIters(timesteps, record=true)
    new_assemblies = simulate!(inputs, assemblies, stop_crit)
    return new_assemblies, stop_crit.spikes, stop_crit.distances
end

function simulate!(
    inputs::Array{IonCurrent{T}, 1},
    assemblies::Array{Assembly{T}, 1},
    tol::Int = 2
    fire_past_convergence::Int = 0,
    max_iters::Int = 50,
    record = true
) where T
    stop_crit = Converge(tol, fire_past_convergence, max_iters, record)
    new_assemblies = simulate!(inputs, assemblies, stop_crit)
    if record
        return new_assemblies, stop_crit.spikes, stop_crit.distances
    else
        return new_assemblies
    end
end


"""
    simulation_currents(
        inputs::Array{IonCurrent{T}, 1},
        assemblies::Array{Assembly{T}, 1}
    ) where T -> currents, attribution

Computes the ionic current flow into each area and creates an attribution
graph to be used for plasticity updates.

**Parameters**

1. `inputs`
2. `assemblies`

"""
function simulation_currents(
    inputs::Array{IonCurrent{T}, 1},
    assemblies::Array{Assembly{T}, 1} = Assembly{T}[]
) where T
    # Allocate
    currents = [zeros(T, size(inp.currents)) for inp in inputs]
    attrib = AttributionGraph(T)
    # Compute currents and attributions
    for i, inp in enumerate(inputs)
        current[i] += inp.currents
        target_area = inp.area
        # Fire each assembly into the target area
        for assem in assemblies:
            fire!(assem, target_area, currents[i], attrib)
        end
    end
    return currents, attrib
end

## FIRE! FUNCTIONS

"""
    fire!(
        assembly::Assembly{T},
        area::NeuralArea{T},
        currents::Array{T, 1},
        attrib::AttributionGraph{T}
    ) where T

Fires each neuron in `assembly` into the target area. Adds resulting ion
currents to corresponding neuron indexes in `currents` array.
"""
function fire!(
    assembly::Assembly{T},
    area::NeuralArea{T},
    area_currents::Array{T, 1},
    attrib::AttributionGraph{T}
) where T
    # Fire each neuron in the assembly into the target area
    for idx in assembly.neurons:
        neuron = assembly.area.neurons[idx]
        fire!(neuron, area, currents, attrib)
    end
end

"""
    fire!(
        source_area::NeuralArea{T},
        target_area::NeuralArea{T},
        currents::Array{T, 1},
        attrib::AttributionGraph{T}
    ) where T

Fires each neuron in `source_area` into `target_area`. Adds resulting ion
currents to corresponding neuron indexes in `currents` array. Collects
neurons into the attribution graph.
"""
function fire!(
    source_area::NeuralArea{T},
    target_area::NeuralArea{T},
    current::Array{T, 1},
    attrib::AttributionGraph{T}
) where T
    # Fire each neuron in the assembly into the target area
    for idx in source_area.firing:
        neuron = source_area.neurons[idx]
        fire!(neuron, target_area, currents, attrib)
    end
end

"""
    fire!(
        neuron::Neuron{T},
        area::NeuralArea{T},
        area_currents::Array{T, 1},
        attrib::AttributionGraph{T}
    ) where T

Fires neuron into the target area. Adds resulting ion
currents to corresponding neuron indexes in `currents` array. Adds neruon
to the `AttributionGraph`.
"""
function fire!(neuron::Neuron{T}, area::NeuralArea, currents::Array{T, 1}, attrib::AttributionGraph)
    # For each synapse into the target area
    for target_neuron, w in neuron.synapses[area]
        # Add ion current to the current array
        currents[target_neuron.area_idx] += w
        # Update attribution graph entry for target neuron
        contribs = get!(attrib, target_neuron)
        push!(contribs, neuron)
    end
end

"""
    fire!(
        areas::Array{NeuralArea{T}, 1},
        currents::Array{Array{T, 1}}
    ) where T -> attrib

Fires all active neurons in each area. Collects ion currents in `currents`.
Returns an `AttributionGraph` of the firing.
"""

function fire!(areas::Array{NeuralArea{T}, 1}, currents::Array{Array{T, 1}}) where T
    attrib = AttributionGraph(T)
    for source_area in areas
        for target_area, current in zip(areas, currents)
            fire!(source_area, target_area, current, attrib)
        end
    end
    return attrib
end

## HEBB_UPDATE! FUNCTIONS

"""
    hebb_update!(neuron::Neuron{T}, attrib::AttributionGraph{T}) where T

Uses the attribution graph to find which synapses fired on to `neuron`
in the previous timestep. Increases these synapses multiplicatively according
to the plasticity of their parent neuron's area.
"""
function hebb_update!(neuron::Neuron{T}, attrib::AttributionGraph{T}) where T
    source_neurons = get(attrib, neuron, false)
    if source_neurons
        for source_neuron in source_neurons
            β = source_neuron.area.plasticity
            w = source_neuron[neuron.area][neuron]
            source_neuron[neuron.area][neuron] = w * (1 + β)
        end
    end
end

"""
    hebb_update!(area::NeuralArea{T}, attrib::AttributionGraph{T}) where T

Uses the attribution graph to update the synapses pointing to all currently
firing neurons in `area`.
"""
function hebb_update!(area::NeuralArea{T}, attrib::AttributionGraph{T}) where T
    map(n -> hebb_update!(n, attrib), area.neurons[area.firing])
end

"""
    hebb_update!(area::NeuralArea{T}, current::Array{T, 1}) where T

Multiplicatively increases the current into currently firing neurons in `area`
by a factor of `(1 + area.plasticity)`.
"""
function hebb_update!(area::NeuralArea{T}, current::Array{T, 1}) where T
    current[area.firing] .*= (1 + area.plasticity)
end

"""
    hebb_update!(
        input::IonCurrent{T},
        attrib::AttributionGraph{T},
        currents::Array{T, 1},
        init_currents::Array{T, 1}
    ) where T

Updates the assembly synapse weights all at once after the simulation is
finished.

Since the contribution of these synapses is included in the simulation current,
we can update the synapses once the simulation is finished based on changes in
the current array due to plasticity. This is faster than updating them each
timestep.
"""
function hebb_update!(
    input::IonCurrent{T},
    attrib::AttributionGraph{T},
    currents::Array{T, 1},
    init_currents::Array{T, 1}
) where T
    nonzero = findall(currents != 0.)
    growth = currents[nonzero] ./ init_currents[nonzero]
    growth_idx = nonzero[findall(growth > 1.)]
    # Update input
    input.current[growth_idx] .*= (1 + input.area.plasticity)
    # Update assembly synapses
    for idx in growth_idx
        neuron = input.area.neurons[idx]
        hebb_update!(neuron, attrib)
    end
end
