"""Functions to run firing and hebbian plasticity updates
on the `AssemblyCalculus` types defined in `types.jl`

Author: DJ Passey
Email: djpasseyjr@gmail.com
"""


## SIMULATION WRAPPERS

function simulate!(
    inputs::Array{IonCurrent{T}, 1},
    assemblies::Array{Assembly{T}, 1},
    stop_criteria::StopCriteria;
    random_initial::Bool = true
) where T
    # TODO Validate inputs and assemblies
    active_areas = [inp.area for inp in inputs]
    # Consolidate input current with current from assemblies
    currents, assem_attrib = simulation_currents(inputs, assemblies)
    # Save initial input currents
    init_currents = deepcopy(currents)
    # Assign random neurons to fire initially
    random_initial && map(random_firing!, active_areas)
    # Allocate ion current arrays for each area
    # TODO better name for next_currents (Stimulus!)
    next_currents = deepcopy(currents)
    # stop_criteria checks for convergence
    while ! stop_criteria(active_areas, currents)
        # All neurons fire
        attrib = fire!(active_areas, next_currents)
        # Compute k winners
        map(winners!, active_areas, next_currents)
        # Hebbian update for synapses
        map(area -> hebb_update!(area, attrib), active_areas)
        # Hebbian update for currents
        map(hebb_update!, active_areas, currents)
        # Reset currents
        next_currents = deepcopy(currents)
    end
    # Hebbian update to assembly synapses
    map(x -> hebb_update!(assem_attrib, x...), zip(inputs, currents, init_currents))
    final_assemblies = map(x -> Assembly(x, inputs, assemblies), active_areas)
    return final_assemblies
end

function simulate!(
    inputs::Array{IonCurrent{T}, 1},
    assemblies::Array{Assembly{T}, 1},
    timesteps::Int;
    random_initial::Bool = true
) where T
    stop_criteria = MaxIters(max_iters=timesteps, record=true)
    new_assemblies = simulate!(inputs, assemblies, stop_criteria, random_initial=random_initial)
    return new_assemblies, stop_criteria.spikes
end

function simulate!(
    inputs::Array{IonCurrent{T}, 1},
    assemblies::Array{Assembly{T}, 1};
    record = true,
    tol::Int = 2,
    fire_past_convergence::Int = 0,
    max_iters::Int = 50,
    random_initial::Bool = true
) where T
    stop_criteria = Converge(
        tol=tol,
        fire_past_convergence=fire_past_convergence,
        max_iters=max_iters,
        record=record
    )
    new_assemblies = simulate!(inputs, assemblies, stop_criteria, random_initial=random_initial)
    if record
        return new_assemblies, stop_criteria.spikes, stop_criteria.distances
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
graph to be used for ion current and assembly plasticity updates.

**Parameters**

1. `inputs` The list of IonCurrents to be used in the simulation
2. `assemblies` The list of active Assemblies for the simulation

"""
function simulation_currents(
    inputs::Array{IonCurrent{T}, 1},
    assemblies::Array{Assembly{T}, 1} = Assembly{T}[]
) where T
    # Allocate
    currents = [zeros(T, size(inp.currents)) for inp in inputs]
    attrib = neuron_attrib_graph(T, typeof(inputs[1].area))
    # Compute currents and attributions
    for (i, inp) in enumerate(inputs)
        currents[i] += inp.currents
        target_area = inp.area
        # Fire each assembly into the target area
        for assem in assemblies
            fire!(assem, target_area, currents[i], attrib)
        end
    end
    return currents, attrib
end

## FIRE! FUNCTIONS

"""
    fire!(
        assembly::Assembly{T},
        area::BrainRegion{T},
        currents::Array{T, 1},
        attrib::AttributionGraph{Neuron{T, U}}
    ) where {T, U}

Fires each neuron in `assembly` into the target area. Adds resulting ion
currents to corresponding neuron indexes in `current` array.
"""
function fire!(
    assembly::Assembly{T},
    area::BrainRegion{T},
    currents::Array{T, 1},
    attrib::AttributionGraph{Neuron{T, U}}
) where {T, U}
    # Fire each neuron in the assembly into the target area
    for idx in assembly.neurons
        neuron = assembly.area.neurons[idx]
        fire!(neuron, area, currents, attrib)
    end
end

"""
    fire!(
        source_area::BrainRegion{T},
        target_area::BrainRegion{T},
        target_currents::Array{T, 1},
        attrib::AttributionGraph{Neuron{T, U}}
    ) where {T, U}

Fires each neuron in `source_area` into `target_area`. Adds resulting ion
currents to corresponding neuron indexes in `currents` array. Collects
neurons into the attribution graph.
"""
function fire!(
    source_area::BrainRegion{T},
    target_area::BrainRegion{T},
    target_currents::Array{T, 1},
    attrib::AttributionGraph{Neuron{T, U}}
) where {T, U}
    # Fire each neuron in the assembly into the target area
    for idx in source_area.firing
        neuron = source_area.neurons[idx]
        fire!(neuron, target_area, target_currents, attrib)
    end
end

"""
    fire!(
        neuron::Neuron{T, U},
        target_area::BrainRegion{T},
        target_currents::Array{T, 1},
        attrib::AttributionGraph{Neuron{T, U}}
    ) where {T, U}

Fires neuron into the target area. Adds resulting ion
currents to corresponding neuron indexes in `currents` array. Adds neruon
to the `AttributionGraph`.
"""
function fire!(
    neuron::Neuron{T, U},
    target_area::BrainRegion{T},
    target_currents::Array{T, 1},
    attrib::AttributionGraph{Neuron{T, U}}
) where {T, U}
    # For each synapse into the target area
    for (target_neuron, w) in get(neuron, target_area, Dict{Neuron{T}, T}())
        # Add ion current to the current array
        target_currents[target_neuron.idx] += w
        # Update attribution graph entry for target neuron
        contributors = get!(attrib, target_neuron)
        push!(contributors, neuron)
    end
end


"""
    fire!(
        areas::Array{<:BrainRegion{T}, 1},
        area_currents::Array{Array{T, 1}}
    ) where T -> attrib

Fires all active neurons in each area. Collects ion currents in `currents`.
Returns an `AttributionGraph` of the firing.
"""

function fire!(areas::Array{U, 1}, area_currents::Array{Array{T, 1}}) where {T, U <:BrainRegion{T}}
    attrib = neuron_attrib_graph(T, U)
    for source_area in areas
        for (target_area, target_currents) in zip(areas, area_currents)
            fire!(source_area, target_area, target_currents, attrib)
        end
    end
    return attrib
end

## HEBB_UPDATE! FUNCTIONS

"""
    hebb_update!(
        neuron::Neuron{T, U},
        attrib::AttributionGraph{Neuron{T, U}};
        pow::Int=1
    ) where {T, U}

Uses the attribution graph to find which synapses fired on to `neuron`
in the previous timestep. Increases these synapses multiplicatively according
to the plasticity of their parent neuron's area. The parameter `pow` controls
how many multiplicative updates to apply.
"""
function hebb_update!(
    neuron::Neuron{T, U},
    attrib::AttributionGraph{Neuron{T, U}};
    pow::T = T(1)
) where {T, U}
    source_neurons = get(attrib, neuron, Neuron{T, U}[])
    if length(source_neurons) != 0
        for source_neuron in source_neurons
            β = source_neuron.area.plasticity
            w = source_neuron[neuron.area][neuron]
            source_neuron[neuron.area][neuron] = w * (1 + β)^pow
        end
    end
end

"""
    hebb_update!(area::BrainRegion{T}, attrib::AttributionGraph{Neuron{T, U}})

Uses the attribution graph to update the synapses pointing to all currently
firing neurons in `area`.
"""
function hebb_update!(area::BrainRegion{T}, attrib::AttributionGraph{Neuron{T, U}}) where {T, U}
    map(n -> hebb_update!(n, attrib), area.neurons[area.firing])
end

"""
    hebb_update!(area::BrainRegion{T}, current::Array{T, 1}) where T

Multiplicatively increases the current into currently firing neurons in `area`
by a factor of `(1 + area.plasticity)`.
"""
function hebb_update!(area::BrainRegion{T}, current::Array{T, 1}) where T
    current[area.firing] .*= (1 + area.plasticity)
end

"""
    hebb_update!(
        attrib::AttributionGraph{Neuron{T, U}},
        input::IonCurrent{T},
        currents::Array{T, 1},
        init_currents::Array{T, 1}
    ) where {T, U}

Updates the assembly synapse weights all at once after the simulation is
finished.

Since the contribution of these synapses is included in the simulation current,
we can update the synapses once the simulation is finished based on changes in
the current array due to plasticity. This is faster than updating them each
timestep.
"""
function hebb_update!(
    attrib::AttributionGraph{Neuron{T, U}},
    ic::IonCurrent{T},
    currents::Array{T, 1},
    init_currents::Array{T, 1},
) where {T, U}
    times_grown = _find_growth(currents, init_currents, ic.area.plasticity)
    # Update assembly synapses
    for (i, pow) in enumerate(times_grown)
        if pow > 0
            ic.currents[i] *= (1 + ic.area.plasticity) ^ pow
            neuron = ic.area.neurons[i]
            hebb_update!(neuron, attrib, pow=pow)
        end
    end
end

"""Computes the number of multiplicative updates applied to the ion
currents during the simulation.
"""
function _find_growth(
    currents::Array{T, 1},
    init_currents::Array{T, 1},
    plasticity::T
) where T
    # Find nonzero ion currents
    nonzero = findall(currents .!= T(0))
    nonzero == nothing && return zeros(T, size(currents))
    # Compute multiplicative change from the initial current
    change = currents[nonzero] ./ init_currents[nonzero]
    # Compute how many multiplicative updates occured given the plasticity
    nz_times_grown = log.(change) ./ log(1 + plasticity)
    times_grown = zeros(T, size(currents))
    times_grown[nonzero] .= nz_times_grown
    return times_grown
end
