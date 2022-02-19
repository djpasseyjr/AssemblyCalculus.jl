## PARTIALNEURALAREA
mutable struct PartialArea{T} <: BrainArea{T}
    neurons::Array{Neuron{T, PartialArea{T}}, 1}
    assembly_size::Int
    firing::Array{Int, 1}
    firing_prev::Array{Int, 1}
    plasticity ::T
    synapse_prob::T
end

function PartialArea(T::Type)
    return PartialArea(Neuron{T, PartialArea{T}}[], 0, Int[], Int[], T(0), T(0))
end

function PartialArea(num_neurons::Int, assem_size::Int, plasticity::T, synapse_prob::T) where T
    area = PartialArea(Neuron{T, PartialArea{T}}[], assem_size, Int[], Int[], plasticity, synapse_prob)
    area.neurons = [Neuron(area, i) for i in 1:num_neurons]
    return area
end

"""Print format for `NeuralArea`."""
function Base.show(io::IO, area::PartialArea{T}) where T
    descr = "PartialArea{$T}: $(num_neurons(area)) neurons"
    print(io, descr)
end

"""Random stimulus currents into a PartialArea."""
rand_stim(area::PartialArea{T}) where T = rand_stim(area, p=area.synapse_prob)

"""Generate synapses into the target neural area, `area`. Each synapses between
`neuron` and a neuron in `area` is generated randomly with probability
`neuron.area.synapse_prob`.
"""
function generate_synapses(
    neuron::Neuron{T, PartialArea{T}},
    area::BrainArea{T}
) where T
    mask = rand(Bernoulli(neuron.area.synapse_prob), length(area))
    target_neruons = area.neurons[mask]
    neuron.synapses[area] = Dict(target_neruons .=> ones(T, sum(mask)))
end

"""
    fire!(
        source_area::PartialArea,
        target_area::BrainArea,
        target_currents::Array{T, 1},
        attrib::AttributionGraph{Neuron{T, U}}
    ) where {T, U}

Fires each neuron in `source_area` into `target_area`. Adds resulting ion
currents to corresponding neuron indexes in `currents` array. Collects
neurons into the attribution graph.
"""
function fire!(
    source_area::PartialArea{T},
    target_area::BrainArea{T},
    target_currents::Array{T, 1},
    attrib::AttributionGraph{Neuron{T, U}}
) where {T, U}
    # Fire each neuron in the assembly into the target area
    for idx in source_area.firing
        neuron = source_area.neurons[idx]
        !haskey(neuron, target_area)  && generate_synapses(neuron, target_area)
        fire!(neuron, target_area, target_currents, attrib)
    end
end

"""
    fire!(
        assembly::Assembly{T},
        area::PartialArea{T},
        currents::Array{T, 1},
        attrib::AttributionGraph{Neuron{T, U}}
    ) where {T, U}

Fires each neuron in `assembly` into the target area. Adds resulting ion
currents to corresponding neuron indexes in `current` array.
"""
function fire!(
    assembly::Assembly{T},
    area::PartialArea{T},
    currents::Array{T, 1},
    attrib::AttributionGraph{Neuron{T, U}}
) where {T, U}
    # Fire each neuron in the assembly into the target area
    for idx in assembly.neurons
        neuron = assembly.area.neurons[idx]
        !haskey(neuron, area)  && generate_synapses(neuron, area)
        fire!(neuron, area, currents, attrib)
    end
end

"""Creates a `Brain` type with generative synapses. That is, because the
majority of synapses are not used in a typical simulation, synapses are
generated as needed during the simulation.
"""
function Brain(
    area_sizes::Array{Int, 1},
    assembly_sizes::Array{Int, 1},
    plasticities::Array{T, 1},
    synapse_probs::Array{T, 1}
) where T
    zipped_args = zip(area_sizes, assembly_sizes, plasticities, synapse_probs)
    areas = [PartialArea(args...) for args in zipped_args]
    brain_areas = Brain(areas, collect(1:length(areas)))
    return brain_areas
end


"""Creates a `Brain` type with generative synapses. That is, because the
majority of synapses are not used in a typical simulation, synapses are
generated as needed during the simulation.

This is the classic assembly calculus model where number of neurons `n`,
assembly size 'k', plasticity 'β' and synapse probability `p` is the same across
all areas.
"""
function Brain(;
    num_areas::Int = 1,
    n::Int = 10000,
    k::Int = 100,
    β::T = 0.01,
    p::T = 0.01
) where T
    int_args = map(x -> repeat(x, num_areas), [[n], [k]])
    float_args = map(x -> repeat(x, num_areas), [[β], [p]])
    return Brain(int_args..., float_args...)
end
