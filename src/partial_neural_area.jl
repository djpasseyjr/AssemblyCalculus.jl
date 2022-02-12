## PARTIALNEURALAREA
mutable struct PartialNeuralArea{T} <: BrainRegion{T}
    neurons::Array{Neuron{T, PartialNeuralArea{T}}, 1}
    assembly_size::Int
    firing::Array{Int, 1}
    firing_prev::Array{Int, 1}
    plasticity ::T
    synapse_prob::T
end

function PartialNeuralArea(T::Type)
    return PartialNeuralArea(Neuron{T, PartialNeuralArea{T}}[], 0, Int[], Int[], T(0), T(0))
end

function PartialNeuralArea(num_neurons::Int, assem_size::Int, plasticity::T, synapse_prob::T) where T
    area = PartialNeuralArea(Neuron{T, PartialNeuralArea{T}}[], assem_size, Int[], Int[], plasticity, synapse_prob)
    area.neurons = [Neuron(area, i) for i in 1:num_neurons]
    return area
end

"""Print format for `NeuralArea`."""
function Base.show(io::IO, area::PartialNeuralArea{T}) where T
    descr = "PartialNeuralArea{$T}: $(num_neurons(area)) neurons"
    print(io, descr)
end

"""Random stimulus currents into a PartialNeuralArea."""
random_current(area::PartialNeuralArea{T}) where T = random_current(area, p=area.synapse_prob)

"""Generate synapses into the target neural area, `area`. Each synapses between
`neuron` and a neuron in `area` is generated randomly with probability
`neuron.area.synapse_prob`.
"""
function generate_synapses(
    neuron::Neuron{T, PartialNeuralArea{T}},
    area::BrainRegion{T}
) where T
    mask = rand(Bernoulli(neuron.area.synapse_prob), length(area))
    target_neruons = area.neurons[mask]
    neuron.synapses[area] = Dict(target_neruons .=> ones(T, sum(mask)))
end

"""
    fire!(
        source_area::BrainRegion,
        target_area::BrainRegion,
        target_currents::Array{T, 1},
        attrib::AttributionGraph{Neuron{T, U}}
    ) where {T, U}

Fires each neuron in `source_area` into `target_area`. Adds resulting ion
currents to corresponding neuron indexes in `currents` array. Collects
neurons into the attribution graph.
"""
function fire!(
    source_area::PartialNeuralArea{T},
    target_area::BrainRegion{T},
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

"""Creates a `BrainAreas` type with generative synapses. That is, because the
majority of synapses are not used in a typical simulation, synapses are
generated as needed during the simulation.
"""
function BrainAreas(
    area_sizes::Array{Int, 1},
    assembly_sizes::Array{Int, 1},
    plasticities::Array{T, 1},
    synapse_probs::Array{T, 1}
) where T
    zipped_args = zip(area_sizes, assembly_sizes, plasticities, synapse_probs)
    areas = [PartialNeuralArea(args...) for args in zipped_args]
    brain_areas = BrainAreas(areas, collect(1:length(areas)))
    return brain_areas
end


"""Creates a `BrainAreas` type with generative synapses. That is, because the
majority of synapses are not used in a typical simulation, synapses are
generated as needed during the simulation.

This is the classic assembly calculus model where number of neurons `n`,
assembly size 'k', plasticity 'β' and synapse probability `p` is the same across
all areas.
"""
function BrainAreas(num_areas::Int, n::Int, k::Int, β::T, p::T) where T
    int_args = map(x -> repeat(x, num_areas), [[n], [k]])
    float_args = map(x -> repeat(x, num_areas), [[β], [p]])
    return BrainAreas(int_args..., float_args...)
end
