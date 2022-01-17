
mutable struct Neuron{T}
    neighbors = Array{Array{Dict{Int, T}, 1}, 1}
end

mutable struct NeuralArea{T}
    neurons::Array{Neuron{T}, 1}
    assembly_size::Int
    prev_firing::Array{Int, 1}
end

"""
    Base.length(na::NeuralArea) -> num_neurons

Number of neurons in a  `NeuralArea`.
"""
Base.length(na::NeuralArea) = length(na.neurons)

"""Iterate through neurons in a NeuralArea."""
Base.iterate(na::NeuralArea) = iterate(na.neurons)
Base.iterate(na.NeuralArea, i::Int64) = iterate(na.neurons, i)

mutable struct AssemblyComplex{T}
    areas::Array{NeuralArea{T}, 1}
    # Plasticity parameter
    β::T
end

"""
    Base.length(ac::AssemblyComplex) -> num_areas

Returns the number of neural areas in an `AssemblyComplex`.
"""
Base.length(ac::AssemblyComplex) = length(ac.areas)

mutable struct AttributionGraph
    contributors::Dict{Tuple{Int, Int}, Array{Tuple{Int, Int}, 1}}
end

"""
    compute_current_attribution(
        ac::AssemblyComplex{T},
        active_areas::Array{Int},
        firing::Array{Array{Int, 1}},
        input_currents::Array{Array{T, 1}}
    ) where T -> currents, attribution

Computes the ionic current flow into each neuron and creates an attribution
graph to be used for plasticity updates.

**Parameters**

1. The `ac` parameter is an `AssemblyComplex` containing all the neural areas

2. `active_areas` is a list of which areas are active during this firing period.
This must contain no repeats and all elements must be greater than zero and less
than `length(ac)`.

3. `firing` is an array containing arrays of indexes that denote which neurons
are currently firing in each active area. This array should have the same length
as `active_areas` and the array of indexes in `firing[i]` should correspond to
`active_areas[i]`. Therefore the indexes in `firing[i]` should represent
neurons in the corresponding `NeuronalArea`.

4. `input_currents` is an array of arrays of ionic currents flowing into the
neurons at this timestep. The array of values at `input_currents[i]` corresponds
to `active_areas[i]`. Then length of `input_currents[i]` should equal the number
of neurons in the corresponding `NeuralArea`.

"""
function compute_current_attribution(
    ac::AssemblyComplex{T},
    active_areas::Array{Int},
    firing::Array{Array{Int, 1}},
    input_currents::Array{Array{T, 1}}
) where T
    nothing
end

function firing_next(ac::AssemblyComples{T}, active_areas::Array{Int}, currents::Array{Array{T, 1}})
    nothing
end
