using StatsPlots
using FileIO
using EM
include("../code/hmm.jl")
include("../code/util.jl")
include("../code/em_scripts.jl")

fns = [
       ("hmm_leaf", run_hmm_leaf),
       ("hmm_leaf_γ2", run_hmm_leaf_γ2),
       ("hmm_leaf_retainbelief", run_hmm_leaf_retainbelief),
       ("hmm_leaf_stay", run_hmm_leaf_stay),
       ("hmm_leaf_turn", run_hmm_leaf_turn),
       ("hmm_leaf_spatial", run_hmm_leaf_spatial),
       ("hmm_leaf_leafspatial", run_hmm_leaf_leafspatial),
       ("hmm_leaf_leafturn", run_hmm_leaf_leafturn),

       ("hmm_leaf_stay_γ2", run_hmm_leaf_stay_γ2),
       ("hmm_leaf_stay_retainbelief", run_hmm_leaf_stay_retainbelief),
       ("hmm_leaf_stay_turn", run_hmm_leaf_stay_turn),
       ("hmm_leaf_stay_spatial", run_hmm_leaf_stay_spatial),
       ("hmm_leaf_stay_turn_leafspatial", run_hmm_leaf_stay_turn_leafspatial),
       ("hmm_leaf_stay_turn_leafturn", run_hmm_leaf_stay_turn_leafturn),
       ("hmm_leaf_stay_spatial_leafspatial", run_hmm_leaf_stay_spatial_leafspatial),
       ("hmm_leaf_stay_spatial_leafturn", run_hmm_leaf_stay_spatial_leafturn),

       ("hmm_leaf_turn_γ2", run_hmm_leaf_turn_γ2),
       ("hmm_leaf_spatial_γ2", run_hmm_leaf_spatial_γ2),
       ("hmm_leaf_turn_leafspatial_γ2", run_hmm_leaf_turn_leafspatial_γ2),
       ("hmm_leaf_turn_leafturn_γ2", run_hmm_leaf_turn_leafturn_γ2),
       ("hmm_leaf_spatial_leafspatial_γ2", run_hmm_leaf_spatial_leafspatial_γ2),
       ("hmm_leaf_spatial_leafturn_γ2", run_hmm_leaf_spatial_leafturn_γ2),

       ("hmm_leaf_stay_turn_γ2", run_hmm_leaf_stay_turn_γ2),
       ("hmm_leaf_stay_spatial_γ2", run_hmm_leaf_stay_spatial_γ2),
       ("hmm_leaf_stay_turn_leafspatial_γ2", run_hmm_leaf_stay_turn_leafspatial_γ2),
       ("hmm_leaf_stay_turn_leafturn_γ2", run_hmm_leaf_stay_turn_leafturn_γ2),
       ("hmm_leaf_stay_spatial_leafspatial_γ2", run_hmm_leaf_stay_spatial_leafspatial_γ2),
       ("hmm_leaf_stay_spatial_leafturn_γ2", run_hmm_leaf_stay_spatial_leafturn_γ2),

       ("hmm_leaf_depletion", run_hmm_leaf_depletion),
       ("hmm_leaf_γ2_depletion", run_hmm_leaf_γ2_depletion),
       ("hmm_leaf_retainbelief_depletion", run_hmm_leaf_retainbelief_depletion),
       ("hmm_leaf_stay_depletion", run_hmm_leaf_stay_depletion),
       ("hmm_leaf_turn_depletion", run_hmm_leaf_turn_depletion),
       ("hmm_leaf_spatial_depletion", run_hmm_leaf_spatial_depletion),
       ("hmm_leaf_leafspatial_depletion", run_hmm_leaf_leafspatial_depletion),
       ("hmm_leaf_leafturn_depletion", run_hmm_leaf_leafturn_depletion),

       ("hmm_leaf_stay_γ2_depletion", run_hmm_leaf_stay_γ2_depletion),
       ("hmm_leaf_stay_retainbelief_depletion", run_hmm_leaf_stay_retainbelief_depletion),
       ("hmm_leaf_stay_turn_depletion", run_hmm_leaf_stay_turn_depletion),
       ("hmm_leaf_stay_spatial_depletion", run_hmm_leaf_stay_spatial_depletion),
       ("hmm_leaf_stay_turn_leafspatial_depletion", run_hmm_leaf_stay_turn_leafspatial_depletion),
       ("hmm_leaf_stay_turn_leafturn_depletion", run_hmm_leaf_stay_turn_leafturn_depletion),
       ("hmm_leaf_stay_spatial_leafspatial_depletion", run_hmm_leaf_stay_spatial_leafspatial_depletion),
       ("hmm_leaf_stay_spatial_leafturn_depletion", run_hmm_leaf_stay_spatial_leafturn_depletion),

       ("hmm_leaf_turn_γ2_depletion", run_hmm_leaf_turn_γ2_depletion),
       ("hmm_leaf_spatial_γ2_depletion", run_hmm_leaf_spatial_γ2_depletion),
       ("hmm_leaf_turn_leafspatial_γ2_depletion", run_hmm_leaf_turn_leafspatial_γ2_depletion),
       ("hmm_leaf_turn_leafturn_γ2_depletion", run_hmm_leaf_turn_leafturn_γ2_depletion),
       ("hmm_leaf_spatial_leafspatial_γ2_depletion", run_hmm_leaf_spatial_leafspatial_γ2_depletion),
       ("hmm_leaf_spatial_leafturn_γ2_depletion", run_hmm_leaf_spatial_leafturn_γ2_depletion),

       ("hmm_leaf_stay_turn_γ2_depletion", run_hmm_leaf_stay_turn_γ2_depletion),
       ("hmm_leaf_stay_spatial_γ2_depletion", run_hmm_leaf_stay_spatial_γ2_depletion),
       ("hmm_leaf_stay_turn_leafspatial_γ2_depletion", run_hmm_leaf_stay_turn_leafspatial_γ2_depletion),
       ("hmm_leaf_stay_turn_leafturn_γ2_depletion", run_hmm_leaf_stay_turn_leafturn_γ2_depletion),
       ("hmm_leaf_stay_spatial_leafspatial_γ2_depletion", run_hmm_leaf_stay_spatial_leafspatial_γ2_depletion),
       ("hmm_leaf_stay_spatial_leafturn_γ2_depletion", run_hmm_leaf_stay_spatial_leafturn_γ2_depletion),
]
i = parse(Int, ARGS[1])
(fn_ind, animal_ind) = fldmod(i - 1, length(animals)) .+ 1
animal = animals[animal_ind]
data = load_animal(animal, ".."; depletion=true)
(fname, fn) = fns[fn_ind]

println(animal)
println(fn)

results = fn(data; extended=true)
save("../results/hmm_biases_depletion/$(fname)_$(animal).jld2", "$(fname)_$(animal)", results)

results = fn(data; extended=true, rewscaled=true)
save("../results/hmm_biases_depletion/$(fname)_rewscaled_$(animal).jld2", "$(fname)_rewscaled_$(animal)", results)
