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
       ("hmm_leaf_stay_leafturn", run_hmm_leaf_stay_leafturn),
       ("hmm_leaf_stay_leafspatial", run_hmm_leaf_stay_leafspatial),
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
       ("hmm_leaf_stay_leafturn_γ2", run_hmm_leaf_stay_leafturn_γ2),
       ("hmm_leaf_stay_leafspatial_γ2", run_hmm_leaf_stay_leafspatial_γ2),
       ("hmm_leaf_stay_turn_leafspatial_γ2", run_hmm_leaf_stay_turn_leafspatial_γ2),
       ("hmm_leaf_stay_turn_leafturn_γ2", run_hmm_leaf_stay_turn_leafturn_γ2),
       ("hmm_leaf_stay_spatial_leafspatial_γ2", run_hmm_leaf_stay_spatial_leafspatial_γ2),
       ("hmm_leaf_stay_spatial_leafturn_γ2", run_hmm_leaf_stay_spatial_leafturn_γ2),
]
i = parse(Int, ARGS[1])
(fn_ind, animal_ind) = fldmod1(i, length(animals))
animal = animals[animal_ind]
data = load_animal(animal, ".."; depletion=false)
(fname, fn) = fns[fn_ind]

println(animal)
println(fn)

results = fn(data; extended=true)
save("../results/hmm_biases/$(fname)_$(animal).jld2", "$(fname)_$(animal)", results; compress=true)

results = fn(data; extended=true, rewscaled=true)
save("../results/hmm_biases/$(fname)_rewscaled_$(animal).jld2", "$(fname)_rewscaled_$(animal)", results; compress=true)
